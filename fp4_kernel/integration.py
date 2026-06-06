from .quant import QuantizedTensor, quantize, dequantize
from torch import nn
import torch.nn.functional as F
from .formats import FP4_E2M1, FP4_E3M0, FP4Format

class FakeQuantLinear(nn.Module):
    def __init__(self, dq_weight, bias, fmt, block_size, scale_mode):
        super().__init__()
        self.register_buffer("dq_weight", dq_weight) # buffer only since we're doing inference
        self.bias = bias
        self.fmt, self.block_size, self.scale_mode = fmt, block_size, scale_mode

    @classmethod
    def from_linear(cls, linear, fmt=FP4_E2M1, block_size=32, scale_mode="absmax"):
        w = linear.weight
        qt = quantize(w, format=fmt, block_size=block_size, scale_mode=scale_mode)
        dq = dequantize(qt).to(dtype=w.dtype, device=w.device)
        return cls(dq, linear.bias, fmt, block_size, scale_mode)

    def forward(self, x):
        return F.linear(x, self.dq_weight, self.bias)


def swap_all(model, which, *, fmt=FP4_E2M1, block_size=32, scale_mode="absmax"):
    originals = {}
    for i, layer in enumerate(model.model.layers):
        for name in which:
            for parent_name in ('mlp', 'self_attn'):
                parent = getattr(layer, parent_name, None)
                if parent is not None and hasattr(parent, name):
                    originals[(i, parent_name, name)] = getattr(parent, name)
                    setattr(parent, name, FakeQuantLinear.from_linear(
                        getattr(parent, name), fmt=fmt, block_size=block_size, scale_mode=scale_mode))
    return originals

def restore(model, originals):
    for (i, p, n), mod in originals.items():
        setattr(getattr(model.model.layers[i], p), n, mod)
    