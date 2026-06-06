# Model integration: swap an nn.Linear for a fake-quantized (qdq) version.
#
# We deliberately keep two questions separate:
#   ACCURACY  — does FP4 hurt the model?      -> FakeQuantLinear (this file)
#   KERNEL    — is the fused path fast/right?  -> fp4_linear / triton (linear.py)
#
# FakeQuantLinear quantizes the weight, immediately dequantizes it back to the
# layer's native dtype, and runs a plain F.linear. So the forward pass sees the
# *exact numerical error* FP4 introduces, with none of the kernel risk. That lets
# us measure MMLU / per-layer drift before trusting the fused kernel.
#
# A QuantizedTensor is a data holder, not an nn.Module — you can't forward through
# it — which is why swap_all installs FakeQuantLinear, not QuantizedTensor.
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import quantize, dequantize
from .formats import FP4Format, FP4_E2M1


class FakeQuantLinear(nn.Module):
    """Drop-in replacement for nn.Linear whose weight has been round-tripped
    through FP4 quantize -> dequantize. The dequantized weight is stored back in
    the layer's original dtype/device, so forward() is an ordinary matmul that
    carries the FP4 rounding error."""

    def __init__(self, dq_weight: torch.Tensor, bias, fmt, block_size, scale_mode):
        super().__init__()
        # buffer, not Parameter: this is inference-only, no grads, but we still
        # want it to move with .to()/.cuda() and show up in state_dict.
        self.register_buffer("dq_weight", dq_weight)
        self.bias = bias  # reuse the original (Parameter or None); not re-quantized
        self.out_features, self.in_features = dq_weight.shape
        # keep the recipe around for introspection / drift reporting
        self.fmt = fmt
        self.block_size = block_size
        self.scale_mode = scale_mode

    @classmethod
    def from_linear(cls, linear: nn.Linear, fmt: FP4Format = FP4_E2M1,
                    block_size: int = 32, scale_mode: str = "absmax") -> "FakeQuantLinear":
        w = linear.weight
        qt = quantize(w, format=fmt, block_size=block_size, scale_mode=scale_mode)
        # dequantize() returns a CPU float32 tensor (it round-trips through numpy);
        # cast/move it back to match the layer we're replacing.
        dq = dequantize(qt).to(dtype=w.dtype, device=w.device)
        return cls(dq, linear.bias, fmt, block_size, scale_mode)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.dq_weight, self.bias)

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias={self.bias is not None}, fmt={self.fmt.name}, "
                f"block_size={self.block_size}, scale_mode={self.scale_mode}")


def swap_all(model, which, *, fmt: FP4Format = FP4_E2M1,
             block_size: int = 32, scale_mode: str = "absmax") -> dict:
    """Replace the named projections in every decoder layer with FakeQuantLinear.

    which: list of submodule names, e.g. ['up_proj', 'gate_proj', 'down_proj']
           (MLP) or ['q_proj', 'k_proj', 'v_proj', 'o_proj'] (attention).

    Returns a dict of the original modules keyed by (layer_idx, parent, name) so
    restore() can put them back exactly.
    """
    originals = {}
    for i, layer in enumerate(model.model.layers):
        for name in which:
            for parent_name in ('mlp', 'self_attn'):
                parent = getattr(layer, parent_name, None)
                if parent is not None and hasattr(parent, name):
                    mod = getattr(parent, name)
                    originals[(i, parent_name, name)] = mod
                    setattr(parent, name, FakeQuantLinear.from_linear(
                        mod, fmt=fmt, block_size=block_size, scale_mode=scale_mode))
    return originals


def restore(model, originals) -> None:
    """Undo swap_all, putting every original module back in place."""
    for (i, parent_name, name), mod in originals.items():
        setattr(getattr(model.model.layers[i], parent_name), name, mod)
