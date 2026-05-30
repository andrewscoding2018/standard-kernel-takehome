# to help with swapping in layers
#
# ===== TARGET SKELETON (reference — delete as you implement) =====
# class FakeQuantLinear(nn.Module):
#     """Wrap an nn.Linear: quantize->dequantize the weight, store back as BF16.
#        Isolates the ACCURACY question (does FP4 hurt the model?) from the
#        KERNEL question (is the fused path fast/correct?). swap_all uses THIS,
#        not QuantizedTensor directly — a QuantizedTensor isn't a forward-able layer."""
#     @classmethod
#     def from_linear(cls, linear, fmt, block_size, scale_mode) -> "FakeQuantLinear": ...
#     def forward(self, x) -> torch.Tensor:  # F.linear(x, self.dq_weight, self.bias)
#
# def swap_all(model, which, *, fmt, block_size, scale_mode) -> dict:   # returns originals
# def restore(model, originals) -> None
# =================================================================
from .quant import QuantizedTensor

def swap_all(model, block_size, which):
    """which is a list like ['up_proj', 'gate_proj', 'down_proj', 'q_proj', ...]"""
    originals = {}
    for i, layer in enumerate(model.model.layers):
        for name in which:
            for parent_name in ['mlp', 'self_attn']:
                parent = getattr(layer, parent_name, None)
                if parent is not None and hasattr(parent, name):
                    mod = getattr(parent, name)
                    originals[(i, parent_name, name)] = mod
                    setattr(parent, name, QuantizedTensor(mod, block_size))
    return originals

def restore(model, originals):
    for (i, p, n), mod in originals.items():
        setattr(getattr(model.model.layers[i], p), n, mod)
    