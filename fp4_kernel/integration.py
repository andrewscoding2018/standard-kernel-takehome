# to help with swapping in layers
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
    