# Tests for fp4_kernel/integration.py.
# Uses a tiny Qwen-shaped stub model (model.model.layers[i].{mlp,self_attn}.<proj>)
# so we can exercise FakeQuantLinear / swap_all / restore without downloading weights.
import torch
import pytest
from torch import nn

from fp4_kernel.integration import FakeQuantLinear, swap_all, restore
from fp4_kernel.formats import FP4_E2M1, FP4_E3M0


class _Attn(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.q_proj = nn.Linear(d, d)
        self.k_proj = nn.Linear(d, d)
        self.v_proj = nn.Linear(d, d)
        self.o_proj = nn.Linear(d, d)


class _MLP(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.gate_proj = nn.Linear(d, h)
        self.up_proj = nn.Linear(d, h)
        self.down_proj = nn.Linear(h, d)


class _Layer(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.self_attn = _Attn(d)
        self.mlp = _MLP(d, h)


class _Inner(nn.Module):
    def __init__(self, n_layers, d, h):
        super().__init__()
        self.layers = nn.ModuleList(_Layer(d, h) for _ in range(n_layers))


class _Model(nn.Module):
    """Mimics the `model.model.layers[i].{mlp,self_attn}` attribute path."""

    def __init__(self, n_layers=3, d=64, h=128):
        super().__init__()
        self.model = _Inner(n_layers, d, h)


def _make_model(seed=0, **kw):
    torch.manual_seed(seed)
    return _Model(**kw)


# ----- FakeQuantLinear ---------------------------------------------------------

def test_fakequant_from_linear_shape_and_dtype():
    torch.manual_seed(0)
    lin = nn.Linear(96, 48)
    fq = FakeQuantLinear.from_linear(lin, fmt=FP4_E2M1, block_size=32, scale_mode="absmax")

    x = torch.randn(4, 96)
    out = fq(x)
    assert out.shape == (4, 48)
    # dequantized weight is stored back in the original dtype
    assert fq.dq_weight.dtype == lin.weight.dtype
    assert fq.dq_weight.shape == lin.weight.shape


def test_fakequant_tracks_fp32_linear_within_tol():
    torch.manual_seed(1)
    lin = nn.Linear(128, 64)
    fq = FakeQuantLinear.from_linear(lin, fmt=FP4_E2M1, block_size=32, scale_mode="absmax")

    x = torch.randn(8, 128)
    ref = lin(x)
    out = fq(x)
    assert (out - ref).norm() / ref.norm() < 0.15


def test_fakequant_preserves_bias():
    torch.manual_seed(2)
    lin = nn.Linear(64, 32, bias=True)
    fq = FakeQuantLinear.from_linear(lin)
    # bias object is carried through untouched
    assert fq.bias is lin.bias
    x = torch.randn(2, 64)
    no_bias = torch.nn.functional.linear(x, fq.dq_weight)
    assert torch.allclose(fq(x), no_bias + lin.bias, atol=1e-5)


# ----- swap_all / restore ------------------------------------------------------

def test_swap_all_replaces_named_projections():
    model = _make_model()
    originals = swap_all(model, ["q_proj", "gate_proj"], fmt=FP4_E2M1, block_size=32)

    for layer in model.model.layers:
        assert isinstance(layer.self_attn.q_proj, FakeQuantLinear)
        assert isinstance(layer.mlp.gate_proj, FakeQuantLinear)
        # untouched projections stay as plain Linear
        assert isinstance(layer.self_attn.k_proj, nn.Linear)
        assert not isinstance(layer.self_attn.k_proj, FakeQuantLinear)

    # one entry per (layer, parent, name) actually swapped
    assert len(originals) == len(model.model.layers) * 2


def test_swap_all_skips_missing_names():
    # a name that lives on neither mlp nor self_attn should be a no-op
    model = _make_model()
    originals = swap_all(model, ["does_not_exist"])
    assert originals == {}
    for layer in model.model.layers:
        assert isinstance(layer.self_attn.q_proj, nn.Linear)


def test_restore_round_trips_exactly():
    model = _make_model()
    before = {
        (i, p, n): getattr(getattr(layer, p), n)
        for i, layer in enumerate(model.model.layers)
        for p in ("mlp", "self_attn")
        for n in ("q_proj", "gate_proj")
        if hasattr(getattr(layer, p), n)
    }

    originals = swap_all(model, ["q_proj", "gate_proj"])
    restore(model, originals)

    # every restored module is the exact same object we started with
    for (i, p, n), mod in before.items():
        assert getattr(getattr(model.model.layers[i], p), n) is mod
        assert isinstance(mod, nn.Linear)


def test_swapped_model_output_drift_bounded():
    model = _make_model()
    d = model.model.layers[0].self_attn.q_proj.in_features
    x = torch.randn(2, d)

    # drive a single q_proj before and after swapping to measure local drift
    layer = model.model.layers[0]
    ref = layer.self_attn.q_proj(x)

    swap_all(model, ["q_proj"], fmt=FP4_E2M1, block_size=32)
    quant_out = model.model.layers[0].self_attn.q_proj(x)

    assert quant_out.shape == ref.shape
    assert (quant_out - ref).norm() / ref.norm() < 0.15


@pytest.mark.parametrize("fmt", [FP4_E2M1, FP4_E3M0])
def test_swap_all_runs_for_both_formats(fmt):
    model = _make_model()
    in_f = model.model.layers[0].mlp.up_proj.in_features  # read before swapping
    swap_all(model, ["q_proj", "up_proj"], fmt=fmt, block_size=16, scale_mode="absmax")
    up = model.model.layers[0].mlp.up_proj
    out = up(torch.randn(2, in_f))
    assert out.shape[-1] == up.dq_weight.shape[0]
