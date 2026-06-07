# Tests for fp4_kernel/linear.py  (Phase 4/6).
# The key one is fp4_linear vs a dequantized reference, within a loose FP4 tolerance.
import torch
import pytest
import torch.nn.functional as F

from fp4_kernel.quant import quantize, dequantize
from fp4_kernel.linear import fp4_linear
from fp4_kernel.formats import FP4_E2M1


def _naive_fp4_linear(x, qweight, bias=None):
    """Hand-written oracle: dequantize, then a plain F.linear."""
    W = dequantize(qweight).to(device=x.device, dtype=x.dtype)
    return F.linear(x, W, bias)


def _make_qweight(out_features, in_features, *, block_size=32, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(out_features, in_features)
    qw = quantize(W, format=FP4_E2M1, block_size=block_size, scale_mode="absmax")
    return W, qw


def test_fp4_linear_matches_naive_reference():
    # fp4_linear(x, qw) allclose to manual  x @ dequantize(qw).T  within a loose FP4 tol
    W, qw = _make_qweight(64, 128)
    x = torch.randn(8, 128)

    out = fp4_linear(x, qw)
    ref = _naive_fp4_linear(x, qw)

    # same dequant-then-matmul path: should agree tightly
    assert out.shape == (8, 64)
    assert torch.allclose(out, ref, rtol=1e-5, atol=1e-5)

    # and the quantized result still tracks full-precision matmul within a loose FP4 tol
    full = F.linear(x, W)
    assert (out - full).norm() / full.norm() < 0.15


def test_fp4_linear_with_bias():
    # bias is added correctly
    W, qw = _make_qweight(32, 64)
    x = torch.randn(4, 64)
    bias = torch.randn(32)

    out = fp4_linear(x, qw, bias)

    # bias path equals no-bias path plus the bias, and matches the oracle
    assert torch.allclose(out, fp4_linear(x, qw) + bias, rtol=1e-5, atol=1e-5)
    assert torch.allclose(out, _naive_fp4_linear(x, qw, bias), rtol=1e-5, atol=1e-5)


def test_fp4_linear_decode_shape():
    # M = 1 (decode-like): single-token activation
    W, qw = _make_qweight(48, 96)
    x = torch.randn(1, 96)

    out = fp4_linear(x, qw)
    assert out.shape == (1, 48)
    assert torch.allclose(out, _naive_fp4_linear(x, qw), rtol=1e-5, atol=1e-5)


def test_fp4_linear_prefill_shape():
    # M large (prefill-like): batched activation
    W, qw = _make_qweight(48, 96)
    x = torch.randn(512, 96)

    out = fp4_linear(x, qw)
    assert out.shape == (512, 48)
    assert torch.allclose(out, _naive_fp4_linear(x, qw), rtol=1e-5, atol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused path needs CUDA + Triton")
def test_fp4_linear_fused_matches_naive():
    # fused=True path agrees with the naive path
    torch.manual_seed(0)
    W = torch.randn(128, 256, device="cuda")
    qw = quantize(W, format=FP4_E2M1, block_size=32, scale_mode="absmax")
    x = torch.randn(16, 256, device="cuda")

    fused = fp4_linear(x, qw, fused=True)
    naive = fp4_linear(x, qw, fused=False)
    assert torch.allclose(fused, naive, rtol=1e-2, atol=1e-2)
