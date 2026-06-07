# Tests for fp4_kernel/triton_kernels.py (the fused FP4 matmul, README requirement 2).
#
# The fused path needs CUDA + Triton, so the whole module skips on a CPU/Mac box.
# These define the contract the kernel must satisfy: it must agree with the naive
# dequantize-then-matmul oracle within a loose FP4 tolerance, across decode/prefill
# shapes, with/without bias, and for both formats.
import pytest
import torch

triton = pytest.importorskip("triton", reason="fused path needs Triton")
pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused path needs a CUDA device"
)

from fp4_kernel.quant import quantize, dequantize
from fp4_kernel.linear import fp4_linear
from fp4_kernel.triton_kernels import fp4_matmul
from fp4_kernel.formats import FP4_E2M1, FP4_E3M0


def _naive(x, qw, bias=None):
    W = dequantize(qw).to(device=x.device, dtype=x.dtype)
    return torch.nn.functional.linear(x, W, bias)


def _qweight(out_f, in_f, *, fmt=FP4_E2M1, block_size=32, seed=0):
    torch.manual_seed(seed)
    W = torch.randn(out_f, in_f, device="cuda")
    return W, quantize(W, format=fmt, block_size=block_size, scale_mode="absmax")


@pytest.mark.parametrize("M", [1, 8, 512])  # decode-like (M=1) through prefill-like
def test_fp4_matmul_matches_naive(M):
    W, qw = _qweight(128, 256)
    x = torch.randn(M, 256, device="cuda")

    out = fp4_matmul(x, qw.packed, qw.scales, qw.fmt.code_to_value, qw.block_size)
    ref = _naive(x, qw)

    assert out.shape == (M, 128)
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_fp4_matmul_with_bias():
    W, qw = _qweight(64, 128)
    x = torch.randn(16, 128, device="cuda")
    bias = torch.randn(64, device="cuda")

    out = fp4_matmul(x, qw.packed, qw.scales, qw.fmt.code_to_value, qw.block_size, bias=bias)
    ref = _naive(x, qw, bias)
    assert torch.allclose(out, ref, rtol=1e-2, atol=1e-2)


def test_fused_dispatch_matches_naive():
    # the linear.py dispatcher (fused=True) should route to the kernel and agree
    W, qw = _qweight(128, 256)
    x = torch.randn(16, 256, device="cuda")

    fused = fp4_linear(x, qw, fused=True)
    naive = fp4_linear(x, qw, fused=False)
    assert torch.allclose(fused, naive, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("fmt", [FP4_E2M1, FP4_E3M0])
def test_fp4_matmul_both_formats(fmt):
    W, qw = _qweight(64, 128, fmt=fmt)
    x = torch.randn(8, 128, device="cuda")
    out = fp4_matmul(x, qw.packed, qw.scales, qw.fmt.code_to_value, qw.block_size)
    assert torch.allclose(out, _naive(x, qw), rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("block_size", [16, 32, 64])
def test_fp4_matmul_block_sizes(block_size):
    W, qw = _qweight(96, 192, block_size=block_size)
    x = torch.randn(8, 192, device="cuda")
    out = fp4_matmul(x, qw.packed, qw.scales, qw.fmt.code_to_value, qw.block_size)
    assert torch.allclose(out, _naive(x, qw), rtol=2e-2, atol=2e-2)
