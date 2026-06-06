# Tests for fp4_kernel/linear.py  (Phase 4/6).
# The key one is fp4_linear vs a dequantized reference, within a loose FP4 tolerance.
import torch
import pytest
import torch.nn.functional as F

from fp4_kernel.quant import quantize, dequantize
from fp4_kernel.linear import fp4_linear
from fp4_kernel.formats import FP4_E2M1

def _naive_fp4_linear(x, qweight, bias):
    W = dequantize(qweight).to(device = x.device, dtype = x.dtype)
    return F.linear(x, W, bias)

def fp4_linear(x, qweight, bias = None, *, fused = False):
    if fused:
        return _fused_fp4_linear(x, qweight, bias)
    return _naive_fp4_linear

def test_fp4_linear_matches_naive_reference():
    # fp4_linear(x, qw) allclose to manual  x @ dequantize(qw).T  within a loose FP4 tol
    raise NotImplementedError


def test_fp4_linear_with_bias():
    # bias is added correctly
    raise NotImplementedError


def test_fp4_linear_decode_shape():
    # M = 1 (decode-like): single-token activation
    raise NotImplementedError


def test_fp4_linear_prefill_shape():
    # M large (prefill-like): batched activation
    raise NotImplementedError


@pytest.mark.skipif(not torch.cuda.is_available(), reason="fused path needs CUDA + Triton")
def test_fp4_linear_fused_matches_naive():
    # fused=True path agrees with the naive path
    raise NotImplementedError
