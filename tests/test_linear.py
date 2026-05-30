# Tests for fp4_kernel/linear.py  (Phase 4/6).
# The key one is fp4_linear vs a dequantized reference, within a loose FP4 tolerance.
import torch
import pytest

from fp4_kernel import quantize, dequantize, fp4_linear, FP4_E2M1


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
