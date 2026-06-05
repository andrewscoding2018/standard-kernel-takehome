# ===== RECOMMENDED TESTS (reference — README "quant/dequant error", both modes, ragged, non-contig) =====
# test_quant_dequant_error_bounded    # both formats: rel error under a loose FP4 tol
# test_scale_mode_absmax              # deterministic small block, hand-checked scale
# test_scale_mode_percentile          # 2nd mode runs + differs from absmax on tailed data
# test_ragged_K                       # K not a multiple of block_size (pad + slice back)
# test_non_contiguous_input           # W.t() / sliced view == contiguous result
# test_block_size_monotone            # smaller block_size -> lower error (seeded)
# (mix deterministic + randomized; seed the RNG)
# =======================================================================================================
import torch
from fp4_kernel.quant import quantize, dequantize, compute_scale
from fp4_kernel.formats import FP4_E2M1
import pytest

def roundtrip_error(W, *, format, block_size, scale_mode):
    """
    verifies that the roundtrip from quantize -> dequantize works
    Returns error (instead of asserting)
    """

    qt = quantize(W, format = format, block_size = block_size, scale_mode=scale_mode)
    W_hat = dequantize(qt)
    return (W_hat - W).norm() / W.norm()


# W = torch.randn(64, 128)
# err = roundtrip_error(W, format=FP4_E2M1, block_size=32, scale_mode="absmax")
# print(err)   # expect a small-ish number, ballpark 0.05–0.10 for E2M1 — NOT ~0, NOT ~1



def test_scale_mode_absmax():
    block = torch.tensor([[3.0, 1.5, -0.75, 0.0, 0.25, -2.0, 1.0, -3.0]])

    s = compute_scale(block, FP4_E2M1, "absmax")
    assert s.flatten().item() == 0.5

    qt = quantize(block, format = FP4_E2M1, block_size=8, scale_mode="absmax")
    assert torch.allclose(dequantize(qt), block)

def test_scale_mode_percentile():
    torch.manual_seed(0)
    block = torch.randn(1, 1024)
    block[0, 0] = 100.0

    s_absmax = compute_scale(block, FP4_E2M1, "absmax")
    s_pctl   = compute_scale(block, FP4_E2M1, "percentile") # ensure that percetile is skewed appropriately

    assert s_pctl < s_absmax / 5

@pytest.mark.parametrize("K", [70, 33, 16, 1000]) # none divisible by 12
def test_ragged_K(K):
    torch.manual_seed(0)
    W = torch.randn(48, K)
    qt = quantize(W, format = FP4_E2M1, block_size=32, scale_mode="absmax")
    W_hat = dequantize(qt)

    assert W_hat.shape == W.shape # make sure that orig_k slices the right amt
    assert (W_hat - W).norm() / W.norm() <0.15 # values should be real!

def test_non_contiguous_input():
    torch.manual_seed(0)
    base = torch.randn(128, 64)

    for W in (base.t(), base[:, ::2]):
        assert not W.is_contiguous()
        a = dequantize(quantize(W, format = FP4_E2M1, block_size=32, scale_mode="absmax"))
        b = dequantize(quantize(W.contiguous(), format = FP4_E2M1, block_size=32, scale_mode="absmax"))
        assert torch.allclose(a, b)
        assert a.shape == W.shape

def test_block_size_monotone():
    torch.manual_seed(0)
    W = torch.randn(64, 512)
    errs = [roundtrip_error(W, format = FP4_E2M1, block_size = bs, scale_mode = "absmax")
    for bs in (256, 128, 64, 32, 16)]

    for finer, coarser in zip(errs[1:], errs[:-1]):
        assert finer <= coarser + 1e-6
