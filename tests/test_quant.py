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

# def roundtrip_error(W, *, format, block_size, scale_mode):
#     """
#     verifies that the roundtrip from quantize -> dequantize works
#     Returns error (instead of asserting)
#     """

#     qt = quantize(W, format = format, block_size = block_size, scale_mode=scale_mode)
#     W_hat = dequantize(qt)
#     return (W_hat - W).norm() / W.norm()


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


def test_non_contiguous_input():
    raise NotImplementedError

def test_block_size_monotone():
    raise NotImplementedError

