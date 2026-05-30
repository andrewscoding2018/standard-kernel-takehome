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
from fp4_kernel import quantize, dequantize

def roundtrip_error(W, *, format, block_size, scale_mode):
    """
    Verifies sure that the roundtrip from quantize -> dequantize works
    Returns error (instead of asserting)
    """

    