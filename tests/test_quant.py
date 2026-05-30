import torch
from fp4_kernel import quantize, dequantize

def roundtrip_error(W, *, format, block_size, scale_mode):
    """
    Verifies sure that the roundtrip from quantize -> dequantize works
    Returns error (instead of asserting)
    """

    