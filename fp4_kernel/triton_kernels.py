# fused kernel (Phase 5)
#
# Build the NAIVE path in linear.py first — it's the correctness oracle this diffs against.
# On a Mac/CPU box Triton won't run; keep the triton imports commented so the package
# still imports, and develop/test this on a CUDA machine.

import torch

import triton
import triton.language as tl


@triton.jit
def fp4_matmul_kernel(
    x_ptr, packed_ptr, scales_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xk,
    stride_wn, stride_wk,
    code_to_value_ptr,            # 16-entry LUT (constant / shared mem)
    BLOCK_SIZE: tl.constexpr,     # quant block size, for indexing scales
    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr,
):
    """Load packed uint8 weight tiles, unpack two nibbles -> codes -> LUT -> * scale,
       accumulate in fp32. The win: never write a dequantized BF16 W back to HBM."""
    
    




def fp4_matmul(x, packed, scales, code_to_value, block_size, bias=None):
    """Python wrapper: allocate output, compute the launch grid, launch fp4_matmul_kernel."""
    raise NotImplementedError
