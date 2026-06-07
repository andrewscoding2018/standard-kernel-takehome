# fp4_linear (naive and optimized)
#
# ===== TARGET SKELETON (reference — delete as you implement) =====
# def fp4_linear(x, qweight: QuantizedTensor, bias=None, *, fused=False) -> torch.Tensor:
#     """Dispatch: fused -> _fused_fp4_linear, else _naive_fp4_linear."""
# def _naive_fp4_linear(x, qweight, bias) -> torch.Tensor:
#     """Reference path: W = dequantize(qweight); return x @ W.T (+ bias).
#        This is BOTH the correctness oracle and the slow benchmark baseline."""
# def _fused_fp4_linear(x, qweight, bias) -> torch.Tensor:
#     """Fast path: call the Triton kernel; never materialize dequantized W in HBM."""
# =================================================================

import torch
import torch.nn.functional as F

from .quant import dequantize


def fp4_linear(x, qweight, bias=None, *, fused=False) -> torch.Tensor:
    """Dispatch: fused -> _fused_fp4_linear, else _naive_fp4_linear."""
    if fused:
        return _fused_fp4_linear(x, qweight, bias)
    return _naive_fp4_linear(x, qweight, bias)


def _naive_fp4_linear(x, qweight, bias=None) -> torch.Tensor:
    """Reference path: W = dequantize(qweight); return x @ W.T (+ bias).
    This is BOTH the correctness oracle and the slow benchmark baseline."""
    W = dequantize(qweight).to(device=x.device, dtype=x.dtype)
    return F.linear(x, W, bias)


def _fused_fp4_linear(x, qweight, bias=None) -> torch.Tensor:
    """Fast path: call the Triton kernel; never materialize dequantized W in HBM."""
    from .triton_kernels import fp4_matmul

    return fp4_matmul(
        x,
        qweight.packed,
        qweight.scales,
        qweight.fmt.code_to_value,
        qweight.block_size,
        bias=bias,
    )