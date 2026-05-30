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


def fp4_linear(x, qweight, bias = None) -> torch.Tensor:
    raise NotImplementedError