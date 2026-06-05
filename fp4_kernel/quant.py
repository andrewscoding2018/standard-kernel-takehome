
# ===== TARGET SKELETON (reference — delete as you implement) =====
# @dataclass
# class QuantizedTensor:                # plain data holder; NOT an nn.Module
#     packed: torch.Tensor      # uint8, two 4-bit codes per byte
#     scales: torch.Tensor      # one per block (fp32 safe / bf16 = MXFP4-ish)
#     shape: tuple              # original weight shape (out_features, in_features)
#     block_size: int
#     fmt: FP4Format
#     scale_axis: int           # dim the blocks run along (K / in_features)
#     orig_k: int               # K before ragged padding, for slice-back
#
# def compute_scale(block, fmt, scale_mode) -> torch.Tensor:   # per-block; the 2nd requirement
#     # "absmax":     block.abs().amax(-1)        / fmt.max_value()
#     # "percentile": block.abs().quantile(0.999) / fmt.max_value()
#
# def quantize(weight, *, format, block_size, scale_mode) -> QuantizedTensor:
#     # 1. weight = weight.contiguous(); remember orig shape + K
#     # 2. pad K up to a multiple of block_size:  pad = (-K) % block_size
#     # 3. reshape (..., n_blocks, block_size); scale per block (compute_scale)
#     # 4. codes = format.nearest_code(block / scale); packed = pack_fp4(codes)
#     # 5. return QuantizedTensor(...)
#
# def dequantize(qweight: QuantizedTensor) -> torch.Tensor:    # single arg, per README
#     # unpack -> codes -> fmt.code_to_value[codes] * scale -> reshape -> slice to orig_k
# =================================================================
import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from .formats import FP4Format
from .pack import pack_fp4, unpack_fp4

@dataclass
class QuantizedTensor:
    packed: torch.Tensor # 
    scales: torch.Tensor #
    shape: tuple         # 
    block_size: int
    fmt: FP4Format
    scale_axis: int
    orig_k: int

def compute_scale(block, fmt, scale_mode) -> torch.Tensor:

    if scale_mode == "absmax":
        scale = block.abs().amax(-1) / fmt.max_value
    elif scale_mode == "percentile":
        scale = block.abs().quantile(0.999, dim = 1) / fmt.max_value
    else:
        raise ValueError(scale_mode)
    return scale.clamp_min(1e-12) # prevents all-zero block from div by 0 err


def quantize(weight, *, format: FP4Format, block_size, scale_mode) -> QuantizedTensor:
    weight = weight.detach().contiguous().float() # detach bc we dont' need grads
    # contiguous so that all of a stride is in memory
    orig_shape = tuple(weight.shape)
    K = weight.shape[-1]
    lead = weight.shape[:-1]

    # pad k up to multiple of block_size
    pad = (-K) % block_size
    if pad:
        weight = F.pad(weight, (0, pad))
    n_blocks = (K+pad) // block_size

    # reshape into blocks
    blocks = weight.reshape(*lead, n_blocks, block_size)

    # scale per block
    scale = compute_scale(blocks, format, scale_mode)
    scaled = blocks / scale.unsqueeze(-1)

    # perf downside when we move torch (gpu) -> numpy (cpu)
    codes = format.nearest_code(scaled.cpu().numpy())

    packed = torch.from_numpy(pack_fp4(codes))

    return QuantizedTensor(
        packed = packed, scales = scale,
        shape = orig_shape, block_size=block_size, fmt = format,
        scale_axis=weight.ndim - 1, orig_k=K
    )

def dequantize(qweight: QuantizedTensor) -> torch.Tensor:
    qt = qweight
    block_size = qt.block_size
    n_blocks = qt.scales.shape[-1]

    codes = unpack_fp4(qt.packed.cpu().numpy(), n= block_size)

    vals = qt.fmt.code_to_value[codes]

    vals = torch.from_numpy(vals).to(qt.scales.dtype)
    weight = vals * qt.scales.unsqueeze(-1)
    lead = weight.shape[:-2]
    weight = weight.reshape(*lead, n_blocks * block_size)

    return weight[..., :qt.orig_k]    
