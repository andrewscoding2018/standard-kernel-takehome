
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
import torch.nn as nn
import numpy as np
from dataclasses import dataclass
from .formats import FP4Format

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




def quantize(self, weight, *, format, block_size, scale_mode):
    # calls .contiugous if neede
    # reshape to (..., n_blocks, block_size, scale_mode) -> pads K up to multiple of block size
    # codes = nearest code (weight / scale), format), pack


    scale = np.abs(x).max() / fp4_max # stretch data into our FP4 grid
    if scale == 0: scale = 1.0
    scaled = x / scale # round each scaled value to nearest representable FP4 level
    idx = np.abs(scaled[:, None] - levels[None, :]).argmin(axis=1)
    output.append([levels[idx], scale])

    return QuantizedTensor()

def dequantize(self, qt: QuantizedTensor, scale):
    # scale can either be `absmax` or `percentile
    if scale == "absmax":
        scale = block.abs().max() / fmt.max_value()
    if scale == "percentile":
        scale = quantile(block.abs(), 0.999) / fmt.max_value


def quantize():
    raise NotImplementedError

def dequantize():
    raise NotImplementedError

