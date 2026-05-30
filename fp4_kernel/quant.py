
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
import torch.nn as nn
from dataclasses import dataclass
from .formats import FP4Format

@dataclass
class QuantizedTensor(nn.Module):

    def __init__(self, packed, scales, shape, block_size, fmt: FP4Format, scale_axis, padded_k, nbytes):
        super().__init__()
        self.packed = packed # uint8
        self.scales = scales # fp32 or fp16

    # packed codes in uint8
    # scales
    # original shape
    # block_size
    # format e2m1 or e3m0
    # scale axis

    # add padding to deal with sizing issues

    def quantize(self, weight, *, format, block_size, scale_mode) -> QuantizedTensor:
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

