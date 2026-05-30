
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

    def quantize(weight, *, format, block_size, scale_mode) -> QuantizedTensor:
        # calls .contiugous if neede
        # reshape to (..., n_blocks, block_size, scale_mode) -> pads K up to multiple of block size
        # codes = nearest code (weight / scale), format), pack

        raise NotImplementedError
        return QuantizedTensor()
    
    def dequantize(qt: QuantizedTensor, scale):
        # scale can either be `absmax` or `percentile
        if scale == "absmax":
            scale = block.abs().max() / fmt.max_value()
        if scale == "percentile":
            scale = quantile(block.abs(), 0.999) / fmt.max_value






def quantize():
    raise NotImplementedError

def dequantize():
    raise NotImplementedError

