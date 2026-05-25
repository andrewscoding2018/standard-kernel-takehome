# quantize, dequantize, quantizedtensor

import pytest
import torch.nn as nn
import copy
from transformers import AutoTokenizer, AutoModelForCausalLM
import matplotlib.pyplot as plt
import numpy as np


class QuantizedTensor(nn.Module):

    def __init__(self):
        super().__init__()

    # packed codes in uint8
    # scales
    # original shape
    # block_size
    # format e2m1 or e3m0
    # scale axis

    # add padding to deal with sizing issues

    # 


def quantize():
    raise NotImplementedError

def dequantize():
    raise NotImplementedError

