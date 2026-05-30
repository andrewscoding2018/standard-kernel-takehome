# e2m1, e3m0, and metadat
#
# ===== TARGET SKELETON (reference — delete each line as you implement it) =====
# NOTE: formats.py is the BOTTOM of the dependency spine. It imports nothing from
#       quant.py. (quant imports FP4Format from here — not the reverse, or you get
#       a circular import.) Keep this module pure numpy.
#
# def generate_fp4_values(exp_bits, mantissa_bits, bias) -> list[float]:
#     """16-entry code->value table from the float decode rule.
#        field extract:  (code >> offset) & ((1 << width) - 1)
#          mantissa: offset 0,                 width mantissa_bits
#          exponent: offset mantissa_bits,     width exp_bits
#          sign:     offset mantissa_bits+exp_bits, width 1
#        normal  (exp!=0): 2**(exp-bias) * (1 + mant/2**mantissa_bits)
#        subnorm (exp==0): 2**(1-bias)  * (mant/2**mantissa_bits)"""
#
# class FP4Format:
#     name: str; exp_bits: int; mantissa_bits: int; bias: int
#     code_to_value: np.ndarray         # shape (16,), float — THE source of truth
#     def max_value(self) -> float:      # max(abs(code_to_value))
#     def nearest_code(self, scaled: np.ndarray) -> np.ndarray:   # uint8 codes 0..15
#         """PURE: inputs already divided by the block scale (in range).
#            argmin over |scaled[...,None] - table[None,:]|, return idx as uint8.
#            NO scale computation, NO quantize() call here."""
#
# FP4_E2M1 = FP4Format(name="e2m1", exp_bits=2, mantissa_bits=1, bias=1)
# FP4_E3M0 = FP4Format(name="e3m0", exp_bits=3, mantissa_bits=0, bias=<choose>)
# =============================================================================
from dataclasses import dataclass
import numpy as np
from .quant import quantize


def generate_fp4_values(exp_bits, mantissa_bits, bias):
    """SEEM layout with exponent bias = 1
    S     E    E      M
    |     |           |
    sign exponent mantissa

    """
    vals = []
    for code in range(16):
        sign = (code >> 3) & exp_bits # shifts right and then mask the last bit
        exp = (code >> mantissa_bits) &  # shifts right 1 and mask to get lowest 2 bits
        mant = code & 1 & bias # masks lowest bit of original

        if exp == 0: # subnormal: value = 0.5 * mant
            v = 0.5 * mant
        
        else: # normal: 2^(exp - 1) * (1 + 0.5 * mant)
            v = (2 ** (exp - 1)) * (1 + 0.5 * mant)
        
        vals.append(-v if sign else v)
    
    return vals

@dataclass
class FP4Format():

    def nearest_code(self, input, levels, fp4_max, fmt):
        vals = []
        self.lookup_table = generate_fp4_values(self.exp_bits, self.mantissa_bits, self.bias)


        output = []
        for x in input:
            output.append(quantize(x))

        return output