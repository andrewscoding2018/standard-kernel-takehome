# e2m1, e3m0, and metadata
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
# =============================================================================
from dataclasses import dataclass, field
import numpy as np

# FP4_E2M1 = FP4Format(name="e2m1", exp_bits=2, mantissa_bits=1, bias=1)
# FP4_E3M0 = FP4Format(name="e3m0", exp_bits=3, mantissa_bits=0, bias=<choose>)


def generate_fp4_values(exp_bits, mantissa_bits, bias):
    """SEEM layout with exponent bias = 1
    S     E    E      M
    |     |           |
    sign exponent mantissa
    """

    vals = []
    for code in range(16):
        mant = (code >> 0) & ((1 << mantissa_bits) - 1)
        exp = (code >> mantissa_bits) & ((1 << exp_bits) - 1)
        sign = (code >> mantissa_bits + exp_bits) & ((1 << 1) - 1)

        v = 0.0
        if exp != 0:
            v = 2**(exp - bias) * (1 + mant/2**mantissa_bits)
        if exp == 0:
            v = 2**(1-bias) * (mant/2**mantissa_bits)
        if sign and v:
            v = -1 * v
        
        vals.append(v)
        
    return vals

@dataclass(frozen=True)
class FP4Format:
    name: str
    exp_bits: int
    mantissa_bits: int
    bias: int
    code_to_value: np.ndarray = field(init = False)
    max_value: float = field(init = False)

    def __post_init__(self):
        table = np.array(generate_fp4_values(self.exp_bits, self.mantissa_bits, self.bias), dtype = np.float32)
        object.__setattr__(self, "code_to_value", table)
        object.__setattr__(self, "max_value", float(np.abs(table).max()))

    def nearest_code(self, scaled):
        scaled = np.asarray(scaled)
        # scaled[..., None] -> broadcast each value against the 16 entry table along new axis
        # use argmin to pick nearest grid point (snapping)
        # clipping is automatic using idx
        idx = np.abs(scaled[..., None] - self.code_to_value).argmin(axis=-1)
        return idx.astype(np.uint8)