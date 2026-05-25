# e2m1, e3m0, and metadat
from dataclasses import dataclass


@dataclass
class FP4Format():
    def __init__(self, name, exp_bits, mantissa_bits, bias, code_to_value):
        self.name = name
        self.exp_bits = exp_bits
        self.mantissa_bits = mantissa_bits
        self.bias = bias
        self.code_to_value = code_to_value


class FP4_E2M1(FP4Format):
    # Instances: FP4_E2M1 (values ±{0, .5, 1, 1.5, 2, 3, 4, 6}) and FP4_E3M0 (powers of two).
    # nearest_code(values, fmt) -> uint8 codes  (vectorized round-to-nearest-even,
    # clip at ±max_value, sign bit handled explicitly so ±0 keep distinct codes).
    raise NotImplementedError


class FP4_E3M0(FP4Format):
    # nearest_code(values, fmt) -> uint8 codes  (vectorized round-to-nearest-even,
    # clip at ±max_value, sign bit handled explicitly so ±0 keep distinct codes).
    raise NotImplementedError