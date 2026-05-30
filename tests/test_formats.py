# Tests for fp4_kernel/formats.py  (Phase 1 — pure numpy, no torch / no GPU needed).
# These are the cheap, hand-checkable tests that lock the foundation. Get these green first.
import numpy as np
import pytest

from fp4_kernel.formats import FP4_E2M1, FP4_E3M0, generate_fp4_values


# ---- table correctness (deterministic, hand-checked) ----
def test_e2m1_table():
    # sorted unique magnitudes == [0, 0.5, 1, 1.5, 2, 3, 4, 6]
    raise NotImplementedError


def test_e3m0_table():
    # zero + powers of two only (NO 1.5 / 3 / 6) — wider exponent range, no mantissa steps
    raise NotImplementedError


# ---- nearest_code: the rounding kernel ----
def test_nearest_code_exact_grid():
    # every representable value maps back to its own code
    raise NotImplementedError


def test_nearest_code_rounds_to_nearer():
    # a value strictly between two grid points -> the closer one
    raise NotImplementedError


def test_nearest_code_clips_over_max():
    # |value| > max_value -> the max-magnitude code (clamp, no wraparound)
    raise NotImplementedError


def test_max_value():
    # FP4_E2M1.max_value() == 6.0
    raise NotImplementedError
