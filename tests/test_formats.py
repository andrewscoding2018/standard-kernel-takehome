# Tests for fp4_kernel/formats.py  (Phase 1 — pure numpy, no torch / no GPU needed).
# These are the cheap, hand-checkable tests that lock the foundation. Get these green first.
import numpy as np
import pytest

from fp4_kernel.formats import FP4_E2M1, FP4_E3M0, FP4Format, generate_fp4_values


# ---- table correctness (deterministic, hand-checked) ----
def test_e2m1_table():
    # GOLDEN test (a "fact", not a roundtrip): the E2M1 grid is fixed by the spec,
    # so we assert the exact 16 values in CODE ORDER. We check the full *signed*
    # array (not just magnitudes) on purpose — that's what catches a missing-sign
    # bug, which a magnitude-only or positive-only roundtrip test would let through.
    #
    # ARRANGE: nothing to build — FP4_E2M1 is already constructed in formats.py,
    #          and its derived table lives on .code_to_value.
    # ACT:     none (the table is the thing under test).
    # ASSERT:  exact equality. Exact (not "allclose") is legitimate here because
    #          every E2M1 value (0.5, 1.5, 3, 6, ...) is exactly representable in
    #          binary float, so there's no rounding to tolerate.
    expected = np.array(
        [
            0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,   # codes 0-7  (sign bit = 0)
            0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,   # codes 8-15 (sign bit = 1)
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(FP4_E2M1.code_to_value, expected)


def test_e3m0_table():
    # zero + powers of two only (NO 1.5 / 3 / 6) — wider exponent range, no mantissa steps
    raise NotImplementedError


# ---- nearest_code: the rounding kernel ----
def test_nearest_code_exact_grid():
    # PROPERTY test — this is the "roundtrip" idea, anchored on the known grid points:
    # every representable grid value should snap to a code that decodes back to that
    # same value (grid points are fixed points of quantization).
    #
    # KEY SUBTLETY: we compare in VALUE space, not code space. Codes 0 and 8 BOTH
    # decode to 0.0 (+0 and -0), so nearest_code(0.0) is allowed to return either —
    # asserting "code maps to itself" would wrongly fail on the second zero. What we
    # actually require is that the value round-trips, and ±0 are the same value.
    fmt = FP4_E2M1
    grid = fmt.code_to_value                      # ARRANGE: the 16 representable values

    codes = fmt.nearest_code(grid)                # ACT: snap each grid value to a code
    decoded = fmt.code_to_value[codes]            # decode the chosen codes back to values

    # invariants worth pinning while we're here (cheap, and they catch dtype/range bugs):
    assert codes.dtype == np.uint8
    assert codes.min() >= 0 and codes.max() <= 15

    # ASSERT: grid values are fixed points in value space.
    np.testing.assert_array_equal(decoded, grid)


def test_nearest_code_rounds_to_nearer():
    # a value strictly between two grid points -> the closer one
    raise NotImplementedError


def test_nearest_code_clips_over_max():
    # |value| > max_value -> the max-magnitude code (clamp, no wraparound)
    raise NotImplementedError


def test_max_value():
    # FP4_E2M1.max_value() == 6.0
    raise NotImplementedError
