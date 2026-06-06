# Tests for fp4_kernel/formats.py  (Phase 1 — pure numpy, no torch / no GPU needed).
# These are the cheap, hand-checkable tests that lock the foundation. Get these green first.
import numpy as np
import pytest

from fp4_kernel.formats import FP4Format, generate_fp4_values, FP4_E2M1, FP4_E3M0


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

    # With exp_bits=3, mantissa_bits=0, bias=1 the decode rule yields zero plus the
    # powers of two 2**0..2**6 in each sign half (16 entries total, code order).
    # NOTE: full 16-entry table — both code 0 and code 8 decode to 0.0 (+0 / -0).
    expected = np.array([
         0.0,  1.0,  2.0,  4.0,  8.0,  16.0,  32.0,  64.0,   # codes 0-7  (sign bit = 0)
         0.0, -1.0, -2.0, -4.0, -8.0, -16.0, -32.0, -64.0,   # codes 8-15 (sign bit = 1)
    ],
    dtype=np.float32,
    )
    np.testing.assert_array_equal(FP4_E3M0.code_to_value, expected)


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
    # PROPERTY test: a value strictly between two grid points snaps to the CLOSER one.
    # This is the complement of test_nearest_code_exact_grid (which only pins the grid
    # points themselves) — it's what proves nearest_code actually rounds rather than,
    # say, truncating toward the lower neighbor.
    #
    # ARRANGE: pick off-grid inputs that sit clearly (never at the midpoint) between two
    #          E2M1 grid values, so the expected neighbor is unambiguous and there's no
    #          tie-break to reason about.
    fmt = FP4_E2M1
    #            2.6    2.4    0.9    4.9    -2.6    -0.1
    # nearest:   3.0    2.0    1.0    4.0    -3.0     0.0   (grid: 0,.5,1,1.5,2,3,4,6)
    scaled = np.array([2.6, 2.4, 0.9, 4.9, -2.6, -0.1], dtype=np.float32)
    expected = np.array([3.0, 2.0, 1.0, 4.0, -3.0, 0.0], dtype=np.float32)

    # ACT + decode back to value space (codes 0/8 both decode to 0.0, so compare values).
    decoded = fmt.code_to_value[fmt.nearest_code(scaled)]

    # ASSERT: each input landed on its nearer neighbor.
    np.testing.assert_array_equal(decoded, expected)


def test_nearest_code_clips_over_max():
    # |value| > max_value -> the max-magnitude code (clamp, NOT wraparound).
    # The wraparound failure mode is the one to guard against: a buggy encoder that
    # overflows the field would send a huge value to a SMALL code (e.g. 100 -> 0.0).
    # Clamping is the correct behavior, and argmin over the table gives it for free.
    fmt = FP4_E2M1
    # Inputs are "a bit over" max in both directions. We deliberately do NOT test
    # astronomically large values (e.g. 1e9): nearest_code receives values already
    # divided by the block scale, so they're roughly in-range, and in float32
    # `1e9 - 6 == 1e9`, which would make every table entry tie and argmin fall back
    # to index 0 (=0.0). That's a float precision artifact of an out-of-contract
    # input, not the clamping behavior we're pinning here.
    scaled = np.array([100.0, -100.0, 6.5, -7.0], dtype=np.float32)

    decoded = fmt.code_to_value[fmt.nearest_code(scaled)]

    # ASSERT: everything saturated at +/- max_value with the right sign.
    expected = np.array([6.0, -6.0, 6.0, -6.0], dtype=np.float32)
    np.testing.assert_array_equal(decoded, expected)
    assert np.all(np.abs(decoded) == fmt.max_value)


def test_max_value():
    # max_value is the largest representable magnitude. It's a derived FIELD on the
    # frozen dataclass (computed in __post_init__), not a method — so no parens.
    assert FP4_E2M1.max_value == 6.0
    assert FP4_E3M0.max_value == 64.0
