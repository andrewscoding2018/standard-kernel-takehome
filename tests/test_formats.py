import numpy as np
import pytest

from fp4_kernel.formats import FP4Format, generate_fp4_values, FP4_E2M1, FP4_E3M0


def test_e2m1_table():
    expected = np.array(
        [
            0.0,  0.5,  1.0,  1.5,  2.0,  3.0,  4.0,  6.0,   # codes 0-7  (sign bit = 0)
            0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,   # codes 8-15 (sign bit = 1)
        ],
        dtype=np.float32,
    )

    np.testing.assert_array_equal(FP4_E2M1.code_to_value, expected)


def test_e3m0_table():

    expected = np.array([
        0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16, # sig bit 0
        -0.125, -0.5, -1, -2.0, -4.0, -8.0, -16.0 # sign bit 1
    ], 
    dtype=np.float32
    )
    np.testing.assert_array_equal(FP4_E3M0.code_to_value, expected)


def test_nearest_code_exact_grid():
    fmt = FP4_E2M1
    grid = fmt.code_to_value 

    codes = fmt.nearest_code(grid)
    decoded = fmt.code_to_value[codes]
    assert codes.dtype == np.uint8
    assert codes.min() >= 0 and codes.max() <= 15

    np.testing.assert_array_equal(decoded, grid)


def test_nearest_code_rounds_to_nearer():

    fmt_E2M1 = FP4_E2M1
    fmt_E3M0 = FP4_E3M0
    scaled = np.array([2.6, 2.4, 0.9, 4.9, -2.6, -0.1], dtype=np.float32)
    expected = np.array([3.0, 2.0, 1.0, 4.0, -3.0, 0.0], dtype=np.float32)
    decoded_E2M1 = fmt_E2M1.code_to_value[fmt_E2M1.nearest_code(scaled)]
    decoded_E3M0 = fmt_E3M0.code_to_value[fmt_E2M1.nearest_code(scaled)]
    

    np.testing.assert_array_equal(decoded_E2M1, expected)
    np.testing.assert_array_equal(decoded_E3M0, expected)


def test_nearest_code_clips_over_max():
    # |value| > max_value -> the max-magnitude code (clamp, no wraparound)
    fmt = FP4_E2M1
    scaled = np.array([100.0, -100.0, 6.5, -7.0], dtype=np.float32)
    decoded = fmt.code_to_value[fmt.nearest_code(scaled)]
    expected = np.array([6.0, -6.0, 6.0, -6.0], dtype=np.float32)
    np.testing.assert_array_equal(decoded, expected)
    assert np.all(np.abs(decoded) == fmt.max_value)

def test_max_value():
    # FP4_E2M1.max_value() == 6.0
    assert FP4_E2M1.max_value == 6.0
    assert FP4_E3M0.max_value == 64.0
