# Tests for fp4_kernel/pack.py  (README: "correctness of packing").
# Covers the nibble convention, odd-length pad/slice, batched (multi-dim) input,
# range checking, and randomized roundtrips. Mix of deterministic + randomized.
import numpy as np
import pytest

from fp4_kernel.pack import pack_fp4, unpack_fp4


def test_pack_hand_checked_byte():
    # convention: low nibble = even index, high nibble = odd index.
    # [0x0, 0xF] -> 0xF0,  [0xA, 0x3] -> 0x3A
    assert pack_fp4(np.array([0x0, 0xF], dtype=np.uint8))[0] == 0xF0
    assert pack_fp4(np.array([0xA, 0x3], dtype=np.uint8))[0] == 0x3A


def test_pack_roundtrip_deterministic():
    codes = np.array([0, 1, 2, 15, 8, 7], dtype=np.uint8)
    packed = pack_fp4(codes)
    assert packed.dtype == np.uint8
    assert packed.shape == (3,)  # ceil(6/2)
    np.testing.assert_array_equal(unpack_fp4(packed, n=len(codes)), codes)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 16, 31, 32, 100, 257])
def test_pack_roundtrip_random(n):
    rng = np.random.default_rng(seed=n)
    codes = rng.integers(0, 16, size=n, dtype=np.uint8)
    packed = pack_fp4(codes)
    assert packed.shape == ((n + 1) // 2,)
    np.testing.assert_array_equal(unpack_fp4(packed, n=n), codes)


@pytest.mark.parametrize("n", [1, 3, 5, 7, 33])
def test_pack_odd_length(n):
    # odd N must pad a zero nibble internally, then slice back off via `n`.
    rng = np.random.default_rng(seed=n)
    codes = rng.integers(0, 16, size=n, dtype=np.uint8)
    packed = pack_fp4(codes)
    out = unpack_fp4(packed, n=n)
    assert out.shape == (n,)
    np.testing.assert_array_equal(out, codes)


@pytest.mark.parametrize("shape", [(2, 4), (4, 2, 8), (3, 5), (8, 31)])
def test_pack_batched_multidim(shape):
    # packing runs along the LAST axis; leading axes are preserved.
    rng = np.random.default_rng(seed=hash(shape) % 2**32)
    codes = rng.integers(0, 16, size=shape, dtype=np.uint8)
    packed = pack_fp4(codes)
    assert packed.shape == (*shape[:-1], (shape[-1] + 1) // 2)
    out = unpack_fp4(packed, n=shape[-1])
    assert out.shape == shape
    np.testing.assert_array_equal(out, codes)


def test_pack_dtype_preserved():
    codes = np.array([[0, 1, 2, 3], [4, 5, 6, 7]], dtype=np.uint8)
    packed = pack_fp4(codes)
    assert packed.dtype == np.uint8
    assert unpack_fp4(packed, n=4).dtype == np.uint8


def test_pack_out_of_range_raises():
    with pytest.raises(ValueError):
        pack_fp4(np.array([14, 15, 16, 18], dtype=np.uint8))


def test_pack_wrong_dtype_raises():
    with pytest.raises(TypeError):
        pack_fp4(np.array([0, 1, 2, 3], dtype=np.int32))
