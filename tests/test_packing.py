import numpy as np
import pytest
from fp4_kernel.pack import pack_fp4, unpack_fp4

def test_pack_fp4():
    N = 20
    
    # identity
    test = np.array([0, 1, 2, 15, 8, 7], dtype = np.uint8)

    packed = pack_fp4(test)
    unpacked = unpack_fp4(packed, n = len(test))
    print("Test is:      ", test)
    print("Packed is:    ", packed)
    print("Unpacked is:  ", unpacked)
    assert np.array_equal(test, unpacked)


    # identity
    test2 = np.random.randint(0, 16, size= N, dtype = np.uint8)
    
    packed2 = pack_fp4(test2)
    unpacked2 = unpack_fp4(packed2, n = len(test2))
    print("Test is:      ", test2)
    print("Packed is:    ", packed2)
    print("Unpacked is:  ", unpacked2)
    assert np.array_equal(test2, unpacked2)

    
    # output shape and dtype
    test3 = np.random.randint(0, 16, size= N, dtype = np.uint8)
    for i in test3:
        assert i.dtype == np.uint8

    # out-of-range inputs
    with pytest.raises(ValueError):
        pack_fp4(np.array([14, 15, 16, 18], dtype = np.uint8))

        pack_fp4(np.array([256, 267, 268, 29], dtype = np.uint8))

    # batched input
    codes = np.array([
        [0, 1, 2, 3],
        [4, 5, 6, 7]],
        dtype = np.uint8
    )

    packed = pack_fp4(codes)
    print(packed)