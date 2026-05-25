# pack_fp4 and unpack_fp4
import numpy as np

def pack_fp4(codes: np.ndarray) -> np.ndarray:
    """codes: uint8 array of shape (..., N) with values in [0, 15].
       Returns: uint8 array of shape (..., ceil(N/2)) packing two 4-bit codes per byte."""
    
    if codes.dtype != np.uint8:
        raise TypeError(f"codes must be uint8, got {codes.dtype} instead")

    if (codes > 15).any():
            raise ValueError(f"codes must be [0, 15], got max={codes.max()}")
    
    res = []
    for i in range(0, len(codes), 2):
        a, b = codes[i], codes[i+1]

        new_byte = a | (b << 4)
        res.append(new_byte)

    return np.array(res, dtype=np.uint8)
    

def unpack_fp4(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse of pack_fp4. n is the original (possibly odd) length along the last axis."""

    new = []

    for byte in packed:
        low_nibble = byte & 0x0F
        high_nibble = (byte >> 4) & 0x0F
        new.extend([low_nibble, high_nibble])

    return np.array(new)