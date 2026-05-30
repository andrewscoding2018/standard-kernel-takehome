# pack_fp4 and unpack_fp4
#
# ===== TARGET SKELETON (reference) =====
# def pack_fp4(codes: np.ndarray) -> np.ndarray:
#     """uint8 [..., N] in [0,15]  ->  uint8 [..., ceil(N/2)], two codes/byte.
#        convention: low nibble = even index, high nibble = odd index.
#        TODO vs current impl: handle ODD N (pad a zero nibble), and
#        operate on the LAST axis so BATCHED 2D input works (test feeds 2D)."""
# def unpack_fp4(packed: np.ndarray, n: int) -> np.ndarray:
#     """inverse; return uint8 [..., n] (slice off the odd-N pad; force dtype uint8)."""
# =======================================
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