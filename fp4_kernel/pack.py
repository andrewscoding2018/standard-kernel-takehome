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
       Returns: uint8 array of shape (..., ceil(N/2)) packing two 4-bit codes per byte.

       Convention: low nibble holds the even index, high nibble the odd index, and we
       pack along the LAST axis so batched (multi-dim) input works. Odd N is padded with
       a zero nibble in the final high position (unpack slices it back off via `n`)."""

    if codes.dtype != np.uint8:
        raise TypeError(f"codes must be uint8, got {codes.dtype} instead")

    if (codes > 15).any():
        raise ValueError(f"codes must be [0, 15], got max={codes.max()}")

    N = codes.shape[-1]
    if N % 2:  # pad one zero nibble onto the last axis so N is even
        pad = [(0, 0)] * (codes.ndim - 1) + [(0, 1)]
        codes = np.pad(codes, pad)

    low = codes[..., 0::2]          # even indices -> low nibble
    high = codes[..., 1::2]         # odd indices  -> high nibble
    return (low | (high << 4)).astype(np.uint8)


def unpack_fp4(packed: np.ndarray, n: int) -> np.ndarray:
    """Inverse of pack_fp4. n is the original (possibly odd) length along the last axis.
       Returns uint8 [..., n], slicing off any zero-nibble pad that pack added."""

    low = packed & 0x0F
    high = (packed >> 4) & 0x0F
    # interleave low/high back into adjacent positions along the last axis
    out = np.stack([low, high], axis=-1).reshape(*packed.shape[:-1], -1)
    return out[..., :n].astype(np.uint8)