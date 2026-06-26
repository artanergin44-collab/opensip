"""G.711 µ-law (PCMU) and A-law (PCMA) codecs.

Operates on 16-bit signed PCM, little-endian, mono.  Input/output is plain
``bytes`` so the rest of the library doesn't have to depend on numpy.

If numpy is available we use it for the hot path; otherwise we fall back to
pure-Python loops (slower but works everywhere).
"""

from __future__ import annotations

import struct

try:
    import numpy as _np  # type: ignore
    _HAVE_NUMPY = True
except ImportError:  # pragma: no cover
    _np = None
    _HAVE_NUMPY = False


# ---------------------------------------------------------------------------
# µ-law (PCMU, payload type 0)
# ---------------------------------------------------------------------------
_ULAW_BIAS = 0x84
_ULAW_CLIP = 32635


def _linear_to_ulaw_sample(pcm_val: int) -> int:
    """Encode one 16-bit linear PCM sample to µ-law (8-bit)."""
    if pcm_val < 0:
        sign = 0x80
        pcm_val = -pcm_val
    else:
        sign = 0
    if pcm_val > _ULAW_CLIP:
        pcm_val = _ULAW_CLIP
    pcm_val += _ULAW_BIAS
    seg = 0
    tmp = pcm_val >> 7
    while tmp:
        tmp >>= 1
        seg += 1
    # seg now in 0..8; clamp to 7 (µ-law has 8 segments)
    if seg > 7:
        seg = 7
    mantissa = (pcm_val >> (seg + 3)) & 0x0F
    ulaw = ~(sign | (seg << 4) | mantissa) & 0xFF
    return ulaw


def _ulaw_to_linear_sample(u_val: int) -> int:
    u_val = ~u_val & 0xFF
    sign = u_val & 0x80
    seg = (u_val >> 4) & 0x07
    mantissa = u_val & 0x0F
    sample = ((mantissa << 3) + _ULAW_BIAS) << seg
    sample -= _ULAW_BIAS
    return -sample if sign else sample


# ---------------------------------------------------------------------------
# A-law (PCMA, payload type 8)
# ---------------------------------------------------------------------------
def _linear_to_alaw_sample(pcm_val: int) -> int:
    if pcm_val < 0:
        sign = 0x00
        pcm_val = -pcm_val - 1
    else:
        sign = 0x80
    if pcm_val > 32767:
        pcm_val = 32767
    if pcm_val >= 256:
        seg = 1
        tmp = pcm_val >> 8
        while tmp:
            tmp >>= 1
            seg += 1
        seg -= 1
        mantissa = (pcm_val >> (seg + 3)) & 0x0F
        alaw = (seg << 4) | mantissa
    else:
        alaw = pcm_val >> 4
    return (alaw | sign) ^ 0x55


def _alaw_to_linear_sample(a_val: int) -> int:
    a_val ^= 0x55
    sign = a_val & 0x80
    seg = (a_val >> 4) & 0x07
    mantissa = a_val & 0x0F
    if seg:
        sample = ((mantissa << 4) + 0x108) << (seg - 1)
    else:
        sample = (mantissa << 4) + 8
    return -sample if not sign else sample


# Pre-computed translation tables. Indexed by the 16-bit little-endian uint
# interpretation of a signed PCM sample (so we can hand a memoryview/numpy
# array of uint16 directly into the lookup).
_LIN2ULAW = bytes(_linear_to_ulaw_sample(((s & 0xFFFF) - 0x10000) if s & 0x8000 else s)
                  for s in range(0x10000))
_ULAW2LIN = struct.pack("<256h", *[_ulaw_to_linear_sample(i) for i in range(256)])
_LIN2ALAW = bytes(_linear_to_alaw_sample(((s & 0xFFFF) - 0x10000) if s & 0x8000 else s)
                  for s in range(0x10000))
_ALAW2LIN = struct.pack("<256h", *[_alaw_to_linear_sample(i) for i in range(256)])

if _HAVE_NUMPY:
    # uint8 LUTs indexed by uint16 PCM sample → µ-law / A-law byte.
    _LIN2ULAW_NP = _np.frombuffer(_LIN2ULAW, dtype=_np.uint8)
    _LIN2ALAW_NP = _np.frombuffer(_LIN2ALAW, dtype=_np.uint8)
    # int16 LUTs indexed by µ-law / A-law byte → linear PCM sample.
    _ULAW2LIN_NP = _np.frombuffer(_ULAW2LIN, dtype=_np.int16)
    _ALAW2LIN_NP = _np.frombuffer(_ALAW2LIN, dtype=_np.int16)


# ---------------------------------------------------------------------------
# Public encode/decode helpers (bytes-in, bytes-out)
# ---------------------------------------------------------------------------
def pcm_to_ulaw(pcm: bytes) -> bytes:
    """Linear 16-bit PCM little-endian → µ-law bytes."""
    if len(pcm) % 2:
        pcm = pcm[:-1]
    if _HAVE_NUMPY:
        idx = _np.frombuffer(pcm, dtype="<u2")
        return _LIN2ULAW_NP[idx].tobytes()
    out = bytearray(len(pcm) // 2)
    for i in range(0, len(pcm), 2):
        lo = pcm[i]
        hi = pcm[i + 1]
        idx = (hi << 8) | lo
        out[i >> 1] = _LIN2ULAW[idx]
    return bytes(out)


def ulaw_to_pcm(ulaw: bytes) -> bytes:
    if _HAVE_NUMPY:
        idx = _np.frombuffer(ulaw, dtype=_np.uint8)
        return _ULAW2LIN_NP[idx].astype("<i2", copy=False).tobytes()
    out = bytearray(len(ulaw) * 2)
    for i, b in enumerate(ulaw):
        out[i * 2:i * 2 + 2] = _ULAW2LIN[b * 2:b * 2 + 2]
    return bytes(out)


def pcm_to_alaw(pcm: bytes) -> bytes:
    if len(pcm) % 2:
        pcm = pcm[:-1]
    if _HAVE_NUMPY:
        idx = _np.frombuffer(pcm, dtype="<u2")
        return _LIN2ALAW_NP[idx].tobytes()
    out = bytearray(len(pcm) // 2)
    for i in range(0, len(pcm), 2):
        idx = (pcm[i + 1] << 8) | pcm[i]
        out[i >> 1] = _LIN2ALAW[idx]
    return bytes(out)


def alaw_to_pcm(alaw: bytes) -> bytes:
    if _HAVE_NUMPY:
        idx = _np.frombuffer(alaw, dtype=_np.uint8)
        return _ALAW2LIN_NP[idx].astype("<i2", copy=False).tobytes()
    out = bytearray(len(alaw) * 2)
    for i, b in enumerate(alaw):
        out[i * 2:i * 2 + 2] = _ALAW2LIN[b * 2:b * 2 + 2]
    return bytes(out)


# ---------------------------------------------------------------------------
# Convenience: codec name → encode/decode function pair
# ---------------------------------------------------------------------------
def get_codec(name: str):
    """Return ``(encode, decode)`` callables for *name* (PCMU / PCMA)."""
    n = name.upper()
    if n == "PCMU":
        return pcm_to_ulaw, ulaw_to_pcm
    if n == "PCMA":
        return pcm_to_alaw, alaw_to_pcm
    raise ValueError(f"unsupported codec: {name!r}")


def silence_pcm(n_samples: int) -> bytes:
    """Return *n_samples* of 16-bit PCM silence."""
    return b"\x00\x00" * n_samples


__all__ = [
    "pcm_to_ulaw", "ulaw_to_pcm",
    "pcm_to_alaw", "alaw_to_pcm",
    "get_codec", "silence_pcm",
]
