"""Fixed-target jitter buffer for inbound RTP audio.

Caller pushes decoded PCM frames as they arrive (out-of-order ok) and pops
one ptime worth of audio per tick. Behaviour:

* Sequence numbers are RFC 1982 mod-2^16 compared.
* A short "prime" phase fills the buffer to ``target_ms`` before playout starts;
  this absorbs network jitter at the cost of a one-time delay.
* Sequence gaps produce :func:`silence_pcm` so playout stays on a steady clock.
* Packets arriving after their play slot has passed are dropped.
* A far-future sequence jump (>= 200 frames) is treated as a stream
  discontinuity and triggers a resync.

Timing lives in the caller (typically :class:`opensip.rtp.RTPSession`'s player
loop ticking once per ptime). This module is intentionally a plain data
structure with no asyncio dependency, so it can be unit-tested directly.
"""

from __future__ import annotations

import logging
import time

from .codecs import silence_pcm

log = logging.getLogger("opensip.jitter")

SEQ_MOD = 1 << 16
SEQ_HALF = 1 << 15


def _signed_seq_diff(a: int, b: int) -> int:
    """Return ``a - b`` as a signed 16-bit serial-number difference."""
    diff = (a - b) & 0xFFFF
    if diff >= SEQ_HALF:
        diff -= SEQ_MOD
    return diff


class JitterBuffer:
    """Reorder, gap-fill, pace, and measure jitter on inbound RTP audio."""

    def __init__(
        self,
        *,
        target_ms: int = 60,
        ptime_ms: int = 20,
        samples_per_frame: int = 160,
        sample_rate: int = 8000,
        reset_window_frames: int = 200,
        recommend_min_ms: int = 20,
        recommend_max_ms: int = 200,
    ):
        if target_ms < ptime_ms:
            raise ValueError("target_ms must be >= ptime_ms")
        self.target_ms = target_ms
        self.ptime_ms = ptime_ms
        self.samples_per_frame = samples_per_frame
        self.sample_rate = sample_rate
        self.reset_window_frames = reset_window_frames
        self.recommend_min_ms = recommend_min_ms
        self.recommend_max_ms = recommend_max_ms

        self._target_frames = max(1, target_ms // ptime_ms)
        self._frames: dict[int, bytes] = {}
        self._next_seq: int | None = None
        self._primed = False

        self._received = 0
        self._lost = 0
        self._late = 0
        self._resets = 0

        # RFC 3550 §A.8 interarrival jitter estimator (in seconds).
        self._jitter_s = 0.0
        self._jitter_samples = 0
        self._last_transit_s: float | None = None

    # ------------------------------------------------------------------
    def push(
        self,
        seq: int,
        pcm: bytes,
        *,
        rtp_timestamp: int | None = None,
        arrival_time: float | None = None,
    ) -> None:
        """Insert a decoded PCM frame received with RTP sequence *seq*.

        Passing ``rtp_timestamp`` (samples) and ``arrival_time`` (monotonic
        seconds) enables RFC 3550 §A.8 interarrival jitter measurement,
        surfaced via :attr:`stats` and :meth:`recommended_target_ms`. If
        ``arrival_time`` is omitted we use :func:`time.monotonic` ourselves.
        """
        seq &= 0xFFFF
        self._received += 1

        if rtp_timestamp is not None:
            self._update_jitter(rtp_timestamp, arrival_time)

        if self._next_seq is None:
            self._next_seq = seq
            self._frames[seq] = pcm
            return

        offset = _signed_seq_diff(seq, self._next_seq)
        if offset < 0:
            self._late += 1
            return
        if offset >= self.reset_window_frames:
            log.info("jitter buffer resync (seq jumped by %d)", offset)
            self._frames.clear()
            self._next_seq = seq
            self._primed = False
            self._resets += 1

        self._frames[seq] = pcm

    def _update_jitter(self, rtp_timestamp: int, arrival_time: float | None) -> None:
        # transit = arrival_time - rtp_timestamp_in_seconds
        if arrival_time is None:
            arrival_time = time.monotonic()
        transit = arrival_time - (rtp_timestamp / self.sample_rate)
        if self._last_transit_s is None:
            self._last_transit_s = transit
            return
        d = abs(transit - self._last_transit_s)
        self._last_transit_s = transit
        self._jitter_s += (d - self._jitter_s) / 16.0
        self._jitter_samples += 1

    def pop_pcm(self) -> bytes | None:
        """Return one ptime of PCM, or ``None`` while priming/empty."""
        if self._next_seq is None:
            return None
        if not self._primed:
            if len(self._frames) < self._target_frames:
                return None
            self._primed = True

        seq = self._next_seq
        self._next_seq = (seq + 1) & 0xFFFF
        frame = self._frames.pop(seq, None)
        if frame is None:
            self._lost += 1
            return silence_pcm(self.samples_per_frame)
        return frame

    def reset(self) -> None:
        """Forget all state. Use on SSRC change or after a long pause."""
        self._frames.clear()
        self._next_seq = None
        self._primed = False
        self._last_transit_s = None
        self._jitter_s = 0.0
        self._jitter_samples = 0

    @property
    def primed(self) -> bool:
        return self._primed

    @property
    def jitter_ms(self) -> float:
        return self._jitter_s * 1000.0

    def recommended_target_ms(self) -> int:
        """Suggested target depth based on measured jitter.

        Returns the current ``target_ms`` while the estimator is still warming
        up (<10 samples). Otherwise uses ``4 × jitter`` (≈ ±2σ for Gaussian
        delay), padded by one ptime, clamped to ``[recommend_min_ms,
        recommend_max_ms]``. Caller decides whether to act on it.
        """
        if self._jitter_samples < 10:
            return self.target_ms
        raw = int(self.jitter_ms * 4) + self.ptime_ms
        return max(self.recommend_min_ms, min(self.recommend_max_ms, raw))

    @property
    def stats(self) -> dict[str, float]:
        return {
            "received": self._received,
            "lost": self._lost,
            "late": self._late,
            "resets": self._resets,
            "buffered_frames": len(self._frames),
            "jitter_ms": round(self.jitter_ms, 3),
            "jitter_samples": self._jitter_samples,
        }


__all__ = ["JitterBuffer"]
