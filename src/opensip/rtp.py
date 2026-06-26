"""RTP packetization (RFC 3550) + a thin asyncio session.

We only need the bare minimum for VoIP-style audio:
  * Pack/unpack the 12-byte fixed header
  * Manage sequence / timestamp / SSRC
  * Schedule outgoing audio in 20ms ptime frames
  * Deliver incoming PCM frames to a callback (or queue)

Higher layers feed PCM in/out; codec choice lives in :mod:`opensip.codecs`.
"""

from __future__ import annotations

import asyncio
import logging
import secrets
import socket
import struct
from dataclasses import dataclass
from typing import Awaitable, Callable

from .codecs import get_codec, silence_pcm
from .exceptions import TransportError
from .jitter import JitterBuffer

log = logging.getLogger("opensip.rtp")

RTP_VERSION = 2
RTP_HEADER_LEN = 12

DTMF_DEFAULT_PT = 101  # RFC 4733 telephone-event, conventional dynamic PT

DTMF_EVENTS: dict[str, int] = {
    "0": 0, "1": 1, "2": 2, "3": 3, "4": 4,
    "5": 5, "6": 6, "7": 7, "8": 8, "9": 9,
    "*": 10, "#": 11,
    "A": 12, "B": 13, "C": 14, "D": 15,
}
DTMF_DIGITS: dict[int, str] = {v: k for k, v in DTMF_EVENTS.items()}


def parse_telephone_event(payload: bytes) -> tuple[int, bool, int, int] | None:
    """Parse an RFC 4733 telephone-event payload.

    Returns ``(event, end, volume, duration_samples)`` or ``None`` if the
    payload is too short to be a valid event.
    """
    if len(payload) < 4:
        return None
    event, e_r_vol, duration = struct.unpack("!BBH", payload[:4])
    return event, bool(e_r_vol & 0x80), e_r_vol & 0x3F, duration


# ---------------------------------------------------------------------------
@dataclass
class RTPPacket:
    payload_type: int
    sequence: int
    timestamp: int
    ssrc: int
    payload: bytes
    marker: bool = False
    version: int = RTP_VERSION
    padding: bool = False
    extension: bool = False
    csrc: tuple[int, ...] = ()

    # ---------- encoding / decoding ----------
    def pack(self) -> bytes:
        b0 = (self.version & 0x03) << 6
        if self.padding:
            b0 |= 0x20
        if self.extension:
            b0 |= 0x10
        b0 |= len(self.csrc) & 0x0F
        b1 = (0x80 if self.marker else 0x00) | (self.payload_type & 0x7F)
        header = struct.pack("!BBHII", b0, b1, self.sequence & 0xFFFF,
                             self.timestamp & 0xFFFFFFFF,
                             self.ssrc & 0xFFFFFFFF)
        return header + b"".join(struct.pack("!I", c) for c in self.csrc) + self.payload

    @classmethod
    def unpack(cls, data: bytes) -> "RTPPacket":
        if len(data) < RTP_HEADER_LEN:
            raise ValueError("RTP packet too short")
        b0, b1, seq, ts, ssrc = struct.unpack("!BBHII", data[:RTP_HEADER_LEN])
        version = (b0 >> 6) & 0x03
        padding = bool(b0 & 0x20)
        extension = bool(b0 & 0x10)
        cc = b0 & 0x0F
        marker = bool(b1 & 0x80)
        pt = b1 & 0x7F
        offset = RTP_HEADER_LEN
        csrc: tuple[int, ...] = ()
        if cc:
            csrc = struct.unpack("!" + "I" * cc, data[offset:offset + 4 * cc])
            offset += 4 * cc
        if extension:
            if len(data) < offset + 4:
                raise ValueError("RTP extension header truncated")
            _, ext_len = struct.unpack("!HH", data[offset:offset + 4])
            offset += 4 + ext_len * 4
        payload = data[offset:]
        if padding and payload:
            pad_len = payload[-1]
            if pad_len <= len(payload):
                payload = payload[:-pad_len]
        return cls(payload_type=pt, sequence=seq, timestamp=ts, ssrc=ssrc,
                   payload=payload, marker=marker, version=version,
                   padding=padding, extension=extension, csrc=csrc)


# ---------------------------------------------------------------------------
class _RTPDatagramProtocol(asyncio.DatagramProtocol):
    def __init__(self, owner: "RTPSession") -> None:
        self.owner = owner

    def connection_made(self, transport: asyncio.BaseTransport) -> None:  # type: ignore[override]
        self.owner._transport = transport  # type: ignore[assignment]

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self.owner._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        log.warning("RTP error: %s", exc)


PCMCallback = Callable[[bytes], Awaitable[None] | None]
DTMFCallback = Callable[[str], Awaitable[None] | None]


class RTPSession:
    """A bidirectional RTP/G.711 audio session.

    ``codec_name`` is ``"PCMU"`` or ``"PCMA"`` (the only codecs the helper
    supports out of the box). The session sends 20ms frames (160 samples at
    8kHz) when fed PCM via :meth:`write_pcm` or :meth:`send_silence`.
    """

    def __init__(
        self,
        *,
        local_addr: tuple[str, int],
        payload_type: int = 0,
        codec_name: str = "PCMU",
        ptime_ms: int = 20,
        sample_rate: int = 8000,
        on_pcm: PCMCallback | None = None,
        on_dtmf: DTMFCallback | None = None,
        dtmf_payload_type: int = DTMF_DEFAULT_PT,
        jitter_ms: int = 60,
    ):
        self.local_addr = local_addr
        self.payload_type = payload_type
        self.codec_name = codec_name
        self.ptime_ms = ptime_ms
        self.sample_rate = sample_rate
        self._samples_per_frame = sample_rate * ptime_ms // 1000
        self._frame_bytes_pcm = self._samples_per_frame * 2  # 16-bit
        self._encode, self._decode = get_codec(codec_name)
        self._on_pcm = on_pcm
        self._on_dtmf = on_dtmf
        self.dtmf_payload_type = dtmf_payload_type
        # Recently-emitted DTMF (timestamp, event) keys for end-packet dedup.
        # RFC 4733 §2.5.1.4 prescribes three redundant end packets per event;
        # we want to surface each digit once.
        self._recent_dtmf_keys: list[tuple[int, int]] = []

        # Counters surfaced via .stats. Maintained inline by sender/receiver.
        self._packets_sent = 0
        self._packets_recv = 0
        self._bytes_sent = 0
        self._bytes_recv = 0
        self._dtmf_recv = 0

        self.ssrc = secrets.randbits(32)
        self._sequence = secrets.randbits(16)
        self._timestamp = secrets.randbits(32)
        self._first_packet = True
        self._dtmf_busy = False

        self.remote_addr: tuple[str, int] | None = None
        self._transport: asyncio.DatagramTransport | None = None
        self._send_buf = bytearray()
        self._send_task: asyncio.Task[None] | None = None
        self._send_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        self._stopped = asyncio.Event()

        self.jitter_ms = jitter_ms
        if jitter_ms > 0:
            self._jitter: JitterBuffer | None = JitterBuffer(
                target_ms=jitter_ms,
                ptime_ms=ptime_ms,
                samples_per_frame=self._samples_per_frame,
                sample_rate=sample_rate,
            )
        else:
            self._jitter = None
        self._player_task: asyncio.Task[None] | None = None

    # ------------------------------------------------------------------
    @property
    def port(self) -> int:
        return self.local_addr[1]

    async def start(self) -> None:
        loop = asyncio.get_running_loop()
        try:
            await loop.create_datagram_endpoint(
                lambda: _RTPDatagramProtocol(self),
                local_addr=self.local_addr,
            )
        except OSError as e:
            raise TransportError(f"cannot bind RTP {self.local_addr}: {e}") from e
        # refresh local_addr in case OS chose port 0
        if self._transport:
            sock = self._transport.get_extra_info("sockname")
            if sock:
                self.local_addr = (sock[0], sock[1])
        log.info("RTP %s session listening on %s:%d",
                 self.codec_name, *self.local_addr)
        self._send_task = asyncio.create_task(self._sender_loop())
        if self._jitter is not None:
            self._player_task = asyncio.create_task(self._player_loop())

    async def stop(self) -> None:
        self._stopped.set()
        if self._send_task:
            self._send_task.cancel()
            try:
                await self._send_task
            except (asyncio.CancelledError, Exception):
                pass
            self._send_task = None
        if self._player_task:
            self._player_task.cancel()
            try:
                await self._player_task
            except (asyncio.CancelledError, Exception):
                pass
            self._player_task = None
        if self._transport:
            self._transport.close()
            self._transport = None

    def set_remote(self, addr: tuple[str, int]) -> None:
        self.remote_addr = addr

    def set_on_pcm(self, cb: PCMCallback | None) -> None:
        self._on_pcm = cb

    def set_on_dtmf(self, cb: DTMFCallback | None) -> None:
        self._on_dtmf = cb

    # ------------------------------------------------------------------
    # Sending. write_pcm() buffers, the sender loop emits one frame per ptime.
    # ------------------------------------------------------------------
    def write_pcm(self, pcm: bytes) -> None:
        """Buffer 16-bit PCM for transmission. Chunked into ptime frames."""
        self._send_buf.extend(pcm)
        while len(self._send_buf) >= self._frame_bytes_pcm:
            frame = bytes(self._send_buf[: self._frame_bytes_pcm])
            del self._send_buf[: self._frame_bytes_pcm]
            try:
                self._send_queue.put_nowait(frame)
            except asyncio.QueueFull:
                # drop oldest to bound latency
                try:
                    self._send_queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
                self._send_queue.put_nowait(frame)

    def send_silence(self, frames: int = 1) -> None:
        for _ in range(frames):
            self.write_pcm(silence_pcm(self._samples_per_frame))

    async def _sender_loop(self) -> None:
        interval = self.ptime_ms / 1000.0
        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        while not self._stopped.is_set():
            try:
                pcm_frame = await asyncio.wait_for(
                    self._send_queue.get(), timeout=interval * 4
                )
            except asyncio.TimeoutError:
                pcm_frame = silence_pcm(self._samples_per_frame)
            except asyncio.CancelledError:
                return
            self._emit_packet(pcm_frame)
            next_tick += interval
            sleep = next_tick - loop.time()
            if sleep < -interval:
                # we are way behind — resync
                next_tick = loop.time()
            elif sleep > 0:
                try:
                    await asyncio.sleep(sleep)
                except asyncio.CancelledError:
                    return

    def _emit_packet(self, pcm_frame: bytes) -> None:
        if self._transport is None or self.remote_addr is None:
            return
        if self._dtmf_busy:
            # DTMF holds the stream; drop this audio frame so timestamps stay
            # aligned with what send_dtmf() will set when it completes.
            return
        payload = self._encode(pcm_frame)
        pkt = RTPPacket(
            payload_type=self.payload_type,
            sequence=self._sequence,
            timestamp=self._timestamp,
            ssrc=self.ssrc,
            payload=payload,
            marker=self._first_packet,
        )
        self._first_packet = False
        self._sequence = (self._sequence + 1) & 0xFFFF
        self._timestamp = (self._timestamp + self._samples_per_frame) & 0xFFFFFFFF
        wire = pkt.pack()
        try:
            self._transport.sendto(wire, self.remote_addr)
        except OSError as e:
            log.debug("RTP send failed: %s", e)
            return
        self._packets_sent += 1
        self._bytes_sent += len(wire)

    # ------------------------------------------------------------------
    # DTMF (RFC 4733 telephone-event)
    # ------------------------------------------------------------------
    async def send_dtmf(
        self, digit: str, duration_ms: int = 160, volume: int = 10
    ) -> None:
        """Send one DTMF tone in-band over RTP as telephone-event packets.

        ``digit`` is one of ``0-9``, ``*``, ``#``, ``A-D``. ``volume`` is in
        -dBm0 (RFC 4733 §2.5.1), range 0..63 — 10 is the recommended default.
        Blocks ~``duration_ms`` plus two extra end-packet ticks (~40 ms).
        """
        key = digit.upper()
        if key not in DTMF_EVENTS:
            raise ValueError(f"invalid DTMF digit: {digit!r}")
        if not 0 <= volume <= 63:
            raise ValueError(f"DTMF volume must be 0..63, got {volume}")
        if self._transport is None or self.remote_addr is None:
            raise TransportError("RTP session has no remote address")

        event = DTMF_EVENTS[key]
        # Round duration up to a whole number of ptime ticks (min 1).
        n_packets = max(1, duration_ms // self.ptime_ms)
        total_samples = n_packets * self._samples_per_frame
        event_ts = self._timestamp & 0xFFFFFFFF
        interval = self.ptime_ms / 1000.0

        self._dtmf_busy = True
        try:
            # Sustained packets: one per ptime, growing duration field.
            for i in range(n_packets):
                cur_samples = (i + 1) * self._samples_per_frame
                self._send_dtmf_packet(
                    event=event, volume=volume,
                    duration_samples=cur_samples, event_ts=event_ts,
                    marker=(i == 0), end=False,
                )
                if i < n_packets - 1:
                    await asyncio.sleep(interval)
            # Three redundant end packets (RFC 4733 §2.5.1.4).
            for _ in range(3):
                self._send_dtmf_packet(
                    event=event, volume=volume,
                    duration_samples=total_samples, event_ts=event_ts,
                    marker=False, end=True,
                )
                await asyncio.sleep(interval)
        finally:
            # Skip audio timestamps forward over the event window so the next
            # audio frame is contiguous from the receiver's perspective.
            self._timestamp = (event_ts + total_samples) & 0xFFFFFFFF
            self._first_packet = False
            self._dtmf_busy = False

    def _send_dtmf_packet(
        self, *, event: int, volume: int, duration_samples: int,
        event_ts: int, marker: bool, end: bool,
    ) -> None:
        # 4-byte payload: event | E|R|volume | duration (16-bit, samples)
        e_r_vol = (0x80 if end else 0x00) | (volume & 0x3F)
        payload = struct.pack(
            "!BBH", event & 0xFF, e_r_vol, duration_samples & 0xFFFF
        )
        pkt = RTPPacket(
            payload_type=self.dtmf_payload_type,
            sequence=self._sequence,
            timestamp=event_ts,
            ssrc=self.ssrc,
            payload=payload,
            marker=marker,
        )
        self._sequence = (self._sequence + 1) & 0xFFFF
        wire = pkt.pack()
        try:
            self._transport.sendto(wire, self.remote_addr)  # type: ignore[union-attr]
        except OSError as e:
            log.debug("DTMF send failed: %s", e)
            return
        self._packets_sent += 1
        self._bytes_sent += len(wire)

    # ------------------------------------------------------------------
    def _on_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        # Auto-learn remote address from first packet (NAT-friendly).
        if self.remote_addr is None:
            self.remote_addr = addr
        self._packets_recv += 1
        self._bytes_recv += len(data)
        try:
            pkt = RTPPacket.unpack(data)
        except ValueError as e:
            log.debug("dropping bad RTP packet from %s: %s", addr, e)
            return
        if pkt.payload_type == self.dtmf_payload_type:
            self._handle_dtmf_packet(pkt)
            return
        if pkt.payload_type != self.payload_type:
            return
        try:
            pcm = self._decode(pkt.payload)
        except Exception as e:  # noqa: BLE001
            log.debug("RTP decode failed: %s", e)
            return
        if self._jitter is not None:
            self._jitter.push(pkt.sequence, pcm, rtp_timestamp=pkt.timestamp)
            return
        if self._on_pcm:
            res = self._on_pcm(pcm)
            if asyncio.iscoroutine(res):
                asyncio.create_task(res)

    def _handle_dtmf_packet(self, pkt: RTPPacket) -> None:
        parsed = parse_telephone_event(pkt.payload)
        if parsed is None:
            return
        event, end, _volume, _duration = parsed
        if not end:
            return  # ignore sustained packets — only emit at end-of-event
        key = (pkt.timestamp, event)
        if key in self._recent_dtmf_keys:
            return  # duplicate end-packet (RFC 4733 §2.5.1.4 triple-send)
        self._recent_dtmf_keys.append(key)
        # Cap dedup history so it can't grow without bound on a long call.
        if len(self._recent_dtmf_keys) > 32:
            del self._recent_dtmf_keys[:16]
        digit = DTMF_DIGITS.get(event)
        if digit is None:
            log.debug("unknown DTMF event: %d", event)
            return
        self._dtmf_recv += 1
        if self._on_dtmf is None:
            return
        res = self._on_dtmf(digit)
        if asyncio.iscoroutine(res):
            asyncio.create_task(res)

    @property
    def stats(self) -> dict:
        """Return a snapshot of session counters (and jitter sub-stats)."""
        snap: dict = {
            "packets_sent": self._packets_sent,
            "packets_recv": self._packets_recv,
            "bytes_sent": self._bytes_sent,
            "bytes_recv": self._bytes_recv,
            "dtmf_recv": self._dtmf_recv,
        }
        if self._jitter is not None:
            snap["jitter"] = self._jitter.stats
        return snap

    async def _player_loop(self) -> None:
        """Tick once per ptime, draining the jitter buffer to ``on_pcm``."""
        assert self._jitter is not None
        interval = self.ptime_ms / 1000.0
        loop = asyncio.get_running_loop()
        next_tick = loop.time()
        while not self._stopped.is_set():
            next_tick += interval
            sleep = next_tick - loop.time()
            if sleep > 0:
                try:
                    await asyncio.sleep(sleep)
                except asyncio.CancelledError:
                    return
            elif sleep < -interval:
                next_tick = loop.time()  # we fell behind — resync
            pcm = self._jitter.pop_pcm()
            if pcm is None or self._on_pcm is None:
                continue
            res = self._on_pcm(pcm)
            if asyncio.iscoroutine(res):
                asyncio.create_task(res)


# ---------------------------------------------------------------------------
def pick_rtp_port_pair(
    host: str = "0.0.0.0", *, low: int = 16384, high: int = 32767
) -> int:
    """Find a free even UDP port in *low..high* (RTP convention)."""
    for _ in range(50):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.bind((host, 0))
            port = s.getsockname()[1]
        finally:
            s.close()
        if port % 2 == 0 and low <= port <= high:
            return port
    # fall back to whatever we got
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind((host, 0))
        return s.getsockname()[1]
    finally:
        s.close()


__all__ = [
    "RTPPacket", "RTPSession", "pick_rtp_port_pair",
    "parse_telephone_event", "DTMF_EVENTS", "DTMF_DIGITS",
]
