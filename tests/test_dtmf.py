"""Tests for inbound DTMF (RFC 4733 telephone-event) handling."""

import asyncio
import struct

import pytest

from opensip.rtp import (
    DTMF_DEFAULT_PT,
    RTPPacket,
    RTPSession,
    parse_telephone_event,
)


def _build_event(event: int, end: bool, *, duration: int = 160, volume: int = 10) -> bytes:
    e_r_vol = (0x80 if end else 0x00) | (volume & 0x3F)
    return struct.pack("!BBH", event & 0xFF, e_r_vol, duration & 0xFFFF)


def _packet(*, event: int, end: bool, ts: int, seq: int) -> RTPPacket:
    return RTPPacket(
        payload_type=DTMF_DEFAULT_PT,
        sequence=seq,
        timestamp=ts,
        ssrc=1,
        payload=_build_event(event, end),
    )


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def test_parse_short_payload_returns_none():
    assert parse_telephone_event(b"\x00\x00") is None


def test_parse_extracts_fields():
    payload = _build_event(event=5, end=True, duration=320, volume=10)
    out = parse_telephone_event(payload)
    assert out == (5, True, 10, 320)


def test_parse_ignores_end_bit_when_off():
    payload = _build_event(event=12, end=False, duration=160)
    assert parse_telephone_event(payload) == (12, False, 10, 160)


# ---------------------------------------------------------------------------
# RTPSession DTMF demux
# ---------------------------------------------------------------------------
async def _make_session() -> RTPSession:
    # asyncio.Queue requires a running loop in this codepath; wrap as async.
    return RTPSession(local_addr=("127.0.0.1", 0), payload_type=0)


async def test_end_packet_emits_callback_once():
    sess = await _make_session()
    received: list[str] = []
    sess.set_on_dtmf(received.append)

    # 3 redundant end-packets for the same event (RFC 4733 §2.5.1.4).
    for seq in (1, 2, 3):
        sess._handle_dtmf_packet(_packet(event=1, end=True, ts=1000, seq=seq))
    assert received == ["1"]


async def test_sustained_packets_are_silent():
    sess = await _make_session()
    received: list[str] = []
    sess.set_on_dtmf(received.append)

    # Several sustained packets before the end packet → no early emission.
    for seq in (1, 2, 3):
        sess._handle_dtmf_packet(_packet(event=2, end=False, ts=2000, seq=seq))
    assert received == []
    sess._handle_dtmf_packet(_packet(event=2, end=True, ts=2000, seq=4))
    assert received == ["2"]


async def test_consecutive_digits_each_emit():
    sess = await _make_session()
    received: list[str] = []
    sess.set_on_dtmf(received.append)

    sess._handle_dtmf_packet(_packet(event=7, end=True, ts=3000, seq=1))
    sess._handle_dtmf_packet(_packet(event=7, end=True, ts=4000, seq=2))  # second '7'
    sess._handle_dtmf_packet(_packet(event=8, end=True, ts=5000, seq=3))
    assert received == ["7", "7", "8"]


async def test_unknown_event_dropped():
    sess = await _make_session()
    received: list[str] = []
    sess.set_on_dtmf(received.append)
    sess._handle_dtmf_packet(_packet(event=200, end=True, ts=6000, seq=1))
    assert received == []


async def test_no_callback_no_crash():
    sess = await _make_session()
    # No callback registered — packets still get deduped without error.
    sess._handle_dtmf_packet(_packet(event=3, end=True, ts=7000, seq=1))


async def test_async_callback_is_scheduled():
    sess = await _make_session()
    seen = asyncio.Event()
    captured: list[str] = []

    async def cb(d: str) -> None:
        captured.append(d)
        seen.set()

    sess.set_on_dtmf(cb)
    sess._handle_dtmf_packet(_packet(event=4, end=True, ts=8000, seq=1))
    await asyncio.wait_for(seen.wait(), timeout=1.0)
    assert captured == ["4"]


async def test_dtmf_demux_via_on_datagram():
    """End-to-end through _on_datagram: an audio packet, then a DTMF packet."""
    sess = await _make_session()
    sess.set_remote(("127.0.0.1", 1234))  # so it doesn't auto-learn-and-skip
    digits: list[str] = []
    sess.set_on_dtmf(digits.append)

    audio_pkt = RTPPacket(
        payload_type=0, sequence=10, timestamp=160,
        ssrc=1, payload=b"\xff" * 160,
    )
    sess._on_datagram(audio_pkt.pack(), ("127.0.0.1", 1234))
    assert digits == []  # audio shouldn't trigger DTMF callback

    dtmf_pkt = _packet(event=9, end=True, ts=999, seq=11)
    sess._on_datagram(dtmf_pkt.pack(), ("127.0.0.1", 1234))
    assert digits == ["9"]
