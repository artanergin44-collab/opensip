"""Unit tests for the inbound RTP jitter buffer."""

from opensip.codecs import silence_pcm
from opensip.jitter import JitterBuffer


def _frame(seq: int) -> bytes:
    return f"f{seq}".encode()


def _make_buf(target_ms: int = 60, ptime_ms: int = 20) -> JitterBuffer:
    return JitterBuffer(target_ms=target_ms, ptime_ms=ptime_ms, samples_per_frame=160)


def _drain(buf: JitterBuffer, n: int) -> list[bytes | None]:
    return [buf.pop_pcm() for _ in range(n)]


def test_in_order_playout():
    buf = _make_buf()
    for s in (100, 101, 102):
        buf.push(s, _frame(s))
    assert _drain(buf, 3) == [_frame(100), _frame(101), _frame(102)]
    buf.push(103, _frame(103))
    assert buf.pop_pcm() == _frame(103)


def test_reorder_within_window():
    buf = _make_buf()
    for s in (100, 102, 101, 103):
        buf.push(s, _frame(s))
    assert _drain(buf, 4) == [_frame(100), _frame(101), _frame(102), _frame(103)]


def test_gap_filled_with_silence():
    buf = _make_buf()
    for s in (100, 101, 102, 104):  # 103 missing
        buf.push(s, _frame(s))
    assert _drain(buf, 5) == [
        _frame(100), _frame(101), _frame(102),
        silence_pcm(160),
        _frame(104),
    ]
    assert buf.stats["lost"] == 1


def test_late_packet_dropped():
    buf = _make_buf()
    for s in (100, 101, 102):
        buf.push(s, _frame(s))
    assert _drain(buf, 2) == [_frame(100), _frame(101)]
    buf.push(100, _frame(100))  # arrives after its slot
    assert buf.pop_pcm() == _frame(102)
    assert buf.stats["late"] == 1


def test_sequence_wraparound():
    buf = _make_buf()
    for s in (65534, 0, 65535, 1):  # straddles the 16-bit wrap, out-of-order
        buf.push(s, _frame(s))
    assert _drain(buf, 4) == [
        _frame(65534), _frame(65535), _frame(0), _frame(1),
    ]


def test_prime_blocks_pop_until_target():
    buf = _make_buf(target_ms=60, ptime_ms=20)  # 3 frames to prime
    buf.push(100, _frame(100))
    buf.push(101, _frame(101))
    assert buf.pop_pcm() is None
    buf.push(102, _frame(102))
    assert buf.pop_pcm() == _frame(100)


def test_resync_on_far_jump():
    buf = _make_buf()
    for s in (100, 101, 102):
        buf.push(s, _frame(s))
    _drain(buf, 3)
    far = (103 + 500) & 0xFFFF
    buf.push(far, _frame(far))
    buf.push((far + 1) & 0xFFFF, _frame((far + 1) & 0xFFFF))
    buf.push((far + 2) & 0xFFFF, _frame((far + 2) & 0xFFFF))
    assert _drain(buf, 3) == [
        _frame(far),
        _frame((far + 1) & 0xFFFF),
        _frame((far + 2) & 0xFFFF),
    ]
    assert buf.stats["resets"] == 1


# ---------------------------------------------------------------------------
# RFC 3550 §A.8 interarrival jitter measurement
# ---------------------------------------------------------------------------
def test_no_jitter_metric_without_timestamps():
    buf = _make_buf()
    for s in range(20):
        buf.push(100 + s, _frame(s))
    assert buf.stats["jitter_samples"] == 0
    assert buf.stats["jitter_ms"] == 0.0


def test_jitter_stays_zero_for_regular_arrivals():
    buf = _make_buf()
    for i in range(50):
        buf.push(100 + i, _frame(i),
                 rtp_timestamp=160 * i, arrival_time=0.020 * i)
    assert buf.stats["jitter_ms"] == 0.0


def test_jitter_grows_with_arrival_variation():
    buf = _make_buf()
    # Alternate on-time and 10ms-late arrivals. RFC 3550's EWMA should
    # converge near the magnitude of the per-packet delta (~10ms).
    for i in range(200):
        skew = 0.010 if i % 2 else 0.0
        buf.push(100 + i, _frame(i),
                 rtp_timestamp=160 * i, arrival_time=0.020 * i + skew)
    j = buf.stats["jitter_ms"]
    assert 5.0 < j < 15.0, f"unexpected converged jitter: {j}"


def test_recommended_target_warmup_returns_default():
    buf = _make_buf(target_ms=60)
    # Fewer than 10 jitter samples → return existing target_ms unchanged.
    for i in range(5):
        buf.push(100 + i, _frame(i),
                 rtp_timestamp=160 * i, arrival_time=0.020 * i)
    assert buf.recommended_target_ms() == 60


def test_recommended_target_clamped_to_max():
    buf = _make_buf(target_ms=60)
    # Pathological 200ms swings → recommendation should saturate at the max.
    for i in range(200):
        skew = 0.200 if i % 2 else 0.0
        buf.push(100 + i, _frame(i),
                 rtp_timestamp=160 * i, arrival_time=0.020 * i + skew)
    assert buf.recommended_target_ms() == 200  # default recommend_max_ms


def test_reset_clears_jitter_state():
    buf = _make_buf()
    for i in range(20):
        buf.push(100 + i, _frame(i),
                 rtp_timestamp=160 * i, arrival_time=0.020 * i + 0.005 * (i % 2))
    assert buf.stats["jitter_samples"] > 0
    buf.reset()
    assert buf.stats["jitter_samples"] == 0
    assert buf.stats["jitter_ms"] == 0.0
