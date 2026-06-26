"""Optional microphone / speaker integration using ``sounddevice``.

This module is imported on demand. If ``sounddevice``/``numpy`` are not
installed, the import raises ``OpenSIPError`` with a hint to install
``opensip[audio]``.

Typical use:

    from opensip.audio import AudioBridge

    bridge = AudioBridge(sample_rate=8000)
    bridge.start()
    call.on_pcm(bridge.feed_speaker)
    while call.is_active:
        pcm = await bridge.read_microphone()
        call.write_pcm(pcm)
    bridge.stop()
"""

from __future__ import annotations

import asyncio
import queue

from .exceptions import OpenSIPError

try:
    import numpy as np  # type: ignore
    import sounddevice as sd  # type: ignore
    _HAVE_AUDIO = True
    _AUDIO_IMPORT_ERR: Exception | None = None
except Exception as e:  # pragma: no cover - depends on host
    np = None  # type: ignore
    sd = None  # type: ignore
    _HAVE_AUDIO = False
    _AUDIO_IMPORT_ERR = e


def _require_audio() -> None:
    if not _HAVE_AUDIO:
        raise OpenSIPError(
            "audio support not available — install with `pip install \"opensip[audio]\"`. "
            f"(import error: {_AUDIO_IMPORT_ERR!r})"
        )


class AudioBridge:
    """Bridge a SIP call's PCM stream to the local default mic/speaker.

    Uses 16-bit signed PCM mono at *sample_rate* (default 8000 to match
    G.711). Microphone audio is queued for the application to drain; speaker
    audio is pushed via :meth:`feed_speaker`.
    """

    def __init__(
        self,
        *,
        sample_rate: int = 8000,
        frame_ms: int = 20,
        input_device: int | str | None = None,
        output_device: int | str | None = None,
    ):
        _require_audio()
        self.sample_rate = sample_rate
        self.frame_ms = frame_ms
        self.frame_samples = sample_rate * frame_ms // 1000
        self._in_stream: "sd.InputStream | None" = None
        self._out_stream: "sd.OutputStream | None" = None
        self._mic_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=50)
        self._spk_queue: "queue.Queue[bytes]" = queue.Queue(maxsize=200)
        self._loop: asyncio.AbstractEventLoop | None = None
        self._input_device = input_device
        self._output_device = output_device

    # ---- lifecycle ----------------------------------------------------
    def start(self) -> None:
        _require_audio()
        self._loop = asyncio.get_event_loop()

        def in_cb(indata, frames, time_info, status):
            if status:
                pass  # we ignore xruns
            # indata is float32 in [-1, 1] — convert to int16 little-endian.
            pcm = (np.clip(indata[:, 0], -1.0, 1.0) * 32767).astype(np.int16).tobytes()
            try:
                self._loop.call_soon_threadsafe(self._mic_queue.put_nowait, pcm)
            except (RuntimeError, asyncio.QueueFull):
                pass

        def out_cb(outdata, frames, time_info, status):
            if status:
                pass
            needed = frames
            buf = bytearray()
            while len(buf) < needed * 2:
                try:
                    buf.extend(self._spk_queue.get_nowait())
                except queue.Empty:
                    break
            if len(buf) < needed * 2:
                buf.extend(b"\x00\x00" * (needed - len(buf) // 2))
            samples = np.frombuffer(bytes(buf[: needed * 2]), dtype=np.int16)
            outdata[:, 0] = samples.astype(np.float32) / 32767.0

        self._in_stream = sd.InputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            blocksize=self.frame_samples, callback=in_cb,
            device=self._input_device,
        )
        self._out_stream = sd.OutputStream(
            samplerate=self.sample_rate, channels=1, dtype="float32",
            blocksize=self.frame_samples, callback=out_cb,
            device=self._output_device,
        )
        self._in_stream.start()
        self._out_stream.start()

    def stop(self) -> None:
        for s in (self._in_stream, self._out_stream):
            try:
                if s is not None:
                    s.stop()
                    s.close()
            except Exception:
                pass
        self._in_stream = None
        self._out_stream = None

    # ---- bridging -----------------------------------------------------
    async def read_microphone(self) -> bytes:
        """Await one frame (≈ frame_ms milliseconds) of PCM from the mic."""
        return await self._mic_queue.get()

    def feed_speaker(self, pcm: bytes) -> None:
        """Push PCM bytes to the speaker. Safe to call from any thread."""
        try:
            self._spk_queue.put_nowait(pcm)
        except queue.Full:
            # drop oldest to keep latency bounded
            try:
                self._spk_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._spk_queue.put_nowait(pcm)
            except queue.Full:
                pass

    # ---- context manager sugar ---------------------------------------
    def __enter__(self) -> "AudioBridge":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


__all__ = ["AudioBridge"]
