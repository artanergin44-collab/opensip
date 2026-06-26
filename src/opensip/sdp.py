"""Minimal SDP (RFC 4566) parser/builder, focused on audio sessions.

The "offer/answer" subset we care about:

    v=0
    o=- <session-id> <session-version> IN IP4 <host>
    s=opensip
    c=IN IP4 <host>
    t=0 0
    m=audio <port> RTP/AVP <fmt> ...
    a=rtpmap:<pt> <encoding>/<clock>
    a=ptime:20
    a=sendrecv
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Iterable

from .exceptions import SDPError


# Default payload-type catalog. We pre-seed the well-known static ones so a
# remote that omits a=rtpmap can still be understood.
STATIC_PAYLOAD_TYPES: dict[int, tuple[str, int, int]] = {
    # pt: (codec, clock_rate, channels)
    0: ("PCMU", 8000, 1),
    3: ("GSM", 8000, 1),
    4: ("G723", 8000, 1),
    8: ("PCMA", 8000, 1),
    9: ("G722", 8000, 1),
    18: ("G729", 8000, 1),
    101: ("telephone-event", 8000, 1),  # by convention, despite being dynamic
}


@dataclass
class Codec:
    payload_type: int
    name: str
    clock_rate: int = 8000
    channels: int = 1
    fmtp: str | None = None

    def rtpmap_line(self) -> str:
        ch = f"/{self.channels}" if self.channels and self.channels != 1 else ""
        return f"a=rtpmap:{self.payload_type} {self.name}/{self.clock_rate}{ch}"


@dataclass
class MediaDescription:
    media: str = "audio"             # "audio" / "video"
    port: int = 0
    proto: str = "RTP/AVP"
    payload_types: list[int] = field(default_factory=list)
    codecs: list[Codec] = field(default_factory=list)
    direction: str = "sendrecv"      # sendrecv / sendonly / recvonly / inactive
    ptime: int | None = 20
    attributes: list[str] = field(default_factory=list)
    connection: tuple[str, str] | None = None  # (net_type, address) override

    def codec_for(self, pt: int) -> Codec | None:
        for c in self.codecs:
            if c.payload_type == pt:
                return c
        return None


@dataclass
class SDPSession:
    origin_user: str = "-"
    session_id: int = 0
    session_version: int = 0
    address: str = "0.0.0.0"
    session_name: str = "opensip"
    media: list[MediaDescription] = field(default_factory=list)

    # ------------------------------------------------------------------
    @classmethod
    def parse(cls, data: bytes | str) -> "SDPSession":
        if isinstance(data, bytes):
            data = data.decode("utf-8", errors="replace")
        text = data.replace("\r\n", "\n").strip()
        if not text:
            raise SDPError("empty SDP")

        sess = cls()
        cur_media: MediaDescription | None = None
        session_addr = "0.0.0.0"

        for raw_line in text.split("\n"):
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            t, _, val = line.partition("=")
            if t == "v":
                if val.strip() != "0":
                    raise SDPError(f"unsupported SDP version {val!r}")
            elif t == "o":
                parts = val.split()
                if len(parts) >= 6:
                    sess.origin_user = parts[0]
                    try:
                        sess.session_id = int(parts[1])
                        sess.session_version = int(parts[2])
                    except ValueError:
                        pass
                    session_addr = parts[5]
                    sess.address = session_addr
            elif t == "s":
                sess.session_name = val.strip()
            elif t == "c":
                parts = val.split()
                if len(parts) >= 3:
                    if cur_media is None:
                        sess.address = parts[2]
                        session_addr = parts[2]
                    else:
                        cur_media.connection = (parts[1], parts[2])
            elif t == "m":
                parts = val.split()
                if len(parts) < 4:
                    raise SDPError(f"bad m= line: {val!r}")
                media = parts[0]
                try:
                    port = int(parts[1].split("/")[0])
                except ValueError as e:
                    raise SDPError(f"bad media port: {parts[1]!r}") from e
                proto = parts[2]
                fmt_pts: list[int] = []
                for f in parts[3:]:
                    try:
                        fmt_pts.append(int(f))
                    except ValueError:
                        pass
                cur_media = MediaDescription(
                    media=media, port=port, proto=proto,
                    payload_types=fmt_pts,
                )
                # pre-seed codecs from static catalog
                for pt in fmt_pts:
                    if pt in STATIC_PAYLOAD_TYPES:
                        name, rate, ch = STATIC_PAYLOAD_TYPES[pt]
                        cur_media.codecs.append(
                            Codec(payload_type=pt, name=name,
                                  clock_rate=rate, channels=ch)
                        )
                sess.media.append(cur_media)
            elif t == "a" and cur_media is not None:
                attr = val.strip()
                if attr in ("sendrecv", "sendonly", "recvonly", "inactive"):
                    cur_media.direction = attr
                elif attr.startswith("rtpmap:"):
                    body = attr[len("rtpmap:"):].strip()
                    try:
                        pt_str, rest = body.split(None, 1)
                        pt = int(pt_str)
                    except ValueError:
                        continue
                    enc = rest.strip()
                    name = enc
                    rate = 8000
                    channels = 1
                    if "/" in enc:
                        bits = enc.split("/")
                        name = bits[0]
                        if len(bits) >= 2:
                            try:
                                rate = int(bits[1])
                            except ValueError:
                                pass
                        if len(bits) >= 3:
                            try:
                                channels = int(bits[2])
                            except ValueError:
                                pass
                    # replace any pre-seeded entry for that pt
                    cur_media.codecs = [c for c in cur_media.codecs if c.payload_type != pt]
                    cur_media.codecs.append(
                        Codec(payload_type=pt, name=name,
                              clock_rate=rate, channels=channels)
                    )
                elif attr.startswith("fmtp:"):
                    body = attr[len("fmtp:"):].strip()
                    try:
                        pt_str, params = body.split(None, 1)
                        pt = int(pt_str)
                    except ValueError:
                        continue
                    for c in cur_media.codecs:
                        if c.payload_type == pt:
                            c.fmtp = params
                            break
                elif attr.startswith("ptime:"):
                    try:
                        cur_media.ptime = int(attr.split(":", 1)[1])
                    except ValueError:
                        pass
                else:
                    cur_media.attributes.append(attr)
        return sess

    # ------------------------------------------------------------------
    def encode(self) -> bytes:
        if not self.session_id:
            self.session_id = int(time.time())
        if not self.session_version:
            self.session_version = self.session_id

        lines = [
            "v=0",
            f"o={self.origin_user} {self.session_id} {self.session_version} IN IP4 {self.address}",
            f"s={self.session_name}",
            f"c=IN IP4 {self.address}",
            "t=0 0",
        ]
        for m in self.media:
            pts = " ".join(str(p) for p in (m.payload_types or [c.payload_type for c in m.codecs]))
            lines.append(f"m={m.media} {m.port} {m.proto} {pts}")
            if m.connection:
                lines.append(f"c={m.connection[0]} IP4 {m.connection[1]}")
            for c in m.codecs:
                lines.append(c.rtpmap_line())
                if c.fmtp:
                    lines.append(f"a=fmtp:{c.payload_type} {c.fmtp}")
            if m.ptime:
                lines.append(f"a=ptime:{m.ptime}")
            lines.append(f"a={m.direction}")
            for a in m.attributes:
                lines.append(f"a={a}")
        return ("\r\n".join(lines) + "\r\n").encode("utf-8")


def make_audio_offer(local_ip: str, rtp_port: int,
                     codecs: Iterable[Codec] | None = None) -> SDPSession:
    """Helper: build an audio-only offer for the local endpoint."""
    if codecs is None:
        codecs = [
            Codec(payload_type=0, name="PCMU", clock_rate=8000),
            Codec(payload_type=8, name="PCMA", clock_rate=8000),
            Codec(payload_type=101, name="telephone-event",
                  clock_rate=8000, fmtp="0-16"),
        ]
    codecs_list = list(codecs)
    media = MediaDescription(
        media="audio",
        port=rtp_port,
        proto="RTP/AVP",
        payload_types=[c.payload_type for c in codecs_list],
        codecs=codecs_list,
    )
    return SDPSession(address=local_ip, media=[media])


def pick_common_codec(local: SDPSession, remote: SDPSession) -> Codec | None:
    """Pick the first locally-supported codec also offered by the remote."""
    if not local.media or not remote.media:
        return None
    remote_pts = {c.payload_type: c for c in remote.media[0].codecs}
    for c in local.media[0].codecs:
        if c.name.lower() == "telephone-event":
            continue
        if c.payload_type in remote_pts:
            r = remote_pts[c.payload_type]
            if r.name.upper() == c.name.upper():
                return c
    # also try name-based match (different PT numbers)
    local_names = {c.name.upper(): c for c in local.media[0].codecs}
    for r in remote.media[0].codecs:
        if r.name.upper() in local_names and r.name.lower() != "telephone-event":
            return local_names[r.name.upper()]
    return None


__all__ = [
    "Codec",
    "MediaDescription",
    "SDPSession",
    "make_audio_offer",
    "pick_common_codec",
    "STATIC_PAYLOAD_TYPES",
]
