"""SIP message parser and serializer.

We model the wire-level message:

    SIPRequest(method, request_uri, headers, body)
    SIPResponse(status_code, reason, headers, body)

Headers is a case-insensitive multi-dict — most names appear once, but
``Via``, ``Route``, ``Record-Route``, and ``Contact`` can be repeated.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Iterator

from .exceptions import SIPParseError
from .headers import canon_header

SIP_VERSION = "SIP/2.0"
CRLF = "\r\n"

# Headers that may legitimately repeat — we preserve all values.
MULTI_VALUED = {"Via", "Route", "Record-Route", "Contact", "Allow", "Supported",
                "Accept", "Allow-Events", "Proxy-Authenticate", "WWW-Authenticate",
                "Authorization", "Proxy-Authorization"}


class Headers:
    """Case-insensitive ordered header collection with multi-value support."""

    __slots__ = ("_items",)

    def __init__(self, items: Iterable[tuple[str, str]] | None = None):
        self._items: list[tuple[str, str]] = []
        if items:
            for k, v in items:
                self.add(k, v)

    # ------------------------------------------------------------------
    def add(self, name: str, value: str) -> None:
        self._items.append((canon_header(name), str(value)))

    def set(self, name: str, value: str) -> None:
        cname = canon_header(name)
        self._items = [(k, v) for (k, v) in self._items if k != cname]
        self._items.append((cname, str(value)))

    def get(self, name: str, default: str | None = None) -> str | None:
        cname = canon_header(name)
        for k, v in self._items:
            if k == cname:
                return v
        return default

    def get_all(self, name: str) -> list[str]:
        cname = canon_header(name)
        return [v for k, v in self._items if k == cname]

    def remove(self, name: str) -> None:
        cname = canon_header(name)
        self._items = [(k, v) for (k, v) in self._items if k != cname]

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def __getitem__(self, name: str) -> str:
        v = self.get(name)
        if v is None:
            raise KeyError(name)
        return v

    def __setitem__(self, name: str, value: str) -> None:
        self.set(name, value)

    def __iter__(self) -> Iterator[tuple[str, str]]:
        return iter(self._items)

    def items(self) -> list[tuple[str, str]]:
        return list(self._items)

    def __repr__(self) -> str:
        return f"Headers({self._items!r})"


# ---------------------------------------------------------------------------
@dataclass
class SIPMessage:
    headers: Headers = field(default_factory=Headers)
    body: bytes = b""

    # --- common header accessors ----------------------------------------
    @property
    def call_id(self) -> str | None:
        return self.headers.get("Call-ID")

    @property
    def cseq(self) -> tuple[int, str] | None:
        raw = self.headers.get("CSeq")
        if not raw:
            return None
        try:
            num, method = raw.split(None, 1)
            return int(num), method.strip().upper()
        except ValueError as e:
            raise SIPParseError(f"bad CSeq: {raw!r}") from e

    def set_body(self, payload: bytes | str, content_type: str | None = None) -> None:
        if isinstance(payload, str):
            payload = payload.encode("utf-8")
        self.body = payload
        if content_type:
            self.headers.set("Content-Type", content_type)
        self.headers.set("Content-Length", str(len(payload)))


@dataclass
class SIPRequest(SIPMessage):
    method: str = ""
    request_uri: str = ""

    def __post_init__(self) -> None:
        self.method = self.method.upper()

    def encode(self) -> bytes:
        # Make sure Content-Length matches the body — many proxies are strict.
        self.headers.set("Content-Length", str(len(self.body)))
        start = f"{self.method} {self.request_uri} {SIP_VERSION}{CRLF}"
        hdrs = "".join(f"{k}: {v}{CRLF}" for k, v in self.headers.items())
        return (start + hdrs + CRLF).encode("utf-8") + self.body


@dataclass
class SIPResponse(SIPMessage):
    status_code: int = 0
    reason: str = ""

    def encode(self) -> bytes:
        self.headers.set("Content-Length", str(len(self.body)))
        start = f"{SIP_VERSION} {self.status_code} {self.reason}{CRLF}"
        hdrs = "".join(f"{k}: {v}{CRLF}" for k, v in self.headers.items())
        return (start + hdrs + CRLF).encode("utf-8") + self.body


# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------
def parse_message(data: bytes) -> SIPRequest | SIPResponse:
    """Parse a single SIP message from raw bytes."""
    if not data:
        raise SIPParseError("empty message")

    # Split header section from body. Tolerate bare LF.
    sep = b"\r\n\r\n"
    idx = data.find(sep)
    if idx == -1:
        sep = b"\n\n"
        idx = data.find(sep)
        if idx == -1:
            raise SIPParseError("missing CRLFCRLF header/body separator")
    head = data[:idx].decode("utf-8", errors="replace")
    body = data[idx + len(sep):]

    lines = head.replace("\r\n", "\n").split("\n")
    if not lines:
        raise SIPParseError("no start-line")

    # Fold continuation lines (RFC 3261 §7.3.1: a header line beginning with
    # whitespace is a continuation of the previous one).
    folded: list[str] = []
    for line in lines:
        if line and line[0] in " \t" and folded:
            folded[-1] += " " + line.strip()
        else:
            folded.append(line)

    start_line = folded[0]
    header_lines = folded[1:]

    headers = Headers()
    for line in header_lines:
        if not line.strip():
            continue
        if ":" not in line:
            raise SIPParseError(f"bad header line: {line!r}")
        name, value = line.split(":", 1)
        headers.add(name.strip(), value.strip())

    # Honor Content-Length if present.
    cl = headers.get("Content-Length")
    if cl is not None:
        try:
            cl_int = int(cl.strip())
            body = body[:cl_int]
        except ValueError:
            pass

    if start_line.startswith(SIP_VERSION + " "):
        rest = start_line[len(SIP_VERSION) + 1 :]
        try:
            code_str, reason = rest.split(" ", 1)
        except ValueError:
            code_str, reason = rest, ""
        try:
            code = int(code_str)
        except ValueError as e:
            raise SIPParseError(f"bad status code: {code_str!r}") from e
        return SIPResponse(status_code=code, reason=reason.strip(),
                           headers=headers, body=body)

    # Request: METHOD SP URI SP SIP/2.0
    parts = start_line.split(" ")
    if len(parts) < 3:
        raise SIPParseError(f"bad request-line: {start_line!r}")
    method = parts[0]
    version = parts[-1]
    request_uri = " ".join(parts[1:-1])
    if version != SIP_VERSION:
        raise SIPParseError(f"unsupported SIP version: {version!r}")
    return SIPRequest(method=method, request_uri=request_uri,
                      headers=headers, body=body)


__all__ = [
    "Headers",
    "SIPMessage",
    "SIPRequest",
    "SIPResponse",
    "parse_message",
    "SIP_VERSION",
    "MULTI_VALUED",
]
