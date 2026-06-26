"""SIP URI, NameAddr, and Via header parsing.

We keep parsing pragmatic — strict enough for real SIP traffic but lenient about
optional whitespace, parameter quoting, and IPv6 brackets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from urllib.parse import quote, unquote

from .exceptions import SIPParseError


# ---------------------------------------------------------------------------
# Compact-form header mapping. Lower-case keys map to their canonical name.
# RFC 3261 §7.3.3 + §20.
# ---------------------------------------------------------------------------
COMPACT_FORMS = {
    "i": "Call-ID",
    "m": "Contact",
    "e": "Content-Encoding",
    "l": "Content-Length",
    "c": "Content-Type",
    "f": "From",
    "s": "Subject",
    "k": "Supported",
    "t": "To",
    "v": "Via",
    "u": "Allow-Events",
    "r": "Refer-To",
    "b": "Referred-By",
    "o": "Event",
    "x": "Session-Expires",
    "y": "Identity",
    "n": "Identity-Info",
}

# Canonical capitalization for commonly seen headers.
CANONICAL = {
    "call-id": "Call-ID",
    "cseq": "CSeq",
    "via": "Via",
    "from": "From",
    "to": "To",
    "contact": "Contact",
    "content-type": "Content-Type",
    "content-length": "Content-Length",
    "max-forwards": "Max-Forwards",
    "user-agent": "User-Agent",
    "expires": "Expires",
    "allow": "Allow",
    "supported": "Supported",
    "www-authenticate": "WWW-Authenticate",
    "proxy-authenticate": "Proxy-Authenticate",
    "authorization": "Authorization",
    "proxy-authorization": "Proxy-Authorization",
    "route": "Route",
    "record-route": "Record-Route",
    "session-expires": "Session-Expires",
    "min-se": "Min-SE",
}


def canon_header(name: str) -> str:
    n = name.strip()
    low = n.lower()
    if low in COMPACT_FORMS:
        return COMPACT_FORMS[low]
    if low in CANONICAL:
        return CANONICAL[low]
    return "-".join(p.capitalize() for p in n.split("-"))


# ---------------------------------------------------------------------------
# Parameter parsing — used for URIs, Via, Contact, etc.
# ---------------------------------------------------------------------------
def _split_top_level(text: str, sep: str) -> list[str]:
    """Split *text* on *sep* but ignore separators inside <...> or "..."."""
    parts: list[str] = []
    depth_angle = 0
    in_quote = False
    buf: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and in_quote and i + 1 < len(text):
            buf.append(ch)
            buf.append(text[i + 1])
            i += 2
            continue
        if ch == '"':
            in_quote = not in_quote
        elif not in_quote:
            if ch == "<":
                depth_angle += 1
            elif ch == ">":
                depth_angle = max(0, depth_angle - 1)
            elif ch == sep and depth_angle == 0:
                parts.append("".join(buf))
                buf = []
                i += 1
                continue
        buf.append(ch)
        i += 1
    parts.append("".join(buf))
    return parts


def parse_params(text: str) -> dict[str, str]:
    """Parse ``;key=val;flag`` style parameter strings."""
    out: dict[str, str] = {}
    if not text:
        return out
    for chunk in _split_top_level(text, ";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk:
            k, v = chunk.split("=", 1)
            v = v.strip()
            if len(v) >= 2 and v[0] == '"' and v[-1] == '"':
                v = v[1:-1]
            out[k.strip().lower()] = v
        else:
            out[chunk.lower()] = ""
    return out


def serialize_params(params: dict[str, str]) -> str:
    out = []
    for k, v in params.items():
        if v == "" or v is None:
            out.append(f";{k}")
        else:
            # quote values containing token-unsafe chars
            if any(c in v for c in ' \t",;'):
                out.append(f';{k}="{v}"')
            else:
                out.append(f";{k}={v}")
    return "".join(out)


# ---------------------------------------------------------------------------
# SIP URI
# ---------------------------------------------------------------------------
@dataclass
class URI:
    scheme: str = "sip"          # sip / sips / tel
    user: str | None = None
    password: str | None = None
    host: str = ""
    port: int | None = None
    params: dict[str, str] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "URI":
        s = text.strip()
        if ":" not in s:
            raise SIPParseError(f"missing URI scheme in {text!r}")
        scheme, rest = s.split(":", 1)
        scheme = scheme.lower()
        if scheme not in {"sip", "sips", "tel"}:
            raise SIPParseError(f"unsupported URI scheme {scheme!r}")

        # headers (?k=v&k=v) at the very end
        headers: dict[str, str] = {}
        if "?" in rest:
            rest, qs = rest.split("?", 1)
            for kv in qs.split("&"):
                if "=" in kv:
                    k, v = kv.split("=", 1)
                    headers[unquote(k)] = unquote(v)

        # parameters (;k=v;...)
        params: dict[str, str] = {}
        if ";" in rest:
            rest, pstr = rest.split(";", 1)
            params = parse_params(";" + pstr)

        # userinfo @ hostport
        user: str | None = None
        password: str | None = None
        if "@" in rest:
            userinfo, hostport = rest.rsplit("@", 1)
            if ":" in userinfo:
                user, password = userinfo.split(":", 1)
                user = unquote(user)
            else:
                user = unquote(userinfo)
        else:
            hostport = rest

        host = hostport
        port: int | None = None
        # IPv6: [::1]:5060
        if host.startswith("["):
            end = host.find("]")
            if end == -1:
                raise SIPParseError(f"unterminated IPv6 literal in {text!r}")
            ipv6 = host[1:end]
            rest2 = host[end + 1 :]
            host = ipv6
            if rest2.startswith(":"):
                port = int(rest2[1:])
        elif host.count(":") == 1:
            h, p = host.split(":", 1)
            host = h
            port = int(p)

        return cls(scheme=scheme, user=user, password=password,
                   host=host, port=port, params=params, headers=headers)

    def __str__(self) -> str:
        userinfo = ""
        if self.user is not None:
            u = quote(self.user, safe="!$&'()*+,;=:")
            if self.password is not None:
                userinfo = f"{u}:{self.password}@"
            else:
                userinfo = f"{u}@"
        host = f"[{self.host}]" if ":" in self.host else self.host
        port = f":{self.port}" if self.port else ""
        params = serialize_params(self.params)
        headers = ""
        if self.headers:
            headers = "?" + "&".join(
                f"{quote(k)}={quote(v)}" for k, v in self.headers.items()
            )
        return f"{self.scheme}:{userinfo}{host}{port}{params}{headers}"


# ---------------------------------------------------------------------------
# name-addr — "Display" <sip:...>;tag=...
# ---------------------------------------------------------------------------
@dataclass
class NameAddr:
    display: str | None = None
    uri: URI = field(default_factory=URI)
    params: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "NameAddr":
        s = text.strip()
        display: str | None = None

        if "<" in s and ">" in s:
            head, rest = s.split("<", 1)
            uri_str, rest = rest.split(">", 1)
            head = head.strip()
            if head:
                if head.startswith('"') and head.endswith('"'):
                    head = head[1:-1]
                display = head
            params = parse_params(rest.lstrip(";").strip()) if rest.strip() else {}
            return cls(display=display, uri=URI.parse(uri_str), params=params)

        # bare URI form (no angle brackets) — params after first ';' on the URI
        return cls(display=None, uri=URI.parse(s), params={})

    def __str__(self) -> str:
        parts = []
        if self.display:
            d = self.display
            if any(c in d for c in ' \t",;<>'):
                d = '"' + d.replace('"', '\\"') + '"'
            parts.append(d + " ")
        parts.append(f"<{self.uri}>")
        if self.params:
            parts.append(serialize_params(self.params))
        return "".join(parts)


# ---------------------------------------------------------------------------
# Via header — SIP/2.0/UDP host:port;branch=...;received=...;rport
# ---------------------------------------------------------------------------
@dataclass
class Via:
    protocol: str = "SIP/2.0"
    transport: str = "UDP"
    host: str = ""
    port: int | None = None
    params: dict[str, str] = field(default_factory=dict)

    @classmethod
    def parse(cls, text: str) -> "Via":
        s = text.strip()
        try:
            sent_proto, sent_by_and_params = s.split(" ", 1)
        except ValueError as e:
            raise SIPParseError(f"bad Via line: {text!r}") from e
        proto_parts = sent_proto.split("/")
        if len(proto_parts) != 3:
            raise SIPParseError(f"bad Via sent-protocol: {sent_proto!r}")
        protocol = f"{proto_parts[0]}/{proto_parts[1]}"
        transport = proto_parts[2]

        rest = sent_by_and_params.lstrip()
        params: dict[str, str] = {}
        if ";" in rest:
            host_part, pstr = rest.split(";", 1)
            params = parse_params(";" + pstr)
        else:
            host_part = rest

        host_part = host_part.strip()
        port: int | None = None
        host = host_part
        if host.startswith("["):
            end = host.find("]")
            if end == -1:
                raise SIPParseError(f"bad IPv6 in Via: {text!r}")
            ipv6 = host[1:end]
            tail = host[end + 1 :]
            host = ipv6
            if tail.startswith(":"):
                port = int(tail[1:])
        elif ":" in host:
            h, p = host.split(":", 1)
            host = h
            port = int(p)

        return cls(protocol=protocol, transport=transport, host=host,
                   port=port, params=params)

    def __str__(self) -> str:
        host = f"[{self.host}]" if ":" in self.host else self.host
        port = f":{self.port}" if self.port else ""
        return f"{self.protocol}/{self.transport} {host}{port}{serialize_params(self.params)}"


def parse_address_list(text: str) -> list[NameAddr]:
    """Parse comma-separated address lists (From/To/Contact/Route)."""
    out = []
    for piece in _split_top_level(text, ","):
        piece = piece.strip()
        if piece:
            out.append(NameAddr.parse(piece))
    return out


def parse_via_list(text: str) -> list[Via]:
    return [Via.parse(p.strip()) for p in _split_top_level(text, ",") if p.strip()]


__all__ = [
    "URI",
    "NameAddr",
    "Via",
    "canon_header",
    "parse_params",
    "serialize_params",
    "parse_address_list",
    "parse_via_list",
    "COMPACT_FORMS",
]
