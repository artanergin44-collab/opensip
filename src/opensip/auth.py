"""HTTP Digest authentication for SIP (RFC 2617 + RFC 3261 §22.4).

We support:
  * MD5 (default) and MD5-sess
  * qop="auth" with client nonce + nc counter
  * SHA-256 / SHA-256-sess (RFC 7616) when servers offer them
"""

from __future__ import annotations

import hashlib
import secrets
from dataclasses import dataclass, field

from .exceptions import AuthenticationError


# ---------------------------------------------------------------------------
# Tokenizer for the WWW-Authenticate / Proxy-Authenticate header value.
# These headers look like:  Digest realm="x", nonce="y", qop="auth,auth-int"
# ---------------------------------------------------------------------------
def parse_challenge(header_value: str) -> dict[str, str]:
    """Parse a Digest challenge header value into a dict of parameters."""
    s = header_value.strip()
    # Strip the scheme prefix ("Digest").
    if s.lower().startswith("digest"):
        s = s[6:].lstrip()
    out: dict[str, str] = {}
    i = 0
    n = len(s)
    while i < n:
        # skip whitespace and commas
        while i < n and s[i] in " ,\t":
            i += 1
        if i >= n:
            break
        # key
        j = i
        while j < n and s[j] not in "=, ":
            j += 1
        key = s[i:j].lower()
        i = j
        # skip = and whitespace
        while i < n and s[i] in " =\t":
            i += 1
        # value: quoted or token
        if i < n and s[i] == '"':
            i += 1
            buf = []
            while i < n and s[i] != '"':
                if s[i] == "\\" and i + 1 < n:
                    buf.append(s[i + 1])
                    i += 2
                else:
                    buf.append(s[i])
                    i += 1
            value = "".join(buf)
            i += 1  # closing quote
        else:
            j = i
            while j < n and s[j] != ",":
                j += 1
            value = s[i:j].strip()
            i = j
        if key:
            out[key] = value
    return out


# ---------------------------------------------------------------------------
@dataclass
class Challenge:
    realm: str = ""
    nonce: str = ""
    opaque: str | None = None
    algorithm: str = "MD5"
    qop: list[str] = field(default_factory=list)
    domain: str | None = None
    stale: bool = False

    @classmethod
    def from_header(cls, header_value: str) -> "Challenge":
        d = parse_challenge(header_value)
        qop_raw = d.get("qop", "")
        qop = [x.strip().lower() for x in qop_raw.split(",") if x.strip()]
        return cls(
            realm=d.get("realm", ""),
            nonce=d.get("nonce", ""),
            opaque=d.get("opaque"),
            algorithm=d.get("algorithm", "MD5"),
            qop=qop,
            domain=d.get("domain"),
            stale=d.get("stale", "").lower() == "true",
        )


# ---------------------------------------------------------------------------
def _hash(algorithm: str, data: bytes) -> str:
    base = algorithm.lower().rstrip("-sess")
    if base == "md5":
        return hashlib.md5(data).hexdigest()
    if base == "sha-256":
        return hashlib.sha256(data).hexdigest()
    if base == "sha-512-256":
        return hashlib.new("sha512_256", data).hexdigest() \
            if "sha512_256" in hashlib.algorithms_available \
            else hashlib.sha256(data).hexdigest()[:64]
    raise AuthenticationError(f"unsupported digest algorithm: {algorithm!r}")


def _h(algorithm: str, s: str) -> str:
    return _hash(algorithm, s.encode("utf-8"))


def build_authorization(
    *,
    challenge: Challenge,
    method: str,
    uri: str,
    username: str,
    password: str,
    body: bytes = b"",
    nc: int = 1,
    cnonce: str | None = None,
    proxy: bool = False,
) -> str:
    """Build a Digest Authorization (or Proxy-Authorization) header value."""
    algo = challenge.algorithm or "MD5"
    algo_sess = algo.lower().endswith("-sess")

    # HA1
    a1 = f"{username}:{challenge.realm}:{password}"
    ha1 = _h(algo, a1)
    if algo_sess:
        ha1 = _h(algo, f"{ha1}:{challenge.nonce}:{cnonce}")

    # qop selection
    qop = None
    if "auth" in challenge.qop:
        qop = "auth"
    elif "auth-int" in challenge.qop:
        qop = "auth-int"

    # HA2
    if qop == "auth-int":
        ha2 = _h(algo, f"{method}:{uri}:{_h(algo, body.decode('latin-1') if isinstance(body, bytes) else body)}")
    else:
        ha2 = _h(algo, f"{method}:{uri}")

    nc_str = f"{nc:08x}"
    if cnonce is None:
        cnonce = secrets.token_hex(8)

    if qop:
        response = _h(algo, f"{ha1}:{challenge.nonce}:{nc_str}:{cnonce}:{qop}:{ha2}")
    else:
        response = _h(algo, f"{ha1}:{challenge.nonce}:{ha2}")

    parts = [
        f'username="{username}"',
        f'realm="{challenge.realm}"',
        f'nonce="{challenge.nonce}"',
        f'uri="{uri}"',
        f'response="{response}"',
        f"algorithm={algo}",
    ]
    if challenge.opaque is not None:
        parts.append(f'opaque="{challenge.opaque}"')
    if qop:
        parts.append(f"qop={qop}")
        parts.append(f"nc={nc_str}")
        parts.append(f'cnonce="{cnonce}"')

    return "Digest " + ", ".join(parts)


__all__ = ["Challenge", "parse_challenge", "build_authorization"]
