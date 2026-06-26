"""Asyncio UDP transport for SIP signaling.

The transport speaks bytes, parses incoming datagrams into SIP messages, and
dispatches them to a single handler. Higher layers (transaction + UA) own
all the SIP semantics; the transport's job is just I/O.
"""

from __future__ import annotations

import asyncio
import ipaddress
import logging
import socket
from typing import Awaitable, Callable

from .exceptions import SIPParseError, TransportError
from .message import SIPRequest, SIPResponse, parse_message

log = logging.getLogger("opensip.transport")

MessageHandler = Callable[
    [SIPRequest | SIPResponse, tuple[str, int]],
    Awaitable[None] | None,
]


class _SIPDatagramProtocol(asyncio.DatagramProtocol):
    def __init__(self, owner: "UDPTransport") -> None:
        self.owner = owner
        self.transport: asyncio.DatagramTransport | None = None

    def connection_made(self, transport: asyncio.BaseTransport) -> None:  # type: ignore[override]
        self.transport = transport  # type: ignore[assignment]
        self.owner._on_connected(self)

    def datagram_received(self, data: bytes, addr: tuple[str, int]) -> None:
        self.owner._on_datagram(data, addr)

    def error_received(self, exc: Exception) -> None:
        log.warning("UDP error received: %s", exc)

    def connection_lost(self, exc: Exception | None) -> None:
        log.debug("UDP transport closed: %s", exc)
        self.owner._on_closed()


class UDPTransport:
    """Single-socket UDP transport for SIP."""

    def __init__(self, local_addr: tuple[str, int] = ("0.0.0.0", 5060)):
        self._local_addr = local_addr
        self._proto: _SIPDatagramProtocol | None = None
        self._handler: MessageHandler | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._ready = asyncio.Event()
        self._on_raw: Callable[[bytes, tuple[str, int]], None] | None = None
        self._resolved: dict[str, str] = {}

    # ------------------------------------------------------------------
    @property
    def local_addr(self) -> tuple[str, int]:
        if self._proto and self._proto.transport:
            sock = self._proto.transport.get_extra_info("sockname")
            if sock:
                return sock[0], sock[1]
        return self._local_addr

    def set_handler(self, handler: MessageHandler) -> None:
        self._handler = handler

    def set_raw_observer(self, cb: Callable[[bytes, tuple[str, int]], None]) -> None:
        """Optional raw-bytes observer (for logging / tracing)."""
        self._on_raw = cb

    # ------------------------------------------------------------------
    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        try:
            transport, proto = await self._loop.create_datagram_endpoint(
                lambda: _SIPDatagramProtocol(self),
                local_addr=self._local_addr,
                allow_broadcast=False,
            )
        except OSError as e:
            raise TransportError(f"could not bind UDP {self._local_addr}: {e}") from e
        # ``proto`` is the instance we created in the factory above.
        _ = transport
        await self._ready.wait()
        log.info("SIP transport listening on %s:%d", *self.local_addr)

    async def stop(self) -> None:
        if self._proto and self._proto.transport:
            self._proto.transport.close()
        self._proto = None

    async def send(self, data: bytes, addr: tuple[str, int]) -> None:
        if self._proto is None or self._proto.transport is None:
            raise TransportError("transport not started")
        ip_port = await self._resolve_addr(addr)
        if self._on_raw:
            try:
                self._on_raw(data, ip_port)
            except Exception:
                log.exception("raw observer raised")
        self._proto.transport.sendto(data, ip_port)

    async def _resolve_addr(self, addr: tuple[str, int]) -> tuple[str, int]:
        # Windows ProactorEventLoop UDP sockets can silently drop datagrams
        # whose destination is a hostname; resolve eagerly so we always hand
        # ``sendto`` a numeric IPv4 address.
        host, port = addr
        try:
            ipaddress.ip_address(host)
            return host, port
        except ValueError:
            pass
        cached = self._resolved.get(host)
        if cached:
            return cached, port
        loop = self._loop or asyncio.get_running_loop()
        try:
            infos = await loop.getaddrinfo(
                host, port, family=socket.AF_INET, type=socket.SOCK_DGRAM
            )
        except OSError as e:
            raise TransportError(f"could not resolve {host}: {e}") from e
        if not infos:
            raise TransportError(f"could not resolve {host}")
        ip = infos[0][4][0]
        self._resolved[host] = ip
        return ip, port

    # ------------------------------------------------------------------
    # Internal hooks called by the protocol object.
    # ------------------------------------------------------------------
    def _on_connected(self, proto: _SIPDatagramProtocol) -> None:
        self._proto = proto
        self._ready.set()

    def _on_closed(self) -> None:
        self._proto = None

    def _on_datagram(self, data: bytes, addr: tuple[str, int]) -> None:
        if self._on_raw:
            try:
                self._on_raw(data, addr)
            except Exception:
                log.exception("raw observer raised")
        try:
            msg = parse_message(data)
        except SIPParseError as e:
            log.warning("dropping unparsable datagram from %s: %s", addr, e)
            return
        if self._handler is None:
            log.debug("no handler installed; dropping %s", type(msg).__name__)
            return
        result = self._handler(msg, addr)
        if asyncio.iscoroutine(result):
            assert self._loop is not None
            self._loop.create_task(result)


__all__ = ["UDPTransport", "MessageHandler"]
