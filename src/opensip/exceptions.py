"""Custom exception hierarchy for opensip."""

from __future__ import annotations


class OpenSIPError(Exception):
    """Base class for all opensip errors."""


class SIPParseError(OpenSIPError):
    """Raised when a SIP message cannot be parsed."""


class TransactionError(OpenSIPError):
    """Transaction layer error (timeout, unexpected response, ...)."""


class AuthenticationError(OpenSIPError):
    """Digest authentication failed."""


class TransportError(OpenSIPError):
    """Transport / network error."""


class SDPError(OpenSIPError):
    """SDP parse or negotiation error."""
