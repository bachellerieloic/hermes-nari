"""Error types raised by the Nari client and the audio helpers.

Every error carries a message written for the person reading a Hermes log or a
Telegram reply, so the provider adapters can pass ``str(exc)`` straight through.
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Optional

KEYS_URL = "https://app.narilabs.com/keys"
DOCS_ERRORS_URL = "https://docs.narilabs.com/errors"
PARTNER_CONTACT = "founders@narilabs.com"

HTTP_UNAUTHORIZED = 401
HTTP_PAYMENT_REQUIRED = 402
HTTP_FORBIDDEN = 403
HTTP_TOO_MANY_REQUESTS = 429
HTTP_SERVER_ERROR_FLOOR = 500

# Advice keyed by Nari's UPPER_SNAKE_CASE error codes (HTTP and WebSocket).
_ADVICE: Mapping[str, str] = {
    "INVALID_API_KEY": f"Check NARI_API_KEY (keys live at {KEYS_URL}).",
    "CONCURRENCY_LIMIT_EXCEEDED": (
        "The free tier allows 2 in-flight requests per organisation; wait for one to finish and retry."
    ),
    "FREE_DAILY_LIMIT_EXCEEDED": (
        "The free daily allowance for this model is used up; it resets at 00:00 UTC, "
        f"or ask {PARTNER_CONTACT} for partner access."
    ),
    "UPSTREAM_RATE_LIMITED": "Nari is rate limited upstream; retry with backoff.",
    "INSUFFICIENT_CREDITS": "Add credits to the Nari organisation or switch to a :free model.",
    "PARTNER_ACCESS_REQUIRED": f"This model needs partner access; use a :free model or email {PARTNER_CONTACT}.",
    "MODEL_NOT_FOUND": "Use a model id from https://docs.narilabs.com/models-and-pricing.",
    "INVALID_VOICE": "Use an exact, case-sensitive voice id (run: hermes-nari voices).",
    "INVALID_VOICE_LANGUAGE": "Omit the language or match the voice's language.",
    "UNSUPPORTED_RESPONSE_FORMAT": "Nari only returns wav or pcm; this is a plugin bug, please report it.",
    "SESSION_SETUP_TIMEOUT": "The session was not configured within 10 seconds of connecting.",
    "SESSION_IDLE_TIMEOUT": "Nari closed the idle transcription session.",
}


class NariError(RuntimeError):
    """Base class for every failure this plugin reports."""

    def __init__(self, message: str, *, code: Optional[str] = None, status: Optional[int] = None,
                 request_id: Optional[str] = None) -> None:
        super().__init__(message)
        self.code = code
        self.status = status
        self.request_id = request_id


class NariConfigError(NariError):
    """Missing API key or an invalid setting."""


class NariAuthError(NariError):
    """HTTP 401: the API key was rejected."""


class NariRateLimitError(NariError):
    """HTTP 429: concurrency or daily allowance exceeded."""


class NariRequestError(NariError):
    """Any other 4xx: the request itself was rejected."""


class NariServerError(NariError):
    """5xx: Nari is having trouble; retry later."""


class NariConnectionError(NariError):
    """Network failure, timeout or an interrupted stream."""


class NariProtocolError(NariError):
    """The WebSocket conversation did not follow the documented shape."""


class FfmpegMissingError(NariError):
    """ffmpeg is not on PATH."""


class FfmpegFailedError(NariError):
    """ffmpeg ran but exited with an error."""


def parse_error_body(body: bytes | str | None) -> Mapping[str, Any]:
    """Return Nari's ``error`` object from a JSON body, or an empty mapping."""
    if not body:
        return {}
    try:
        text = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else body
        data = json.loads(text)
    except (ValueError, TypeError):
        return {}
    error = data.get("error") if isinstance(data, dict) else None
    return error if isinstance(error, dict) else {}


def _describe(code: Optional[str], message: Optional[str], request_id: Optional[str]) -> str:
    parts = [p for p in (code, message) if p]
    text = ": ".join(parts) if parts else "no error details"
    advice = _ADVICE.get(code or "")
    if advice:
        text = f"{text}. {advice}"
    if request_id:
        text = f"{text} (request {request_id})"
    return text


def error_from_http(status: int, body: bytes | str | None, *, what: str = "Nari request") -> NariError:
    """Map an HTTP failure to the matching :class:`NariError` subclass."""
    error = parse_error_body(body)
    code = error.get("code") if isinstance(error.get("code"), str) else None
    message = error.get("message") if isinstance(error.get("message"), str) else None
    request_id = error.get("requestId") if isinstance(error.get("requestId"), str) else None
    if not code and not message and body:
        raw = body.decode("utf-8", errors="replace") if isinstance(body, bytes) else str(body)
        message = raw.strip()[:200] or None
    detail = _describe(code, message, request_id)
    text = f"{what} failed with HTTP {status}: {detail}"
    kwargs = {"code": code, "status": status, "request_id": request_id}
    if status == HTTP_UNAUTHORIZED:
        return NariAuthError(text, **kwargs)
    if status == HTTP_TOO_MANY_REQUESTS:
        return NariRateLimitError(text, **kwargs)
    if status >= HTTP_SERVER_ERROR_FLOOR:
        return NariServerError(text, **kwargs)
    return NariRequestError(text, **kwargs)


def error_from_event(event: Mapping[str, Any], *, what: str = "Nari transcription") -> NariError:
    """Map a WebSocket ``error`` event to the matching :class:`NariError` subclass."""
    error = event.get("error") if isinstance(event.get("error"), dict) else {}
    code = error.get("code") if isinstance(error.get("code"), str) else None
    message = error.get("message") if isinstance(error.get("message"), str) else None
    request_id = error.get("requestId") if isinstance(error.get("requestId"), str) else None
    text = f"{what} failed: {_describe(code, message, request_id)}"
    kwargs = {"code": code, "request_id": request_id}
    if code == "INVALID_API_KEY":
        return NariAuthError(text, **kwargs)
    if code in {"CONCURRENCY_LIMIT_EXCEEDED", "FREE_DAILY_LIMIT_EXCEEDED", "UPSTREAM_RATE_LIMITED"}:
        return NariRateLimitError(text, **kwargs)
    if code in {"INTERNAL_ERROR", "UPSTREAM_UNAVAILABLE", "SERVER_DRAINING", "VAD_OVERLOADED",
                "VAD_UNAVAILABLE", "VAD_RUNTIME_ERROR", "REQUEST_GATE_UNAVAILABLE"}:
        return NariServerError(text, **kwargs)
    return NariRequestError(text, **kwargs)
