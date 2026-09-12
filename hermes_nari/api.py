"""Nari Labs API client: speech over HTTPS, transcription over WebSocket.

Only the standard library plus ``websockets`` are used, so the plugin installs cleanly inside
the Hermes environment (which already ships ``websockets``).
"""

from __future__ import annotations

import asyncio
import base64
import http.client
import json
import logging
import socket
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Coroutine, Dict, Iterator, Mapping, Optional, Tuple, TypeVar

from . import __version__
from .config import DEFAULT_BASE_URL, DEFAULT_WS_URL
from .errors import (NariConnectionError, NariError, NariProtocolError, error_from_event,
                     error_from_http)
from .text import MAX_INPUT_CODE_POINTS, split_text

logger = logging.getLogger(__name__)

SPEECH_PATH = "/v1/audio/speech"
VOICES_PATH = "/v1/voices"
USER_AGENT = f"hermes-nari/{__version__}"

PCM_FORMAT = "pcm"
HTTP_READ_SIZE = 4096

# Transcription framing: Nari caps a WebSocket message at 128 KiB including the JSON envelope.
# 48 KiB of PCM becomes 64 KiB of base64, leaving ample room.
WS_MESSAGE_LIMIT_BYTES = 128 * 1024
AUDIO_FRAME_PCM_BYTES = 48 * 1024
END_OF_INPUT_EVENT_ID = "end_of_input"

EVENT_SESSION_CONFIGURED = "session.configured"
EVENT_COMMITTED = "input_audio_buffer.committed"
EVENT_COMMIT_EMPTY = "input_audio_buffer.commit_empty"
EVENT_PARTIAL = "transcript.partial"
EVENT_COMPLETED = "transcript.completed"
EVENT_ERROR = "error"

T = TypeVar("T")


@dataclass(frozen=True)
class Timeouts:
    """Seconds. ``tts`` covers connect plus each read of one speech request."""

    tts: float = 120.0
    voices: float = 15.0
    ws_open: float = 20.0
    ws_configure: float = 30.0
    ws_result: float = 120.0


@dataclass(frozen=True)
class Voice:
    id: str
    display: str
    language: Optional[str] = None
    gender: Optional[str] = None
    preview_url: Optional[str] = None
    description: Optional[str] = None

    def as_hermes_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "display": self.display, "language": self.language,
                "gender": self.gender, "preview_url": self.preview_url}


@dataclass(frozen=True)
class TranscriptionResult:
    transcript: str
    language: Optional[str] = None
    utterances: Tuple[str, ...] = field(default_factory=tuple)
    partials: Tuple[str, ...] = field(default_factory=tuple)
    audio_seconds: Optional[float] = None


def run_sync(coroutine_factory: Callable[[], Coroutine[Any, Any, T]]) -> T:
    """Run a coroutine from synchronous code, even when this thread already has a loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine_factory())
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="hermes-nari-ws") as pool:
        return pool.submit(lambda: asyncio.run(coroutine_factory())).result()


def audio_frames(pcm: bytes, *, frame_bytes: int = AUDIO_FRAME_PCM_BYTES) -> Iterator[str]:
    """Serialised ``input_audio_buffer.append`` messages, each well under the 128 KiB cap."""
    if frame_bytes < 2 or frame_bytes % 2:
        raise ValueError("frame_bytes must be an even number of bytes of at least 2")
    usable = pcm[:-1] if len(pcm) % 2 else pcm
    for start in range(0, len(usable), frame_bytes):
        chunk = usable[start:start + frame_bytes]
        yield json.dumps({"type": "input_audio_buffer.append",
                          "audio": base64.b64encode(chunk).decode("ascii")})


def _voice_from_entry(entry: Mapping[str, Any]) -> Optional[Voice]:
    voice_id = entry.get("id")
    if not isinstance(voice_id, str) or not voice_id:
        return None
    display = entry.get("display_name") or entry.get("display") or entry.get("name") or voice_id
    return Voice(
        id=voice_id, display=str(display), language=entry.get("language"), gender=entry.get("gender"),
        preview_url=entry.get("preview_url"), description=entry.get("description"),
    )


def parse_voices(payload: Any) -> Tuple[Voice, ...]:
    """Accept a bare list or an object holding the list under ``voices``/``data``/any list key."""
    entries = payload
    if isinstance(payload, dict):
        entries = next((payload[k] for k in ("voices", "data", "items") if isinstance(payload.get(k), list)),
                       next((v for v in payload.values() if isinstance(v, list)), []))
    if not isinstance(entries, list):
        raise NariProtocolError("voice catalog response is not a list")
    voices = tuple(v for v in (_voice_from_entry(e) for e in entries if isinstance(e, dict)) if v)
    return voices


class NariClient:
    def __init__(self, api_key: str, *, base_url: str = DEFAULT_BASE_URL, ws_url: str = DEFAULT_WS_URL,
                 timeouts: Timeouts = Timeouts()) -> None:
        if not api_key:
            raise ValueError("api_key is required")
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._ws_url = ws_url
        self._timeouts = timeouts

    # ---- HTTP ---------------------------------------------------------------------------

    def _headers(self, *, json_body: bool) -> Dict[str, str]:
        headers = {"Authorization": f"Bearer {self._api_key}", "User-Agent": USER_AGENT}
        if json_body:
            headers["Content-Type"] = "application/json"
        return headers

    def _open(self, request: urllib.request.Request, *, timeout: float, what: str):
        try:
            return urllib.request.urlopen(request, timeout=timeout)  # noqa: S310 - https to a fixed host
        except urllib.error.HTTPError as exc:
            body = exc.read() if hasattr(exc, "read") else b""
            raise error_from_http(exc.code, body, what=what) from None
        except (urllib.error.URLError, socket.timeout, TimeoutError, OSError, http.client.HTTPException) as exc:
            reason = getattr(exc, "reason", None) or exc
            raise NariConnectionError(f"{what} could not reach {self._base_url}: {reason}") from exc

    def list_voices(self, model: str) -> Tuple[Voice, ...]:
        query = urllib.parse.urlencode({"model": model})
        request = urllib.request.Request(f"{self._base_url}{VOICES_PATH}?{query}",
                                         headers=self._headers(json_body=False), method="GET")
        with self._open(request, timeout=self._timeouts.voices, what="Listing Nari voices") as response:
            raw = response.read()
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (ValueError, UnicodeDecodeError) as exc:
            raise NariProtocolError(f"voice catalog is not JSON: {exc}") from exc
        return parse_voices(payload)

    def _speech_request(self, text: str, *, model: str, voice: str, stream: bool,
                        language: Optional[str] = None) -> urllib.request.Request:
        if len(text) < 1 or len(text) > MAX_INPUT_CODE_POINTS:
            raise ValueError(f"speech input must be 1 to {MAX_INPUT_CODE_POINTS} code points, got {len(text)}")
        body: Dict[str, Any] = {"model": model, "voice": voice, "input": text,
                                "response_format": PCM_FORMAT, "stream": stream}
        if language:
            body["language"] = language
        return urllib.request.Request(
            f"{self._base_url}{SPEECH_PATH}", data=json.dumps(body).encode("utf-8"),
            headers=self._headers(json_body=True), method="POST",
        )

    def _iter_speech_body(self, request: urllib.request.Request) -> Iterator[bytes]:
        with self._open(request, timeout=self._timeouts.tts, what="Nari speech") as response:
            read = getattr(response, "read1", response.read)  # read1 returns bytes as they arrive
            try:
                while True:
                    data = read(HTTP_READ_SIZE)
                    if not data:
                        return
                    yield data
            except (http.client.IncompleteRead, socket.timeout, TimeoutError, OSError) as exc:
                raise NariConnectionError(f"Nari speech stream was interrupted: {exc}") from exc

    def synthesize_pcm(self, text: str, *, model: str, voice: str, language: Optional[str] = None) -> bytes:
        """Complete 24 kHz PCM16 mono audio for ``text``; long text is split and concatenated."""
        pieces = split_text(text)
        if not pieces:
            raise ValueError("speech input is empty")
        audio = []
        for piece in pieces:
            request = self._speech_request(piece, model=model, voice=voice, stream=False, language=language)
            audio.append(b"".join(self._iter_speech_body(request)))
        pcm = b"".join(audio)
        if not pcm:
            raise NariProtocolError("Nari returned no audio")
        return pcm

    def stream_pcm(self, text: str, *, model: str, voice: str, language: Optional[str] = None) -> Iterator[bytes]:
        """PCM16 chunks as Nari produces them; pieces of long text play back to back."""
        pieces = split_text(text)
        if not pieces:
            raise ValueError("speech input is empty")
        for piece in pieces:
            request = self._speech_request(piece, model=model, voice=voice, stream=True, language=language)
            yield from self._iter_speech_body(request)

    # ---- WebSocket transcription --------------------------------------------------------

    def transcribe_pcm16(self, pcm: bytes, *, model: str, language: Optional[str] = None,
                         prompt: Optional[str] = None,
                         on_partial: Optional[Callable[[str], None]] = None) -> TranscriptionResult:
        """Transcribe 16 kHz PCM16 mono audio; blocks until the final transcript arrives."""
        if not pcm:
            raise ValueError("no audio to transcribe")
        return run_sync(lambda: self._transcribe_async(pcm, model=model, language=language,
                                                       prompt=prompt, on_partial=on_partial))

    def _session_message(self, model: str, language: Optional[str], prompt: Optional[str]) -> str:
        session: Dict[str, Any] = {"model": model, "language": language or None, "turn_detection": None}
        if prompt:
            session["prompt"] = prompt
        return json.dumps({"type": "session.configure", "session": session})

    async def _transcribe_async(self, pcm: bytes, *, model: str, language: Optional[str],
                                prompt: Optional[str], on_partial: Optional[Callable[[str], None]]) -> TranscriptionResult:
        try:
            from websockets.asyncio.client import connect
            from websockets.exceptions import ConnectionClosed, InvalidStatus, WebSocketException
        except ImportError as exc:
            raise NariConnectionError("the 'websockets' package (14 or newer) is required for transcription; "
                                      "add it to the Hermes environment") from exc
        headers = {"Authorization": f"Bearer {self._api_key}", "User-Agent": USER_AGENT}
        try:
            async with connect(self._ws_url, additional_headers=headers, max_size=None,
                               open_timeout=self._timeouts.ws_open) as socket_:
                return await self._run_session(socket_, pcm, model=model, language=language,
                                               prompt=prompt, on_partial=on_partial, closed_error=ConnectionClosed)
        except InvalidStatus as exc:
            raise error_from_http(exc.response.status_code, bytes(exc.response.body or b""),
                                  what="Nari transcription handshake") from None
        except NariError:
            raise
        except (WebSocketException, OSError, asyncio.TimeoutError, TimeoutError) as exc:
            raise NariConnectionError(f"Nari transcription connection failed: {exc}") from exc

    async def _receive(self, socket_, timeout: float) -> Mapping[str, Any]:
        raw = await asyncio.wait_for(socket_.recv(), timeout)
        try:
            event = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise NariProtocolError(f"Nari sent a non-JSON message: {exc}") from exc
        if not isinstance(event, dict):
            raise NariProtocolError("Nari sent a non-object message")
        if event.get("type") == EVENT_ERROR:
            raise error_from_event(event)
        return event

    async def _run_session(self, socket_, pcm: bytes, *, model: str, language: Optional[str],
                           prompt: Optional[str], on_partial: Optional[Callable[[str], None]],
                           closed_error: type) -> TranscriptionResult:
        await socket_.send(self._session_message(model, language, prompt))
        try:
            configured = await self._receive(socket_, self._timeouts.ws_configure)
        except asyncio.TimeoutError as exc:
            raise NariConnectionError("Nari did not acknowledge session.configure in time") from exc
        if configured.get("type") != EVENT_SESSION_CONFIGURED:
            raise NariProtocolError(f"expected {EVENT_SESSION_CONFIGURED}, got {configured.get('type')!r}")
        for frame in audio_frames(pcm):
            await socket_.send(frame)
        await socket_.send(json.dumps({"type": "input_audio_buffer.commit", "event_id": END_OF_INPUT_EVENT_ID}))
        return await self._collect(socket_, on_partial=on_partial, closed_error=closed_error)

    async def _collect(self, socket_, *, on_partial: Optional[Callable[[str], None]],
                       closed_error: type) -> TranscriptionResult:
        pending: set = set()
        completed: Dict[str, str] = {}
        order: list = []
        partials: list = []
        language: Optional[str] = None
        audio_seconds = 0.0
        end_acknowledged = False
        while True:
            try:
                event = await self._receive(socket_, self._timeouts.ws_result)
            except asyncio.TimeoutError as exc:
                raise NariConnectionError("timed out waiting for the Nari transcript") from exc
            except closed_error as exc:
                if end_acknowledged and not pending:
                    break
                detail = f" (last partial: {partials[-1]!r})" if partials else ""
                raise NariConnectionError(f"Nari closed the transcription session early: {exc}{detail}") from exc
            kind = event.get("type")
            item_id = str(event.get("item_id", ""))
            if kind == EVENT_COMMITTED:
                pending.add(item_id)
                if item_id not in order:
                    order.append(item_id)
            elif kind == EVENT_PARTIAL:
                text = str(event.get("transcript") or "")
                partials.append(text)
                if on_partial is not None:
                    on_partial(text)
            elif kind == EVENT_COMPLETED:
                pending.discard(item_id)
                if item_id not in order:
                    order.append(item_id)
                completed[item_id] = str(event.get("transcript") or "").strip()
                language = event.get("language") or language
                usage = event.get("usage") if isinstance(event.get("usage"), dict) else {}
                audio_seconds += float(usage.get("input_audio_seconds") or 0.0)
            if kind in (EVENT_COMMITTED, EVENT_COMMIT_EMPTY) and event.get("client_event_id") == END_OF_INPUT_EVENT_ID:
                end_acknowledged = True
            if end_acknowledged and not pending:
                break
        utterances = tuple(completed[i] for i in order if completed.get(i))
        return TranscriptionResult(
            transcript=" ".join(utterances).strip(), language=language, utterances=utterances,
            partials=tuple(partials), audio_seconds=audio_seconds or None,
        )
