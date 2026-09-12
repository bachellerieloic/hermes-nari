"""In-process stand-ins for the Nari Labs API: an HTTP server and a WebSocket server.

They speak the documented protocol closely enough to exercise the client end to end without
the network: bearer auth, the speech endpoint (wav/pcm, streamed or not, 2048-code-point
limit), the voice catalog, and the realtime transcription session with partial and completed
events, forced errors and early closes.
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

from hermes_nari import audio

MAX_CODE_POINTS = 2048
WS_MESSAGE_LIMIT = 128 * 1024
PCM_REPEAT = 2000
STREAM_PIECES = 4
STT_BYTES_PER_SECOND = 32_000


def pcm_for(text: str) -> bytes:
    """Deterministic even-length PCM stand-in for ``text``."""
    return hashlib.sha256(text.strip().encode("utf-8")).digest()[:8] * PCM_REPEAT


def default_transcript(pcm: bytes) -> str:
    return f"heard {len(pcm)} bytes"


DEFAULT_VOICES: Tuple[Dict[str, Any], ...] = (
    {"id": "leon", "display_name": "Leon", "language": "en", "gender": "male",
     "description": "Warm and steady", "preview_url": "https://example.invalid/leon.mp3"},
    {"id": "diana", "display_name": "Diana", "language": "en", "gender": "female",
     "description": "Measured and professional"},
    {"id": "gabriel", "display_name": "Gabriel", "language": "es", "gender": "male"},
)


class FakeState:
    """Mutable knobs and recordings shared by both fake servers (test-only object)."""

    def __init__(self) -> None:
        self.api_key = "test-key-0123456789"
        self.speech_requests: List[Dict[str, Any]] = []
        self.voice_requests: List[Dict[str, Any]] = []
        self.next_http_error: Optional[Tuple[int, str, str]] = None
        self.voices: Tuple[Dict[str, Any], ...] = DEFAULT_VOICES
        self.voices_shape = "object"  # or "list"
        self.stream_pieces = STREAM_PIECES
        self.ws_sessions: List[Dict[str, Any]] = []
        self.ws_reject: Optional[Tuple[int, str]] = None
        self.ws_error_after_configure: Optional[Tuple[str, str]] = None
        self.transcript_for: Callable[[bytes], str] = default_transcript
        self.utterance_bytes: Optional[int] = None
        self.partials_per_utterance = 2
        self.close_early = False


def error_body(code: str, message: str) -> bytes:
    return json.dumps({"error": {"code": code, "message": message, "requestId": str(uuid.uuid4())}}).encode()


# ---- HTTP -----------------------------------------------------------------------------------


def _make_handler(state: FakeState):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"

        def log_message(self, *_args: Any) -> None:  # keep pytest output clean
            return

        def _fail(self, status: int, code: str, message: str) -> None:
            body = error_body(code, message)
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _authorized(self) -> bool:
            if self.headers.get("Authorization") == f"Bearer {state.api_key}":
                return True
            self._fail(401, "INVALID_API_KEY", "The API key is missing or invalid")
            return False

        def do_GET(self) -> None:  # noqa: N802 - http.server API
            parsed = urlparse(self.path)
            if parsed.path != "/v1/voices":
                self._fail(404, "NOT_FOUND", "no such route")
                return
            if not self._authorized():
                return
            models = parse_qs(parsed.query).get("model", [])
            state.voice_requests.append({"model": models, "headers": dict(self.headers)})
            if len(models) != 1:
                self._fail(400, "INVALID_VOICES_REQUEST", "Provide exactly one model")
                return
            voices = list(state.voices)
            payload = voices if state.voices_shape == "list" else {"catalog_version": "test", "voices": voices}
            body = json.dumps(payload).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - http.server API
            if urlparse(self.path).path != "/v1/audio/speech":
                self._fail(404, "NOT_FOUND", "no such route")
                return
            if not self._authorized():
                return
            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            try:
                body = json.loads(raw)
            except ValueError:
                self._fail(400, "INVALID_SPEECH_REQUEST", "body is not JSON")
                return
            state.speech_requests.append({"body": body, "headers": dict(self.headers)})
            if state.next_http_error is not None:
                status, code, message = state.next_http_error
                state.next_http_error = None
                self._fail(status, code, message)
                return
            allowed = {"model", "voice", "input", "language", "response_format", "seed", "stream"}
            if set(body) - allowed or any(v is None for v in body.values()):
                self._fail(400, "INVALID_SPEECH_REQUEST", "unknown or null field")
                return
            text = str(body.get("input", "")).strip()
            if not (1 <= len(text) <= MAX_CODE_POINTS) or not body.get("model") or not body.get("voice"):
                self._fail(400, "INVALID_SPEECH_REQUEST", "The speech request is invalid")
                return
            fmt = body.get("response_format", "wav")
            if fmt not in ("wav", "pcm"):
                self._fail(400, "UNSUPPORTED_RESPONSE_FORMAT", "Use wav or pcm")
                return
            pcm = pcm_for(text)
            payload = pcm if fmt == "pcm" else audio.wav_bytes(pcm)
            self.send_response(200)
            self.send_header("Content-Type", "audio/pcm" if fmt == "pcm" else "audio/wav")
            self.send_header("x-request-id", str(uuid.uuid4()))
            self.end_headers()
            if body.get("stream"):
                step = max(1, len(payload) // state.stream_pieces)
                for start in range(0, len(payload), step):
                    self.wfile.write(payload[start:start + step])
                    self.wfile.flush()
                    time.sleep(0.03)
            else:
                self.wfile.write(payload)

    return Handler


class FakeNariHTTP:
    def __init__(self, state: FakeState) -> None:
        self._server = ThreadingHTTPServer(("127.0.0.1", 0), _make_handler(state))
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    @property
    def base_url(self) -> str:
        host, port = self._server.server_address[:2]
        return f"http://{host}:{port}"

    def start(self) -> "FakeNariHTTP":
        self._thread.start()
        return self

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()


# ---- WebSocket ------------------------------------------------------------------------------


def _split_audio(pcm: bytes, size: Optional[int]) -> List[bytes]:
    if not size or size <= 0:
        return [pcm]
    return [pcm[i:i + size] for i in range(0, len(pcm), size)]


class FakeNariWS:
    def __init__(self, state: FakeState) -> None:
        self._state = state
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._stop: Optional[asyncio.Event] = None
        self._ready = threading.Event()
        self._port = 0
        self._thread = threading.Thread(target=lambda: asyncio.run(self._serve()), daemon=True)

    @property
    def ws_url(self) -> str:
        return f"ws://127.0.0.1:{self._port}/v1/realtime?intent=transcription"

    def start(self) -> "FakeNariWS":
        self._thread.start()
        if not self._ready.wait(10):
            raise RuntimeError("fake websocket server did not start")
        return self

    def stop(self) -> None:
        if self._loop and self._stop:
            self._loop.call_soon_threadsafe(self._stop.set)
        self._thread.join(timeout=10)

    async def _serve(self) -> None:
        from websockets.asyncio.server import serve

        self._loop = asyncio.get_running_loop()
        self._stop = asyncio.Event()
        async with serve(self._handle, "127.0.0.1", 0, process_request=self._process_request,
                         max_size=None) as server:
            self._port = server.sockets[0].getsockname()[1]
            self._ready.set()
            await self._stop.wait()

    def _process_request(self, connection, request):
        state = self._state
        if state.ws_reject is not None:
            status, code = state.ws_reject
            return connection.respond(HTTPStatus(status), error_body(code, "rejected at handshake").decode())
        if request.headers.get("Authorization") != f"Bearer {state.api_key}":
            return connection.respond(HTTPStatus.UNAUTHORIZED, error_body("INVALID_API_KEY", "bad key").decode())
        return None

    async def _send(self, ws, payload: Dict[str, Any]) -> None:
        await ws.send(json.dumps(payload))

    async def _error(self, ws, code: str, message: str, close_code: int = 1008) -> None:
        await self._send(ws, {"type": "error", "event_id": "event_err",
                              "error": {"code": code, "message": message, "requestId": str(uuid.uuid4())}})
        await ws.close(close_code)

    async def _handle(self, ws) -> None:
        state = self._state
        session: Dict[str, Any] = {"session": None, "frames": [], "raw_sizes": [], "audio": bytearray(),
                                   "commits": [], "headers": {k.lower(): v for k, v in ws.request.headers.items()}}
        state.ws_sessions.append(session)
        first = json.loads(await ws.recv())
        session["first_message"] = first
        configured = first.get("session") if isinstance(first.get("session"), dict) else None
        if first.get("type") != "session.configure" or not configured or not configured.get("model"):
            await self._error(ws, "INVALID_REQUEST", "First event must be session.configure")
            return
        session["session"] = configured
        if state.ws_error_after_configure is not None:
            code, message = state.ws_error_after_configure
            await self._error(ws, code, message)
            return
        await self._send(ws, {"type": "session.configured",
                              "session": {**configured, "limits": {"max_utterance_seconds": 36, "idle_seconds": 60}}})
        commit_number = 0
        async for raw in ws:
            session["raw_sizes"].append(len(raw))
            if len(raw) > WS_MESSAGE_LIMIT:
                await self._error(ws, "INVALID_REQUEST", "message too large", close_code=1009)
                return
            message = json.loads(raw)
            kind = message.get("type")
            if kind == "input_audio_buffer.append":
                chunk = base64.b64decode(message.get("audio", ""))
                if not chunk or len(chunk) % 2:
                    await self._error(ws, "INVALID_REQUEST", "chunks must hold whole PCM16 samples")
                    return
                session["frames"].append(len(chunk))
                session["audio"] += chunk
            elif kind == "input_audio_buffer.commit":
                pcm = bytes(session["audio"])
                session["audio"] = bytearray()
                client_event_id = message.get("event_id")
                session["commits"].append(client_event_id)
                if not pcm:
                    await self._send(ws, {"type": "input_audio_buffer.commit_empty", "client_event_id": client_event_id})
                    continue
                parts = _split_audio(pcm, state.utterance_bytes)
                for index, part in enumerate(parts):
                    commit_number += 1
                    item_id = f"utterance_{commit_number}"
                    committed: Dict[str, Any] = {"type": "input_audio_buffer.committed", "item_id": item_id}
                    if index == len(parts) - 1:
                        committed["client_event_id"] = client_event_id
                    await self._send(ws, committed)
                    text = state.transcript_for(part)
                    for revision in range(1, state.partials_per_utterance + 1):
                        cut = max(1, len(text) * revision // (state.partials_per_utterance + 1))
                        await self._send(ws, {"type": "transcript.partial", "item_id": item_id,
                                              "transcript": text[:cut], "revision": revision})
                    if state.close_early:
                        await ws.close(1011)
                        return
                    await self._send(ws, {"type": "transcript.completed", "item_id": item_id, "transcript": text,
                                          "language": "en", "commit_reason": "manual" if index == len(parts) - 1 else "max_duration",
                                          "usage": {"input_audio_seconds": len(part) / STT_BYTES_PER_SECOND}})
            else:
                await self._error(ws, "INVALID_REQUEST", f"unknown event {kind!r}")
                return
