"""Hermes ``TranscriptionProvider`` backed by Nari Labs realtime transcription."""

from __future__ import annotations

import importlib.util
import logging
import os
from typing import Any, Callable, Dict, List, Mapping, Optional

from . import audio
from ._base import TranscriptionBase
from .api import NariClient, Timeouts
from .config import (DISPLAY_NAME, KEYS_URL, PROVIDER_NAME, STT_MODELS, STTSettings, require_api_key,
                     resolve_api_key, resolve_stt_settings)
from .errors import NariError

logger = logging.getLogger(__name__)

ClientFactory = Callable[[STTSettings], NariClient]
SETUP_BADGE = "free"
SETUP_TAG = "Hosted Qwen3 ASR from Nari Labs over WebSocket (free tier)"
WEBSOCKETS_MODULE = "websockets"
EMPTY_TRANSCRIPT_MESSAGE = "Nari returned no transcript (silence, or audio it could not understand)"


def websockets_available() -> bool:
    return importlib.util.find_spec(WEBSOCKETS_MODULE) is not None


class NariTranscriptionProvider(TranscriptionBase):
    def __init__(self, *, client_factory: Optional[ClientFactory] = None,
                 env: Optional[Mapping[str, str]] = None, config: Optional[Mapping[str, Any]] = None,
                 timeouts: Timeouts = Timeouts()) -> None:
        self._env = env if env is not None else os.environ
        self._config = config
        self._timeouts = timeouts
        self._client_factory = client_factory or self._default_client

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return DISPLAY_NAME

    def is_available(self) -> bool:
        try:
            return bool(resolve_api_key(self._env)) and websockets_available()
        except Exception:  # noqa: BLE001 - the contract says never raise
            return False

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": DISPLAY_NAME, "badge": SETUP_BADGE, "tag": SETUP_TAG,
            "env_vars": [{"key": "NARI_API_KEY", "prompt": "Nari Labs API key", "url": KEYS_URL}],
        }

    def settings(self, *, model: Optional[str] = None, language: Optional[str] = None) -> STTSettings:
        return resolve_stt_settings(model=model, language=language, config=self._config, env=self._env)

    def list_models(self) -> List[Dict[str, Any]]:
        return [m.as_dict() for m in STT_MODELS]

    def default_model(self) -> Optional[str]:
        return self.settings().model

    def transcribe(self, file_path: str, *, model: Optional[str] = None, language: Optional[str] = None,
                   **extra: Any) -> Dict[str, Any]:
        try:
            settings = self.settings(model=model, language=language)
            pcm = audio.decode_to_pcm16(file_path)
            prompt = extra.get("prompt") if isinstance(extra.get("prompt"), str) else None
            result = self._client_factory(settings).transcribe_pcm16(
                pcm, model=settings.model, language=settings.language, prompt=prompt or None,
            )
        except NariError as exc:
            logger.warning("Nari transcription failed: %s", exc)
            return self._error(str(exc))
        except Exception as exc:  # noqa: BLE001 - the contract says never raise
            logger.exception("Nari transcription crashed")
            return self._error(f"Nari transcription failed: {exc}")
        if not result.transcript:
            return self._error(EMPTY_TRANSCRIPT_MESSAGE)
        logger.info("Nari transcribed %s (%s): %d characters", file_path, settings.model, len(result.transcript))
        return {"success": True, "transcript": result.transcript, "provider": PROVIDER_NAME}

    def _error(self, message: str) -> Dict[str, Any]:
        return {"success": False, "transcript": "", "error": message, "provider": PROVIDER_NAME}

    def _default_client(self, settings: STTSettings) -> NariClient:
        return NariClient(require_api_key(self._env), ws_url=settings.ws_url, timeouts=self._timeouts)
