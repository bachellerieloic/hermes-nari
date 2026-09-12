"""Hermes ``TTSProvider`` backed by Nari Labs."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from . import audio
from ._base import TTSBase
from .api import NariClient, Timeouts, Voice
from .config import (DISPLAY_NAME, KEYS_URL, PROVIDER_NAME, TTS_MODELS, TTSSettings, require_api_key,
                     resolve_api_key, resolve_tts_settings)
from .errors import NariError

logger = logging.getLogger(__name__)

ClientFactory = Callable[[TTSSettings], NariClient]
SETUP_BADGE = "free"
SETUP_TAG = "Hosted Qwen3 voices from Nari Labs (free tier, no GPU needed)"


def with_suffix_for(output_path: str, output_format: str) -> str:
    """``output_path`` with the extension Hermes expects for ``output_format``."""
    path = Path(output_path)
    return str(path.with_suffix(f".{output_format}")) if path.suffix.lower() != f".{output_format}" else output_path


class NariTTSProvider(TTSBase):
    def __init__(self, *, client_factory: Optional[ClientFactory] = None,
                 env: Optional[Mapping[str, str]] = None, config: Optional[Mapping[str, Any]] = None,
                 timeouts: Timeouts = Timeouts()) -> None:
        self._env = env if env is not None else os.environ
        self._config = config
        self._timeouts = timeouts
        self._client_factory = client_factory or self._default_client
        self._voice_cache: Dict[str, Tuple[Voice, ...]] = {}

    # ---- identity ---------------------------------------------------------------------

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def display_name(self) -> str:
        return DISPLAY_NAME

    def is_available(self) -> bool:
        try:
            return bool(resolve_api_key(self._env))
        except Exception:  # noqa: BLE001 - the contract says never raise
            return False

    def get_setup_schema(self) -> Dict[str, Any]:
        return {
            "name": DISPLAY_NAME, "badge": SETUP_BADGE, "tag": SETUP_TAG,
            "env_vars": [{"key": "NARI_API_KEY", "prompt": "Nari Labs API key", "url": KEYS_URL}],
        }

    # ---- catalog ----------------------------------------------------------------------

    def settings(self, *, voice: Optional[str] = None, model: Optional[str] = None,
                 output_format: Optional[str] = None) -> TTSSettings:
        return resolve_tts_settings(voice=voice, model=model, output_format=output_format,
                                    config=self._config, env=self._env)

    def list_models(self) -> List[Dict[str, Any]]:
        return [m.as_dict() for m in TTS_MODELS]

    def default_model(self) -> Optional[str]:
        return self.settings().model

    def default_voice(self) -> Optional[str]:
        return self.settings().voice

    def voices(self, model: Optional[str] = None) -> Tuple[Voice, ...]:
        """Voice catalog for ``model`` (cached per model for the life of the provider). Raises."""
        settings = self.settings(model=model)
        cached = self._voice_cache.get(settings.model)
        if cached is not None:
            return cached
        voices = self._client_factory(settings).list_voices(settings.model)
        self._voice_cache = {**self._voice_cache, settings.model: voices}
        return voices

    def list_voices(self) -> List[Dict[str, Any]]:
        try:
            return [v.as_hermes_dict() for v in self.voices()]
        except NariError as exc:
            logger.warning("Could not list Nari voices: %s", exc)
            return []

    # ---- synthesis --------------------------------------------------------------------

    def synthesize(self, text: str, output_path: str, *, voice: Optional[str] = None,
                   model: Optional[str] = None, speed: Optional[float] = None, format: str = "mp3",
                   **extra: Any) -> str:
        settings = self.settings(voice=voice, model=model, output_format=format)
        if speed not in (None, 1, 1.0):
            logger.debug("Nari has no speech-rate control; ignoring speed=%s", speed)
        client = self._client_factory(settings)
        pcm = client.synthesize_pcm(text, model=settings.model, voice=settings.voice)
        target = with_suffix_for(output_path, settings.output_format)
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        audio.encode_pcm(pcm, settings.output_format, target)
        logger.info("Nari spoke %d characters with %s/%s into %s", len(text), settings.model, settings.voice, target)
        return target

    def stream(self, text: str, *, voice: Optional[str] = None, model: Optional[str] = None,
               format: str = "opus", **extra: Any) -> Iterator[bytes]:
        settings = self.settings(voice=voice, model=model, output_format=format)
        client = self._client_factory(settings)
        pcm_chunks = client.stream_pcm(text, model=settings.model, voice=settings.voice)
        return audio.transcode_stream(pcm_chunks, settings.output_format)

    @property
    def voice_compatible(self) -> bool:
        return True

    # ---- helpers ----------------------------------------------------------------------

    def _default_client(self, settings: TTSSettings) -> NariClient:
        return NariClient(require_api_key(self._env), base_url=settings.base_url, timeouts=self._timeouts)
