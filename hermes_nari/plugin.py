"""Hermes plugin entry point."""

from __future__ import annotations

import logging

from ._base import HERMES_ABCS_AVAILABLE, IMPORT_PROBLEM
from .config import PROVIDER_NAME

logger = logging.getLogger(__name__)


def register(ctx) -> None:
    """Register the Nari text-to-speech and speech-to-text providers with Hermes."""
    from .stt import NariTranscriptionProvider
    from .tts import NariTTSProvider

    if not HERMES_ABCS_AVAILABLE:
        logger.warning(
            "hermes-nari could not import agent.tts_provider / agent.transcription_provider (%s). "
            "Hermes will reject the providers below; this plugin needs Hermes 0.16.0 or newer.",
            IMPORT_PROBLEM,
        )
    ctx.register_tts_provider(NariTTSProvider())
    ctx.register_transcription_provider(NariTranscriptionProvider())
    logger.info("hermes-nari registered TTS and STT providers named %r", PROVIDER_NAME)
