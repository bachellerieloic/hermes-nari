"""Import the Hermes provider base classes, or degrade to plain objects outside Hermes.

Hermes type-checks providers at registration, so inside Hermes the real classes must import.
Outside Hermes (the ``hermes-nari`` CLI, the test suite) plain ``object`` bases keep the
adapters usable; ``HERMES_ABCS_AVAILABLE`` tells ``register`` to warn loudly if it ever runs
in that state.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from agent.transcription_provider import TranscriptionProvider as TranscriptionBase  # type: ignore[import-not-found]
    from agent.tts_provider import TTSProvider as TTSBase  # type: ignore[import-not-found]
    HERMES_ABCS_AVAILABLE = True
    IMPORT_PROBLEM = ""
except Exception as exc:  # noqa: BLE001 - ImportError normally, but never let Hermes die here
    TTSBase = object  # type: ignore[assignment,misc]
    TranscriptionBase = object  # type: ignore[assignment,misc]
    HERMES_ABCS_AVAILABLE = False
    IMPORT_PROBLEM = f"{type(exc).__name__}: {exc}"
    logger.debug("Hermes provider base classes unavailable (%s); using duck-typed bases", IMPORT_PROBLEM)
