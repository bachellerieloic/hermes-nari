"""Nari Labs speech-to-text and text-to-speech for Hermes Agent."""

__version__ = "0.1.0"

from .plugin import register  # noqa: E402,F401

__all__ = ["register", "__version__"]
