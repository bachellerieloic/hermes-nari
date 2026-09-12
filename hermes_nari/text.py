"""Split text into pieces Nari accepts: 1 to 2048 Unicode code points each."""

from __future__ import annotations

import re
from typing import Tuple

MAX_INPUT_CODE_POINTS = 2048

# A sentence ends at ., !, ? or an ellipsis, optionally followed by a closing quote or bracket.
_SENTENCE_END = re.compile(r"(?<=[.!?…])\s+|(?<=[.!?…][\"'”’)\]])\s+")


def _pack(units: Tuple[str, ...], limit: int) -> Tuple[str, ...]:
    """Greedily join units with single spaces without exceeding ``limit`` code points."""
    chunks: Tuple[str, ...] = ()
    current = ""
    for unit in units:
        candidate = f"{current} {unit}" if current else unit
        if len(candidate) <= limit:
            current = candidate
            continue
        if current:
            chunks = chunks + (current,)
        current = unit
    return chunks + (current,) if current else chunks


def _split_oversized(unit: str, limit: int) -> Tuple[str, ...]:
    """Split one unit longer than ``limit``: by whitespace first, then by code points."""
    words = tuple(w for w in unit.split() if w)
    pieces: Tuple[str, ...] = ()
    for word in words:
        if len(word) <= limit:
            pieces = pieces + (word,)
            continue
        pieces = pieces + tuple(word[i:i + limit] for i in range(0, len(word), limit))
    return _pack(pieces, limit)


def split_text(text: str, limit: int = MAX_INPUT_CODE_POINTS) -> Tuple[str, ...]:
    """Return trimmed, non-empty chunks of at most ``limit`` code points.

    Splitting happens at sentence boundaries; a sentence that is itself too long is split at
    whitespace, and a single oversized word is cut at the limit. Returns an empty tuple for
    whitespace-only input.
    """
    if limit < 1:
        raise ValueError("limit must be at least 1")
    stripped = text.strip()
    if not stripped:
        return ()
    if len(stripped) <= limit:
        return (stripped,)
    sentences = tuple(s.strip() for s in _SENTENCE_END.split(stripped) if s.strip())
    units: Tuple[str, ...] = ()
    for sentence in sentences:
        units = units + ((sentence,) if len(sentence) <= limit else _split_oversized(sentence, limit))
    return _pack(units, limit)
