"""The ``hermes-nari`` command: ``doctor``, ``voices`` and ``selftest``."""

from __future__ import annotations

import argparse
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Callable, Optional, Sequence

from . import __version__, audio
from .config import (API_KEY_ENV, KEYS_URL, env_file_candidates, is_shadowed_by_command_provider,
                     load_hermes_config, resolve_api_key, resolve_stt_settings, resolve_tts_settings)
from .errors import NariError
from .stt import NariTranscriptionProvider, websockets_available
from .tts import NariTTSProvider

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_NOT_CONFIGURED = 2
DEFAULT_SELFTEST_TEXT = "Hermes here. The Nari voice round trip is working."
SELFTEST_FILE = "hermes-nari-selftest.wav"
WORD_RE = re.compile(r"[a-z0-9']+")

Printer = Callable[[str], None]


def _print(line: str = "") -> None:
    print(line)


def _mask(key: str) -> str:
    return f"{key[:4]}...{key[-2:]}" if len(key) > 8 else "set"


def cmd_doctor(_: argparse.Namespace, out: Printer = _print) -> int:
    key = resolve_api_key()
    config = load_hermes_config()
    tts = resolve_tts_settings(config=config)
    stt = resolve_stt_settings(config=config)
    out(f"hermes-nari {__version__}")
    out(f"{API_KEY_ENV}: {_mask(key) if key else 'MISSING (looked in env, ' + ', '.join(map(str, env_file_candidates())) + ')'}")
    out(f"ffmpeg: {'found' if audio.has_ffmpeg() else 'MISSING (needed for mp3/ogg/opus/flac output and for every transcription)'}")
    out(f"websockets: {'installed' if websockets_available() else 'MISSING (add the websockets package to the Hermes environment)'}")
    out(f"tts: model={tts.model} voice={tts.voice} output_format={tts.output_format} base_url={tts.base_url}")
    out(f"stt: model={stt.model} language={stt.language or 'auto'} ws_url={stt.ws_url}")
    for section in ("tts", "stt"):
        if is_shadowed_by_command_provider(config, section):
            out(f"WARNING: {section}.providers.nari is a command provider; it shadows this plugin. Delete that block.")
    return EXIT_OK if key else EXIT_NOT_CONFIGURED


def cmd_voices(args: argparse.Namespace, out: Printer = _print) -> int:
    if not resolve_api_key():
        out(f"{API_KEY_ENV} is not set; create one at {KEYS_URL}.")
        return EXIT_NOT_CONFIGURED
    provider = NariTTSProvider()
    try:
        voices = provider.voices(args.model)
    except NariError as exc:
        out(str(exc))
        return EXIT_FAILED
    out(f"{len(voices)} voices for {provider.settings(model=args.model).model}")
    for voice in voices:
        extras = ", ".join(x for x in (voice.language, voice.gender, voice.description) if x)
        out(f"  {voice.id:<12} {voice.display}{'  (' + extras + ')' if extras else ''}")
    return EXIT_OK


def _words(text: str) -> set:
    return set(WORD_RE.findall(text.lower()))


def cmd_selftest(args: argparse.Namespace, out: Printer = _print) -> int:
    if not resolve_api_key():
        out(f"{API_KEY_ENV} is not set; the self test talks to the real Nari API and needs a key from {KEYS_URL}.")
        return EXIT_NOT_CONFIGURED
    keep_dir = Path(args.keep).expanduser() if args.keep else None
    workdir = keep_dir if keep_dir else Path(tempfile.mkdtemp(prefix="hermes-nari-"))
    workdir.mkdir(parents=True, exist_ok=True)
    wav_path = workdir / SELFTEST_FILE
    tts = NariTTSProvider()
    stt = NariTranscriptionProvider()
    out(f"text:       {args.text}")
    started = time.perf_counter()
    try:
        written = tts.synthesize(args.text, str(wav_path), voice=args.voice, model=args.tts_model, format="wav")
    except NariError as exc:
        out(f"TTS failed: {exc}")
        return EXIT_FAILED
    tts_seconds = time.perf_counter() - started
    settings = tts.settings(voice=args.voice, model=args.tts_model)
    out(f"tts:        {settings.model}/{settings.voice} -> {written} ({os.path.getsize(written)} bytes) in {tts_seconds:.2f}s")
    started = time.perf_counter()
    result = stt.transcribe(written, model=args.stt_model, language=args.language)
    stt_seconds = time.perf_counter() - started
    if not result["success"]:
        out(f"STT failed: {result['error']}")
        return EXIT_FAILED
    transcript = result["transcript"]
    expected, heard = _words(args.text), _words(transcript)
    overlap = len(expected & heard) / len(expected) if expected else 0.0
    out(f"stt:        {stt.settings(model=args.stt_model).model} -> {transcript!r} in {stt_seconds:.2f}s")
    out(f"round trip: {overlap:.0%} of the words came back ({tts_seconds + stt_seconds:.2f}s total)")
    if keep_dir:
        out(f"audio kept in {workdir}")
    else:
        Path(written).unlink(missing_ok=True)
    return EXIT_OK


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hermes-nari", description="Nari Labs voice for Hermes Agent")
    parser.add_argument("--version", action="version", version=f"hermes-nari {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)
    doctor = sub.add_parser("doctor", help="show the resolved settings without touching the network")
    doctor.set_defaults(func=cmd_doctor)
    voices = sub.add_parser("voices", help="list the voices Nari offers for a model")
    voices.add_argument("--model", help="TTS model id (default: the configured one)")
    voices.set_defaults(func=cmd_voices)
    selftest = sub.add_parser("selftest", help="speak a sentence, transcribe it back, print timings (real API)")
    selftest.add_argument("--text", default=DEFAULT_SELFTEST_TEXT)
    selftest.add_argument("--voice")
    selftest.add_argument("--tts-model", dest="tts_model")
    selftest.add_argument("--stt-model", dest="stt_model")
    selftest.add_argument("--language", help="language hint for transcription (default: configured, else en)")
    selftest.add_argument("--keep", metavar="DIR", help="keep the generated audio in DIR")
    selftest.set_defaults(func=cmd_selftest)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except NariError as exc:
        _print(str(exc))
        return EXIT_FAILED
    except KeyboardInterrupt:
        return EXIT_FAILED


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
