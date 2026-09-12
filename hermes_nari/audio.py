"""Audio plumbing around ffmpeg.

Nari speaks and hears raw PCM: 24 kHz signed 16-bit little-endian mono out of text to speech,
16 kHz of the same into transcription. ffmpeg does every other conversion. WAV output needs no
ffmpeg at all; the standard library writes the header.
"""

from __future__ import annotations

import io
import os
import shutil
import struct
import subprocess
import threading
import wave
from types import MappingProxyType
from typing import Iterable, Iterator, Mapping, Optional, Tuple

from .errors import FfmpegFailedError, FfmpegMissingError

TTS_SAMPLE_RATE = 24_000
STT_SAMPLE_RATE = 16_000
CHANNELS = 1
SAMPLE_WIDTH = 2  # bytes per PCM16 sample

FFMPEG_ENV = "HERMES_NARI_FFMPEG"  # override the binary name or path
DEFAULT_FFMPEG = "ffmpeg"
DECODE_TIMEOUT_SECONDS = 120
ENCODE_TIMEOUT_SECONDS = 120
STREAM_READ_SIZE = 4096
STREAMING_WAV_DATA_SIZE = 0xFFFFFFFF  # unknown length, the convention for streamed WAV

FFMPEG_INSTALL_HINT = (
    "ffmpeg is required (apt install ffmpeg, brew install ffmpeg, or set "
    f"{FFMPEG_ENV}=/path/to/ffmpeg). The official Hermes Docker image already ships it."
)

# Encoder arguments per Hermes output format. ``-f`` is explicit so the container never
# depends on the file extension.
ENCODER_ARGS: Mapping[str, Tuple[str, ...]] = MappingProxyType({
    "mp3": ("-c:a", "libmp3lame", "-q:a", "4", "-f", "mp3"),
    "ogg": ("-c:a", "libopus", "-b:a", "48k", "-vbr", "on", "-f", "ogg"),
    "opus": ("-c:a", "libopus", "-b:a", "48k", "-vbr", "on", "-f", "opus"),
    "flac": ("-c:a", "flac", "-f", "flac"),
})


def ffmpeg_path() -> str:
    """Absolute path of the ffmpeg binary; raises :class:`FfmpegMissingError` when absent."""
    candidate = os.environ.get(FFMPEG_ENV, "").strip() or DEFAULT_FFMPEG
    found = shutil.which(candidate)
    if not found:
        raise FfmpegMissingError(f"{candidate} was not found on PATH. {FFMPEG_INSTALL_HINT}")
    return found


def has_ffmpeg() -> bool:
    try:
        ffmpeg_path()
    except FfmpegMissingError:
        return False
    return True


def _stderr_text(data: Optional[bytes]) -> str:
    return (data or b"").decode("utf-8", errors="replace").strip()[:400] or "no stderr"


def _run(args: Tuple[str, ...], *, stdin: Optional[bytes], timeout: int, what: str) -> bytes:
    try:
        completed = subprocess.run(
            (ffmpeg_path(),) + args, input=stdin, capture_output=True, timeout=timeout, check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise FfmpegFailedError(f"ffmpeg timed out after {timeout}s while {what}") from exc
    except OSError as exc:
        raise FfmpegFailedError(f"ffmpeg could not be started while {what}: {exc}") from exc
    if completed.returncode != 0:
        raise FfmpegFailedError(
            f"ffmpeg failed while {what} (exit {completed.returncode}): {_stderr_text(completed.stderr)}"
        )
    return completed.stdout


def decode_to_pcm16(path: str, *, sample_rate: int = STT_SAMPLE_RATE,
                    timeout: int = DECODE_TIMEOUT_SECONDS) -> bytes:
    """Any audio file to mono PCM16 at ``sample_rate`` (what Nari transcription expects)."""
    if not os.path.isfile(path):
        raise FfmpegFailedError(f"audio file not found: {path}")
    pcm = _run(
        ("-v", "error", "-nostdin", "-i", path, "-ac", str(CHANNELS), "-ar", str(sample_rate),
         "-f", "s16le", "-"),
        stdin=None, timeout=timeout, what=f"decoding {path}",
    )
    if len(pcm) % SAMPLE_WIDTH:
        pcm = pcm[:-1]
    if not pcm:
        raise FfmpegFailedError(f"{path} decoded to zero samples")
    return pcm


def write_pcm_as_wav(pcm: bytes, path: str, *, sample_rate: int = TTS_SAMPLE_RATE) -> str:
    with wave.open(path, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return path


def wav_bytes(pcm: bytes, *, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as handle:
        handle.setnchannels(CHANNELS)
        handle.setsampwidth(SAMPLE_WIDTH)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm)
    return buffer.getvalue()


def streaming_wav_header(*, sample_rate: int = TTS_SAMPLE_RATE) -> bytes:
    """A 44-byte WAV header with unknown data length, for progressive delivery."""
    byte_rate = sample_rate * CHANNELS * SAMPLE_WIDTH
    block_align = CHANNELS * SAMPLE_WIDTH
    return b"".join((
        b"RIFF", struct.pack("<I", STREAMING_WAV_DATA_SIZE), b"WAVE",
        b"fmt ", struct.pack("<IHHIIHH", 16, 1, CHANNELS, sample_rate, byte_rate, block_align, SAMPLE_WIDTH * 8),
        b"data", struct.pack("<I", STREAMING_WAV_DATA_SIZE),
    ))


def pcm_duration_seconds(pcm: bytes, *, sample_rate: int = TTS_SAMPLE_RATE) -> float:
    return len(pcm) / (sample_rate * CHANNELS * SAMPLE_WIDTH)


def _pcm_input_args(sample_rate: int) -> Tuple[str, ...]:
    return ("-v", "error", "-f", "s16le", "-ar", str(sample_rate), "-ac", str(CHANNELS), "-i", "pipe:0")


def encode_pcm(pcm: bytes, output_format: str, output_path: str, *,
               sample_rate: int = TTS_SAMPLE_RATE, timeout: int = ENCODE_TIMEOUT_SECONDS) -> str:
    """Write ``pcm`` to ``output_path`` in ``output_format`` and return the path."""
    if output_format == "wav":
        return write_pcm_as_wav(pcm, output_path, sample_rate=sample_rate)
    encoder = ENCODER_ARGS.get(output_format)
    if encoder is None:
        raise FfmpegFailedError(f"unsupported output format {output_format!r}")
    _run(_pcm_input_args(sample_rate) + encoder + ("-y", output_path),
         stdin=pcm, timeout=timeout, what=f"encoding {output_format}")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) == 0:
        raise FfmpegFailedError(f"ffmpeg produced no {output_format} output at {output_path}")
    return output_path


def transcode_stream(pcm_chunks: Iterable[bytes], output_format: str, *,
                     sample_rate: int = TTS_SAMPLE_RATE) -> Iterator[bytes]:
    """Turn a PCM chunk stream into ``output_format`` bytes as they become available."""
    if output_format == "pcm":
        yield from pcm_chunks
        return
    if output_format == "wav":
        yield streaming_wav_header(sample_rate=sample_rate)
        yield from pcm_chunks
        return
    encoder = ENCODER_ARGS.get(output_format)
    if encoder is None:
        raise FfmpegFailedError(f"unsupported output format {output_format!r}")
    process = subprocess.Popen(
        (ffmpeg_path(),) + _pcm_input_args(sample_rate) + encoder + ("pipe:1",),
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    feed_errors = []

    def feed() -> None:
        try:
            for chunk in pcm_chunks:
                if chunk:
                    process.stdin.write(chunk)
        except Exception as exc:  # noqa: BLE001 - surfaced after the read loop
            feed_errors.append(exc)
        finally:
            try:
                process.stdin.close()
            except OSError:
                pass

    feeder = threading.Thread(target=feed, name="hermes-nari-ffmpeg-feed", daemon=True)
    feeder.start()
    try:
        while True:
            data = process.stdout.read(STREAM_READ_SIZE)
            if not data:
                break
            yield data
    finally:
        feeder.join(timeout=ENCODE_TIMEOUT_SECONDS)
        stderr = process.stderr.read()
        process.stdout.close()
        process.stderr.close()
        returncode = process.wait(timeout=ENCODE_TIMEOUT_SECONDS)
    if feed_errors:
        raise feed_errors[0]
    if returncode != 0:
        raise FfmpegFailedError(f"ffmpeg failed while streaming {output_format} (exit {returncode}): {_stderr_text(stderr)}")
