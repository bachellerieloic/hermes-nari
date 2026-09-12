import json
import os
import shutil
import struct
import wave

import pytest

from hermes_nari import audio
from hermes_nari.errors import FfmpegFailedError, FfmpegMissingError
from tests.support import REAL_FFMPEG

PCM = struct.pack("<" + "h" * 2400, *[(i * 37) % 65536 - 32768 for i in range(2400)])


def _calls(log):
    return [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]


def test_missing_ffmpeg_is_named_clearly(no_ffmpeg, tmp_path):
    with pytest.raises(FfmpegMissingError, match="ffmpeg"):
        audio.ffmpeg_path()
    assert audio.has_ffmpeg() is False
    with pytest.raises(FfmpegMissingError, match="apt install ffmpeg"):
        audio.encode_pcm(PCM, "mp3", str(tmp_path / "x.mp3"))


def test_binary_override_env(fake_ffmpeg, monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_NARI_FFMPEG", "definitely-not-a-binary")
    with pytest.raises(FfmpegMissingError, match="definitely-not-a-binary"):
        audio.ffmpeg_path()


def test_wav_output_needs_no_ffmpeg(no_ffmpeg, tmp_path):
    path = str(tmp_path / "out.wav")
    assert audio.encode_pcm(PCM, "wav", path) == path
    with wave.open(path, "rb") as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (1, 2, 24_000)
        assert handle.readframes(handle.getnframes()) == PCM
    assert audio.pcm_duration_seconds(PCM) == pytest.approx(0.1)


def test_wav_bytes_round_trip():
    data = audio.wav_bytes(PCM)
    assert data[:4] == b"RIFF" and data[8:12] == b"WAVE" and data[44:] == PCM


@pytest.mark.parametrize("fmt,codec", [("mp3", "libmp3lame"), ("ogg", "libopus"), ("opus", "libopus"), ("flac", "flac")])
def test_encode_uses_the_right_encoder_arguments(fake_ffmpeg, tmp_path, fmt, codec):
    path = str(tmp_path / f"out.{fmt}")
    assert audio.encode_pcm(PCM, fmt, path) == path
    (call,) = _calls(fake_ffmpeg)
    assert call[:9] == ["-v", "error", "-f", "s16le", "-ar", "24000", "-ac", "1", "-i"]
    assert call[9] == "pipe:0"
    assert "-c:a" in call and call[call.index("-c:a") + 1] == codec
    assert call[call.index("-f", 3) + 1] == fmt
    assert call[-2:] == ["-y", path]
    with open(path, "rb") as handle:
        assert handle.read() == f"FAKE:{fmt}:".encode() + PCM


def test_encode_rejects_unknown_format(fake_ffmpeg, tmp_path):
    with pytest.raises(FfmpegFailedError, match="unsupported"):
        audio.encode_pcm(PCM, "aiff", str(tmp_path / "x.aiff"))


def test_encode_failure_includes_stderr(fake_ffmpeg, monkeypatch, tmp_path):
    monkeypatch.setenv("FAKE_FFMPEG_FAIL", "1")
    with pytest.raises(FfmpegFailedError, match="forced failure"):
        audio.encode_pcm(PCM, "mp3", str(tmp_path / "x.mp3"))


def test_decode_to_pcm16_asks_for_16k_mono_s16le(fake_ffmpeg, tmp_path):
    source = tmp_path / "note.ogg"
    source.write_bytes(PCM)
    assert audio.decode_to_pcm16(str(source)) == PCM
    (call,) = _calls(fake_ffmpeg)
    assert call == ["-v", "error", "-nostdin", "-i", str(source), "-ac", "1", "-ar", "16000", "-f", "s16le", "-"]


def test_decode_trims_odd_byte_and_rejects_missing_or_empty(fake_ffmpeg, tmp_path):
    odd = tmp_path / "odd.bin"
    odd.write_bytes(b"abc")
    assert audio.decode_to_pcm16(str(odd)) == b"ab"
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(FfmpegFailedError, match="zero samples"):
        audio.decode_to_pcm16(str(empty))
    with pytest.raises(FfmpegFailedError, match="not found"):
        audio.decode_to_pcm16(str(tmp_path / "nope.ogg"))


def test_transcode_stream_passthrough_formats(no_ffmpeg):
    chunks = [PCM[:100], PCM[100:]]
    assert b"".join(audio.transcode_stream(iter(chunks), "pcm")) == PCM
    streamed = b"".join(audio.transcode_stream(iter(chunks), "wav"))
    header = audio.streaming_wav_header()
    assert len(header) == 44 and header[:4] == b"RIFF" and header[36:40] == b"data"
    assert struct.unpack("<I", header[40:44])[0] == 0xFFFFFFFF
    assert streamed == header + PCM


def test_transcode_stream_through_ffmpeg(fake_ffmpeg):
    out = b"".join(audio.transcode_stream(iter([PCM[:1000], PCM[1000:]]), "ogg"))
    assert out == b"FAKE:ogg:" + PCM
    (call,) = _calls(fake_ffmpeg)
    assert call[-1] == "pipe:1" and "libopus" in call


def test_transcode_stream_surfaces_ffmpeg_failure(fake_ffmpeg, monkeypatch):
    monkeypatch.setenv("FAKE_FFMPEG_FAIL", "1")
    with pytest.raises(FfmpegFailedError, match="forced failure"):
        list(audio.transcode_stream(iter([PCM]), "mp3"))


def test_transcode_stream_surfaces_source_errors(fake_ffmpeg):
    def broken():
        yield PCM[:100]
        raise RuntimeError("upstream died")

    with pytest.raises(RuntimeError, match="upstream died"):
        list(audio.transcode_stream(broken(), "mp3"))


MP3_MAGIC = (b"ID3", bytes([0xFF, 0xFB]), bytes([0xFF, 0xF3]))


@pytest.mark.skipif(not REAL_FFMPEG, reason="real ffmpeg not installed")
@pytest.mark.parametrize("fmt,magic", [("mp3", MP3_MAGIC), ("ogg", (b"OggS",)),
                                       ("opus", (b"OggS",)), ("flac", (b"fLaC",))])
def test_real_ffmpeg_encodes_every_hermes_format(monkeypatch, tmp_path, fmt, magic):
    monkeypatch.setenv("HERMES_NARI_FFMPEG", REAL_FFMPEG)
    path = audio.encode_pcm(PCM * 20, fmt, str(tmp_path / f"real.{fmt}"))
    with open(path, "rb") as handle:
        head = handle.read(4)
    assert any(head.startswith(m) for m in magic)
    assert os.path.getsize(path) > 100


@pytest.mark.skipif(not REAL_FFMPEG, reason="real ffmpeg not installed")
def test_real_ffmpeg_decodes_to_16k_and_streams_opus(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_NARI_FFMPEG", REAL_FFMPEG)
    source = tmp_path / "in.wav"
    audio.write_pcm_as_wav(PCM * 20, str(source))
    pcm16 = audio.decode_to_pcm16(str(source))
    assert abs(len(pcm16) - len(PCM) * 20 * 16_000 // 24_000) < 400
    streamed = b"".join(audio.transcode_stream(iter([PCM * 10, PCM * 10]), "ogg"))
    assert streamed.startswith(b"OggS")
    assert shutil.which("ffmpeg")
