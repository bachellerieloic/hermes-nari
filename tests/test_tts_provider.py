import importlib
import json
import sys
import wave

import pytest

import hermes_nari._base as base_module
import hermes_nari.tts as tts_module
from hermes_nari.errors import NariConfigError
from hermes_nari.tts import NariTTSProvider, with_suffix_for
from tests.fake_nari import pcm_for

# Names reserved by Hermes (agent/tts_registry.py); a plugin may not use them.
BUILTIN_TTS = {"edge", "elevenlabs", "openai", "minimax", "xai", "mistral", "gemini", "neutts", "kittentts", "piper",
               "deepinfra", "nous"}


def test_identity_and_contract_shape():
    from agent.tts_provider import TTSProvider

    provider = NariTTSProvider(env={})
    assert isinstance(provider, TTSProvider)
    assert provider.name == "nari" and provider.name not in BUILTIN_TTS and provider.name.islower()
    assert provider.display_name == "Nari Labs"
    assert provider.voice_compatible is True
    schema = provider.get_setup_schema()
    assert schema["env_vars"][0]["key"] == "NARI_API_KEY" and schema["badge"] == "free"
    assert [m["id"] for m in provider.list_models()] == ["qwen3-tts:free", "qwen3-tts-fast:free", "qwen3-tts", "qwen3-tts-fast"]


def test_availability_never_raises_and_tracks_the_key(monkeypatch):
    assert NariTTSProvider(env={}).is_available() is False
    assert NariTTSProvider(env={"NARI_API_KEY": "k"}).is_available() is True
    assert NariTTSProvider(env=None).is_available() is False  # falls back to os.environ, which is clean here
    monkeypatch.setenv("NARI_API_KEY", "k")
    assert NariTTSProvider().is_available() is True

    class Explosive(dict):
        def get(self, *_a, **_k):
            raise RuntimeError("boom")

    assert NariTTSProvider(env=Explosive()).is_available() is False


def test_defaults_come_from_the_config_block(write_config):
    write_config("tts:\n  providers:\n    nari:\n      model: qwen3-tts-fast:free\n      voice: diana\n")
    provider = NariTTSProvider()
    assert provider.default_model() == "qwen3-tts-fast:free" and provider.default_voice() == "diana"
    assert NariTTSProvider(config={}).default_voice() == "leon"


def test_list_voices_matches_the_hermes_shape_and_is_cached(fake_nari):
    provider = NariTTSProvider()
    voices = provider.list_voices()
    assert voices[0] == {"id": "leon", "display": "Leon", "language": "en", "gender": "male",
                         "preview_url": "https://example.invalid/leon.mp3"}
    assert set(voices[1]) == {"id", "display", "language", "gender", "preview_url"}
    provider.list_voices()
    assert len(fake_nari.voice_requests) == 1
    assert fake_nari.voice_requests[0]["model"] == ["qwen3-tts:free"]


def test_list_voices_swallows_api_errors(fake_nari):
    fake_nari.api_key = "rotated"
    assert NariTTSProvider().list_voices() == []


def test_synthesize_wav_writes_without_ffmpeg(fake_nari, no_ffmpeg, tmp_path):
    provider = NariTTSProvider()
    target = str(tmp_path / "nested" / "reply.wav")
    written = provider.synthesize("Hello from Hermes.", target, format="wav")
    assert written == target
    with wave.open(target, "rb") as handle:
        assert handle.readframes(handle.getnframes()) == pcm_for("Hello from Hermes.")
    body = fake_nari.speech_requests[-1]["body"]
    assert body["voice"] == "leon" and body["model"] == "qwen3-tts:free"


def test_synthesize_rewrites_extension_and_converts(fake_nari, fake_ffmpeg, tmp_path):
    provider = NariTTSProvider()
    written = provider.synthesize("Bubble.", str(tmp_path / "tts_1.mp3"), voice="diana", format="ogg", speed=1.5, unknown="x")
    assert written == str(tmp_path / "tts_1.ogg")
    with open(written, "rb") as handle:
        assert handle.read() == b"FAKE:ogg:" + pcm_for("Bubble.")
    assert fake_nari.speech_requests[-1]["body"]["voice"] == "diana"
    call = json.loads(fake_ffmpeg.read_text().splitlines()[0])
    assert "libopus" in call and call[-1] == written


def test_block_settings_win_over_dispatcher_kwargs(fake_nari, fake_ffmpeg, write_config, tmp_path):
    write_config("tts:\n  voice: en-US-AriaNeural\n  providers:\n    nari:\n      voice: gabriel\n      output_format: flac\n")
    written = NariTTSProvider().synthesize("Hola.", str(tmp_path / "x.mp3"), voice="en-US-AriaNeural", format="mp3")
    assert written.endswith(".flac")
    assert fake_nari.speech_requests[-1]["body"]["voice"] == "gabriel"


def test_synthesize_raises_without_key(tmp_path):
    with pytest.raises(NariConfigError, match="NARI_API_KEY"):
        NariTTSProvider(env={}).synthesize("Hi.", str(tmp_path / "x.mp3"))


def test_synthesize_propagates_api_errors(fake_nari, tmp_path):
    fake_nari.next_http_error = (429, "CONCURRENCY_LIMIT_EXCEEDED", "busy")
    with pytest.raises(Exception, match="429"):
        NariTTSProvider().synthesize("Hi.", str(tmp_path / "x.wav"), format="wav")


def test_stream_yields_requested_format(fake_nari, fake_ffmpeg):
    provider = NariTTSProvider()
    assert b"".join(provider.stream("Stream me.", format="ogg")) == b"FAKE:ogg:" + pcm_for("Stream me.")
    assert b"".join(provider.stream("Raw.", format="wav")).endswith(pcm_for("Raw."))
    assert fake_nari.speech_requests[-1]["body"]["stream"] is True


def test_with_suffix_for():
    assert with_suffix_for("/tmp/a.mp3", "ogg") == "/tmp/a.ogg"
    assert with_suffix_for("/tmp/a.OGG", "ogg") == "/tmp/a.OGG"
    assert with_suffix_for("/tmp/noext", "wav") == "/tmp/noext.wav"


def test_duck_typed_fallback_when_hermes_is_absent(monkeypatch, fake_nari, no_ffmpeg, tmp_path):
    monkeypatch.setitem(sys.modules, "agent.tts_provider", None)
    monkeypatch.setitem(sys.modules, "agent.transcription_provider", None)
    try:
        importlib.reload(base_module)
        assert base_module.HERMES_ABCS_AVAILABLE is False and "agent" in base_module.IMPORT_PROBLEM
        fallback_tts = importlib.reload(tts_module)
        provider = fallback_tts.NariTTSProvider()
        target = str(tmp_path / "fallback.wav")
        assert provider.synthesize("Still works.", target, format="wav") == target
        assert provider.voice_compatible is True
    finally:
        monkeypatch.undo()
        importlib.reload(base_module)
        importlib.reload(tts_module)
    assert base_module.HERMES_ABCS_AVAILABLE is True
