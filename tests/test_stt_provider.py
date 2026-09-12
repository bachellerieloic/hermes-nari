import wave

import pytest

from hermes_nari.stt import NariTranscriptionProvider
from tests.support import FAST_TIMEOUTS
from hermes_nari.api import NariClient

# Names reserved by Hermes (agent/transcription_registry.py).
BUILTIN_STT = {"local", "local_command", "groq", "openai", "mistral", "xai", "elevenlabs", "deepinfra"}
PCM = bytes(range(256)) * 20


@pytest.fixture
def note(tmp_path):
    path = tmp_path / "voice-note.wav"
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(16_000)
        handle.writeframes(PCM)
    return str(path)


def _fast_provider(**kwargs):
    return NariTranscriptionProvider(timeouts=FAST_TIMEOUTS, **kwargs)


def test_identity_and_contract_shape():
    from agent.transcription_provider import TranscriptionProvider

    provider = NariTranscriptionProvider(env={})
    assert isinstance(provider, TranscriptionProvider)
    assert provider.name == "nari" and provider.name not in BUILTIN_STT
    assert provider.display_name == "Nari Labs"
    assert [m["id"] for m in provider.list_models()] == ["qwen3-asr:free", "qwen3-asr-fast:free", "qwen3-asr", "qwen3-asr-fast"]
    assert all({"id", "display", "languages"} <= set(m) for m in provider.list_models())
    assert provider.default_model() == "qwen3-asr:free"
    assert provider.get_setup_schema()["env_vars"][0]["key"] == "NARI_API_KEY"


def test_availability(monkeypatch):
    assert NariTranscriptionProvider(env={}).is_available() is False
    assert NariTranscriptionProvider(env={"NARI_API_KEY": "k"}).is_available() is True


def test_success_envelope(fake_nari, fake_ffmpeg, note):
    result = _fast_provider().transcribe(note, model="qwen3-asr-fast:free", language="en", prompt="Hermes", unknown=1)
    with open(note, "rb") as handle:
        expected = f"heard {len(handle.read())} bytes"
    assert result == {"success": True, "transcript": expected, "provider": "nari"}
    session = fake_nari.ws_sessions[-1]["session"]
    assert session == {"model": "qwen3-asr-fast:free", "language": "en", "turn_detection": None, "prompt": "Hermes"}


def test_block_config_wins_and_auto_language_is_null(fake_nari, fake_ffmpeg, note, write_config):
    write_config("stt:\n  language: fr\n  providers:\n    nari:\n      model: qwen3-asr-fast:free\n      language: auto\n")
    _fast_provider().transcribe(note, model="qwen3-asr:free", language="en")
    session = fake_nari.ws_sessions[-1]["session"]
    assert session["model"] == "qwen3-asr-fast:free" and session["language"] is None


def test_error_envelope_never_raises(fake_nari, fake_ffmpeg, note):
    fake_nari.api_key = "rotated"
    result = _fast_provider().transcribe(note)
    assert result["success"] is False and result["transcript"] == "" and result["provider"] == "nari"
    assert "401" in result["error"] and "NARI_API_KEY" in result["error"]


def test_missing_key_envelope(fake_ffmpeg, note):
    result = _fast_provider(env={}).transcribe(note)
    assert result["success"] is False and "NARI_API_KEY" in result["error"]


def test_missing_ffmpeg_envelope(fake_nari, no_ffmpeg, note):
    result = _fast_provider().transcribe(note)
    assert result["success"] is False and "ffmpeg" in result["error"]


def test_missing_file_envelope(fake_nari, fake_ffmpeg, tmp_path):
    result = _fast_provider().transcribe(str(tmp_path / "gone.ogg"))
    assert result["success"] is False and "not found" in result["error"]


def test_empty_transcript_is_an_error_envelope(fake_nari, fake_ffmpeg, note):
    fake_nari.transcript_for = lambda pcm: "   "
    result = _fast_provider().transcribe(note)
    assert result["success"] is False and "no transcript" in result["error"]


def test_unexpected_exception_is_wrapped(fake_nari, fake_ffmpeg, note):
    def exploding_factory(_settings):
        raise RuntimeError("kaboom")

    result = _fast_provider(client_factory=exploding_factory).transcribe(note)
    assert result == {"success": False, "transcript": "", "error": "Nari transcription failed: kaboom", "provider": "nari"}


def test_custom_client_factory_receives_settings(fake_nari, fake_ffmpeg, note):
    seen = []

    def factory(settings):
        seen.append(settings)
        return NariClient(fake_nari.api_key, ws_url=fake_nari.ws_url, timeouts=FAST_TIMEOUTS)

    assert _fast_provider(client_factory=factory).transcribe(note, language="es")["success"] is True
    assert seen[0].language == "es"
