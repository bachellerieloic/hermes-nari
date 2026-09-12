import sys
import types
from pathlib import Path

import pytest

from hermes_nari import config
from hermes_nari.errors import NariConfigError


def test_hermes_home_prefers_env(tmp_path):
    assert config.hermes_home({"HERMES_HOME": str(tmp_path)}) == tmp_path
    assert config.hermes_home({}) == config.DEFAULT_HERMES_HOME


def test_read_env_file_handles_quotes_exports_and_comments(tmp_path):
    path = tmp_path / ".env"
    path.write_text('# comment\nexport NARI_API_KEY="abc"\nOTHER=\'x y\'\nBROKEN\nEMPTY=\n', encoding="utf-8")
    values = config.read_env_file(path)
    assert dict(values) == {"NARI_API_KEY": "abc", "OTHER": "x y", "EMPTY": ""}
    assert config.read_env_file(tmp_path / "missing") == {}


def test_api_key_resolution_order(isolated_hermes_home, write_env, monkeypatch):
    assert config.resolve_api_key() is None
    default_home = config.DEFAULT_HERMES_HOME
    default_home.mkdir()
    (default_home / ".env").write_text("NARI_API_KEY=from-default-home\n", encoding="utf-8")
    assert config.resolve_api_key() == "from-default-home"
    write_env("NARI_API_KEY=from-hermes-home\n")
    assert config.resolve_api_key() == "from-hermes-home"
    monkeypatch.setenv("NARI_API_KEY", "from-env")
    assert config.resolve_api_key() == "from-env"


def test_require_api_key_names_the_env_file(isolated_hermes_home):
    with pytest.raises(NariConfigError) as excinfo:
        config.require_api_key()
    assert "NARI_API_KEY" in str(excinfo.value)
    assert str(isolated_hermes_home / ".env") in str(excinfo.value)


def test_provider_block_prefers_canonical_location():
    cfg = {"tts": {"providers": {"nari": {"voice": "diana"}}, "nari": {"voice": "leon"}}}
    assert config.provider_block(cfg, "tts") == {"voice": "diana"}
    assert config.provider_block({"tts": {"nari": {"voice": "leon"}}}, "tts") == {"voice": "leon"}
    assert config.provider_block({"tts": "nonsense"}, "tts") == {}
    assert config.provider_block({}, "stt") == {}


def test_command_provider_shadow_detection():
    assert config.is_shadowed_by_command_provider({"tts": {"providers": {"nari": {"type": "command"}}}}, "tts")
    assert config.is_shadowed_by_command_provider({"stt": {"providers": {"nari": {"command": "x {input_path}"}}}}, "stt")
    assert not config.is_shadowed_by_command_provider({"tts": {"providers": {"nari": {"voice": "leon"}}}}, "tts")


def test_tts_defaults():
    settings = config.resolve_tts_settings(config={}, env={})
    assert settings == config.TTSSettings(model="qwen3-tts:free", voice="leon", output_format="mp3",
                                          base_url="https://api.narilabs.com")


def test_tts_precedence_block_over_kwargs_over_env():
    env = {"NARI_TTS_MODEL": "env-model", "NARI_TTS_VOICE": "env-voice", "NARI_TTS_FORMAT": "flac",
           "NARI_BASE_URL": "https://alt.example/"}
    from_env = config.resolve_tts_settings(config={}, env=env)
    assert (from_env.model, from_env.voice, from_env.output_format, from_env.base_url) == (
        "env-model", "env-voice", "flac", "https://alt.example")
    from_kwargs = config.resolve_tts_settings(voice="kw-voice", model="kw-model", output_format="ogg", config={}, env=env)
    assert (from_kwargs.model, from_kwargs.voice, from_kwargs.output_format) == ("kw-model", "kw-voice", "ogg")
    block = {"tts": {"providers": {"nari": {"model": "block-model", "voice": "block-voice", "output_format": "OPUS"}}}}
    from_block = config.resolve_tts_settings(voice="kw-voice", model="kw-model", output_format="ogg", config=block, env=env)
    assert (from_block.model, from_block.voice, from_block.output_format) == ("block-model", "block-voice", "opus")


def test_tts_rejects_unknown_output_format():
    with pytest.raises(NariConfigError, match="Unsupported output format"):
        config.resolve_tts_settings(output_format="aiff", config={}, env={})


def test_stt_defaults_and_language_rules():
    assert config.resolve_stt_settings(config={}, env={}) == config.STTSettings(
        model="qwen3-asr:free", language="en", ws_url=config.DEFAULT_WS_URL)
    assert config.resolve_stt_settings(language="", config={}, env={}).language is None
    assert config.resolve_stt_settings(language="auto", config={}, env={}).language is None
    assert config.resolve_stt_settings(language="ES", config={}, env={}).language == "es"
    top_level = {"stt": {"language": ""}}
    assert config.resolve_stt_settings(config=top_level, env={"NARI_STT_LANGUAGE": "ko"}).language is None
    assert config.resolve_stt_settings(config={"stt": {"language": "fr"}}, env={}).language == "fr"
    block = {"stt": {"language": "fr", "providers": {"nari": {"language": "ko", "model": "qwen3-asr-fast:free"}}}}
    settings = config.resolve_stt_settings(model="kw", language="es", config=block, env={})
    assert (settings.model, settings.language) == ("qwen3-asr-fast:free", "ko")
    assert config.resolve_stt_settings(model="kw", config={}, env={"NARI_STT_MODEL": "env"}).model == "kw"


def test_load_config_reads_the_yaml_file(write_config, monkeypatch):
    monkeypatch.setattr(config, "_load_config_via_hermes", lambda: None)  # force the file path even with Hermes present
    write_config("tts:\n  providers:\n    nari:\n      voice: diana\n")
    assert config.load_hermes_config()["tts"]["providers"]["nari"]["voice"] == "diana"
    write_config("- not a mapping\n")
    assert config.load_hermes_config() == {}
    write_config("tts: [unclosed\n")
    assert config.load_hermes_config() == {}


def test_load_config_uses_hermes_loader_when_present(monkeypatch, write_config):
    write_config("tts:\n  providers:\n    nari:\n      voice: file\n")
    package = types.ModuleType("hermes_cli")
    package.__path__ = []  # type: ignore[attr-defined]
    module = types.ModuleType("hermes_cli.config")
    module.load_config_readonly = lambda: {"tts": {"providers": {"nari": {"voice": "hermes"}}}}  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "hermes_cli", package)
    monkeypatch.setitem(sys.modules, "hermes_cli.config", module)
    assert config.resolve_tts_settings().voice == "hermes"


def test_missing_config_file_is_fine(isolated_hermes_home, monkeypatch):
    monkeypatch.setattr(config, "_load_config_via_hermes", lambda: None)
    assert not (isolated_hermes_home / "config.yaml").exists()
    assert config.load_hermes_config() == {}
    assert Path(isolated_hermes_home).is_dir()
