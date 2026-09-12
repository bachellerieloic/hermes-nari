"""Test bootstrap: Hermes ABC stubs when Hermes is absent, fake Nari servers, fake ffmpeg."""

from __future__ import annotations

import abc
import os
import stat
import sys
import types
from pathlib import Path
from typing import Iterator

import pytest

FAKE_FFMPEG_SOURCE = Path(__file__).with_name("fake_ffmpeg.py")


def _install_stub_abcs() -> None:
    """Mirror agent.tts_provider / agent.transcription_provider when hermes-agent is absent."""
    try:
        import agent.transcription_provider  # noqa: F401
        import agent.tts_provider  # noqa: F401
        return
    except ImportError:
        pass

    class ProviderBase(abc.ABC):
        @property
        @abc.abstractmethod
        def name(self) -> str: ...

        @property
        def display_name(self) -> str:
            return self.name.title()

        def is_available(self) -> bool:
            return True

        def list_models(self):
            return []

        def default_model(self):
            models = self.list_models()
            return models[0].get("id") if models else None

        def get_setup_schema(self):
            return {"name": self.display_name, "badge": "", "tag": "", "env_vars": []}

    class TTSProvider(ProviderBase):
        def list_voices(self):
            return []

        def default_voice(self):
            voices = self.list_voices()
            return voices[0].get("id") if voices else None

        @abc.abstractmethod
        def synthesize(self, text, output_path, *, voice=None, model=None, speed=None, format="mp3", **extra) -> str: ...

        def stream(self, text, *, voice=None, model=None, format="opus", **extra):
            raise NotImplementedError

        @property
        def voice_compatible(self) -> bool:
            return False

    class TranscriptionProvider(ProviderBase):
        @abc.abstractmethod
        def transcribe(self, file_path, *, model=None, language=None, **extra): ...

    agent_pkg = types.ModuleType("agent")
    agent_pkg.__path__ = []  # type: ignore[attr-defined]
    tts_mod = types.ModuleType("agent.tts_provider")
    tts_mod.TTSProvider = TTSProvider  # type: ignore[attr-defined]
    stt_mod = types.ModuleType("agent.transcription_provider")
    stt_mod.TranscriptionProvider = TranscriptionProvider  # type: ignore[attr-defined]
    sys.modules["agent"] = agent_pkg
    sys.modules["agent.tts_provider"] = tts_mod
    sys.modules["agent.transcription_provider"] = stt_mod


_install_stub_abcs()

from hermes_nari import config as config_module  # noqa: E402
from hermes_nari.api import NariClient  # noqa: E402
from tests.fake_nari import FakeNariHTTP, FakeNariWS, FakeState  # noqa: E402
from tests.support import FAST_TIMEOUTS  # noqa: E402


@pytest.fixture(autouse=True)
def isolated_hermes_home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Keep every test away from the real ~/.hermes and from ambient NARI_* variables."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setattr(config_module, "DEFAULT_HERMES_HOME", tmp_path / "default-hermes-home")
    for name in list(os.environ):
        if name.startswith("NARI_") or name == "HERMES_NARI_FFMPEG":
            monkeypatch.delenv(name, raising=False)
    return home


@pytest.fixture
def write_config(isolated_hermes_home: Path):
    def _write(text: str) -> Path:
        path = isolated_hermes_home / "config.yaml"
        path.write_text(text, encoding="utf-8")
        return path
    return _write


@pytest.fixture
def write_env(isolated_hermes_home: Path):
    def _write(text: str) -> Path:
        path = isolated_hermes_home / ".env"
        path.write_text(text, encoding="utf-8")
        return path
    return _write


@pytest.fixture
def state() -> FakeState:
    return FakeState()


@pytest.fixture
def fake_http(state: FakeState) -> Iterator[FakeNariHTTP]:
    server = FakeNariHTTP(state).start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def fake_ws(state: FakeState) -> Iterator[FakeNariWS]:
    server = FakeNariWS(state).start()
    try:
        yield server
    finally:
        server.stop()


@pytest.fixture
def fake_nari(state: FakeState, fake_http: FakeNariHTTP, fake_ws: FakeNariWS, monkeypatch: pytest.MonkeyPatch) -> FakeState:
    """Both fake servers, wired into the environment the providers read."""
    monkeypatch.setenv("NARI_API_KEY", state.api_key)
    monkeypatch.setenv("NARI_BASE_URL", fake_http.base_url)
    monkeypatch.setenv("NARI_WS_URL", fake_ws.ws_url)
    state.base_url = fake_http.base_url  # type: ignore[attr-defined]
    state.ws_url = fake_ws.ws_url  # type: ignore[attr-defined]
    return state


@pytest.fixture
def client(fake_nari: FakeState) -> NariClient:
    return NariClient(fake_nari.api_key, base_url=fake_nari.base_url, ws_url=fake_nari.ws_url,  # type: ignore[attr-defined]
                      timeouts=FAST_TIMEOUTS)


@pytest.fixture
def fake_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A fake ``ffmpeg`` executable first on PATH; returns the JSON-lines log of its invocations."""
    bin_dir = tmp_path / "fake-bin"
    bin_dir.mkdir()
    script = bin_dir / "ffmpeg"
    script.write_text(f"#!{sys.executable}\n" + FAKE_FFMPEG_SOURCE.read_text(encoding="utf-8"), encoding="utf-8")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    log = tmp_path / "ffmpeg-calls.jsonl"
    monkeypatch.setenv("FAKE_FFMPEG_LOG", str(log))
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ.get('PATH', '')}")
    return log


@pytest.fixture
def no_ffmpeg(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    empty = tmp_path / "empty-bin"
    empty.mkdir()
    monkeypatch.setenv("PATH", str(empty))
