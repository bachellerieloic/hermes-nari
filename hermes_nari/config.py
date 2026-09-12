"""Settings for the Nari providers.

Sources, from most to least specific:

1. the plugin's own block in Hermes ``config.yaml`` (``tts.providers.nari`` or ``tts.nari``,
   ``stt.providers.nari`` or ``stt.nari``),
2. the keyword arguments Hermes passes to ``synthesize`` / ``transcribe`` (Hermes fills them
   from the top-level ``tts`` / ``stt`` keys),
3. ``NARI_*`` environment variables,
4. the defaults below.

The API key comes from ``NARI_API_KEY`` in the environment, then ``$HERMES_HOME/.env``,
then ``~/.hermes/.env``.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from .errors import NariConfigError

logger = logging.getLogger(__name__)

PROVIDER_NAME = "nari"
DISPLAY_NAME = "Nari Labs"

DEFAULT_BASE_URL = "https://api.narilabs.com"
DEFAULT_WS_URL = "wss://api.narilabs.com/v1/realtime?intent=transcription"
DEFAULT_TTS_MODEL = "qwen3-tts:free"
DEFAULT_TTS_VOICE = "leon"
DEFAULT_STT_MODEL = "qwen3-asr:free"
DEFAULT_STT_LANGUAGE = "en"
DEFAULT_OUTPUT_FORMAT = "mp3"

API_KEY_ENV = "NARI_API_KEY"
ENV_BASE_URL = "NARI_BASE_URL"
ENV_WS_URL = "NARI_WS_URL"
ENV_TTS_MODEL = "NARI_TTS_MODEL"
ENV_TTS_VOICE = "NARI_TTS_VOICE"
ENV_TTS_FORMAT = "NARI_TTS_FORMAT"
ENV_STT_MODEL = "NARI_STT_MODEL"
ENV_STT_LANGUAGE = "NARI_STT_LANGUAGE"
HERMES_HOME_ENV = "HERMES_HOME"

DEFAULT_HERMES_HOME = Path.home() / ".hermes"
ENV_FILE_NAME = ".env"
CONFIG_FILE_NAME = "config.yaml"
KEYS_URL = "https://app.narilabs.com/keys"

# Hermes accepts exactly these output formats (agent/tts_provider.py, VALID_OUTPUT_FORMATS).
VALID_OUTPUT_FORMATS = frozenset({"mp3", "wav", "ogg", "opus", "flac"})
AUTO_LANGUAGE_VALUES = frozenset({"", "auto"})


@dataclass(frozen=True)
class ModelInfo:
    id: str
    display: str
    languages: Tuple[str, ...]
    note: str

    def as_dict(self) -> dict:
        return {"id": self.id, "display": self.display, "languages": list(self.languages), "note": self.note}


TTS_MODELS: Tuple[ModelInfo, ...] = (
    ModelInfo("qwen3-tts:free", "Qwen3 TTS (free)", ("en", "es"), "free, best-effort"),
    ModelInfo("qwen3-tts-fast:free", "Qwen3 TTS Fast (free)", ("en", "es"), "free, best-effort"),
    ModelInfo("qwen3-tts", "Qwen3 TTS", ("en", "es"), "partner access, USD 5 per million characters"),
    ModelInfo("qwen3-tts-fast", "Qwen3 TTS Fast", ("en", "es"), "partner access, USD 10 per million characters"),
)

STT_MODELS: Tuple[ModelInfo, ...] = (
    ModelInfo("qwen3-asr:free", "Qwen3 ASR (free)", ("en", "es", "ko"), "free, best-effort"),
    ModelInfo("qwen3-asr-fast:free", "Qwen3 ASR Fast (free)", ("en", "es", "ko"), "free, best-effort"),
    ModelInfo("qwen3-asr", "Qwen3 ASR", ("en", "es", "ko"), "partner access, USD 0.06 per audio hour"),
    ModelInfo("qwen3-asr-fast", "Qwen3 ASR Fast", ("en", "es", "ko"), "partner access, USD 0.12 per audio hour"),
)


@dataclass(frozen=True)
class TTSSettings:
    model: str
    voice: str
    output_format: str
    base_url: str


@dataclass(frozen=True)
class STTSettings:
    model: str
    language: Optional[str]
    ws_url: str


# ---- Hermes home, .env and config.yaml -----------------------------------------------------


def hermes_home(env: Mapping[str, str] = os.environ) -> Path:
    """``$HERMES_HOME`` when set, else ``~/.hermes``."""
    raw = env.get(HERMES_HOME_ENV, "").strip()
    return Path(raw).expanduser() if raw else DEFAULT_HERMES_HOME


def read_env_file(path: Path) -> Mapping[str, str]:
    """Parse ``KEY=VALUE`` lines (comments, blank lines and ``export`` prefixes tolerated)."""
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return MappingProxyType({})
    values = {}
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return MappingProxyType(values)


def env_file_candidates(env: Mapping[str, str] = os.environ) -> Tuple[Path, ...]:
    home = hermes_home(env)
    candidates = (home / ENV_FILE_NAME, DEFAULT_HERMES_HOME / ENV_FILE_NAME)
    return tuple(dict.fromkeys(candidates))


def resolve_api_key(env: Mapping[str, str] = os.environ) -> Optional[str]:
    """``NARI_API_KEY`` from the environment, else from the Hermes ``.env`` files."""
    direct = env.get(API_KEY_ENV, "").strip()
    if direct:
        return direct
    for candidate in env_file_candidates(env):
        value = read_env_file(candidate).get(API_KEY_ENV, "").strip()
        if value:
            return value
    return None


def require_api_key(env: Mapping[str, str] = os.environ) -> str:
    key = resolve_api_key(env)
    if not key:
        locations = " or ".join(str(p) for p in env_file_candidates(env))
        raise NariConfigError(
            f"{API_KEY_ENV} is not set. Create a key at {KEYS_URL} and add "
            f"{API_KEY_ENV}=... to {locations}, then restart Hermes."
        )
    return key


def _load_config_via_hermes() -> Optional[Mapping[str, Any]]:
    """Use Hermes's own loader when running inside Hermes (it knows about profiles)."""
    try:
        from hermes_cli import config as hermes_config  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - any import trouble means "not inside Hermes"
        return None
    for attr in ("load_config_readonly", "load_config"):
        loader = getattr(hermes_config, attr, None)
        if loader is None:
            continue
        try:
            loaded = loader()
        except Exception as exc:  # noqa: BLE001
            logger.debug("hermes_cli.config.%s failed: %s", attr, exc)
            continue
        if isinstance(loaded, dict):
            return loaded
    return None


def _load_config_file(env: Mapping[str, str]) -> Mapping[str, Any]:
    path = hermes_home(env) / CONFIG_FILE_NAME
    try:
        import yaml  # type: ignore[import-untyped]
    except ImportError:
        logger.warning("PyYAML is not installed; %s is ignored", path)
        return {}
    try:
        with path.open(encoding="utf-8-sig") as handle:
            data = yaml.safe_load(handle)
    except FileNotFoundError:
        return {}
    except (OSError, yaml.YAMLError) as exc:
        logger.warning("Could not read %s: %s", path, exc)
        return {}
    return data if isinstance(data, dict) else {}


def load_hermes_config(env: Mapping[str, str] = os.environ) -> Mapping[str, Any]:
    """The parsed ``config.yaml`` (via Hermes when available, else read directly)."""
    via_hermes = _load_config_via_hermes()
    return via_hermes if via_hermes is not None else _load_config_file(env)


def provider_block(config: Mapping[str, Any], section: str) -> Mapping[str, Any]:
    """``<section>.providers.nari`` (canonical) or ``<section>.nari`` (built-in style)."""
    top = config.get(section)
    if not isinstance(top, dict):
        return {}
    providers = top.get("providers")
    canonical = providers.get(PROVIDER_NAME) if isinstance(providers, dict) else None
    if isinstance(canonical, dict):
        return canonical
    legacy = top.get(PROVIDER_NAME)
    return legacy if isinstance(legacy, dict) else {}


def is_shadowed_by_command_provider(config: Mapping[str, Any], section: str) -> bool:
    """True when ``<section>.providers.nari`` is a ``type: command`` block (it hides the plugin)."""
    block = provider_block(config, section)
    return str(block.get("type", "")).strip().lower() == "command" or bool(block.get("command"))


# ---- Resolution -----------------------------------------------------------------------------


def _first_text(*candidates: Any) -> Optional[str]:
    for value in candidates:
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def resolve_tts_settings(
    *, voice: Optional[str] = None, model: Optional[str] = None, output_format: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None, env: Mapping[str, str] = os.environ,
) -> TTSSettings:
    block = provider_block(config if config is not None else load_hermes_config(env), "tts")
    fmt = (_first_text(block.get("output_format"), block.get("format"), output_format,
                       env.get(ENV_TTS_FORMAT), DEFAULT_OUTPUT_FORMAT) or DEFAULT_OUTPUT_FORMAT).lower()
    if fmt not in VALID_OUTPUT_FORMATS:
        raise NariConfigError(
            f"Unsupported output format {fmt!r}; use one of {', '.join(sorted(VALID_OUTPUT_FORMATS))}."
        )
    return TTSSettings(
        model=_first_text(block.get("model"), model, env.get(ENV_TTS_MODEL)) or DEFAULT_TTS_MODEL,
        voice=_first_text(block.get("voice"), voice, env.get(ENV_TTS_VOICE)) or DEFAULT_TTS_VOICE,
        output_format=fmt,
        base_url=(_first_text(block.get("base_url"), env.get(ENV_BASE_URL)) or DEFAULT_BASE_URL).rstrip("/"),
    )


def _language_from(candidates: Tuple[Any, ...], default: Optional[str]) -> Optional[str]:
    """First present candidate decides: ``""``/``auto`` mean auto-detect (None)."""
    for value in candidates:
        if value is None:
            continue
        text = str(value).strip().lower()
        return None if text in AUTO_LANGUAGE_VALUES else text
    return default


def resolve_stt_settings(
    *, model: Optional[str] = None, language: Optional[str] = None,
    config: Optional[Mapping[str, Any]] = None, env: Mapping[str, str] = os.environ,
) -> STTSettings:
    loaded = config if config is not None else load_hermes_config(env)
    block = provider_block(loaded, "stt")
    top = loaded.get("stt") if isinstance(loaded.get("stt"), dict) else {}
    return STTSettings(
        model=_first_text(block.get("model"), model, env.get(ENV_STT_MODEL)) or DEFAULT_STT_MODEL,
        language=_language_from(
            (block.get("language"), language, top.get("language"), env.get(ENV_STT_LANGUAGE)),
            DEFAULT_STT_LANGUAGE,
        ),
        ws_url=_first_text(block.get("ws_url"), env.get(ENV_WS_URL)) or DEFAULT_WS_URL,
    )
