import importlib.util
import re
import sys
import types
from pathlib import Path
from unittest import mock

import yaml

import hermes_nari

ROOT = Path(__file__).resolve().parents[1]
# Hermes classifies a plugin's kind by scanning its __init__.py for these markers; none may appear.
HERMES_KIND_MARKERS = ("register_memory_provider", "MemoryProvider", "ProviderProfile")


def test_register_wires_both_providers():
    ctx = mock.Mock()
    hermes_nari.register(ctx)
    (tts,) = ctx.register_tts_provider.call_args.args
    (stt,) = ctx.register_transcription_provider.call_args.args
    assert tts.name == "nari" and stt.name == "nari"
    assert tts.voice_compatible is True


def test_register_needs_only_the_two_provider_hooks():
    ctx = mock.Mock(spec=["register_tts_provider", "register_transcription_provider"])
    hermes_nari.register(ctx)
    ctx.register_tts_provider.assert_called_once()
    ctx.register_transcription_provider.assert_called_once()


def _load_like_hermes(plugin_dir: Path, module_name: str) -> types.ModuleType:
    """Import a directory plugin the way hermes_cli.plugins does (v0.16.0 and later)."""
    namespace = "hermes_plugins"
    if namespace not in sys.modules:
        package = types.ModuleType(namespace)
        package.__path__ = []  # type: ignore[attr-defined]
        sys.modules[namespace] = package
    full_name = f"{namespace}.{module_name}"
    spec = importlib.util.spec_from_file_location(full_name, plugin_dir / "__init__.py",
                                                  submodule_search_locations=[str(plugin_dir)])
    module = importlib.util.module_from_spec(spec)
    module.__package__ = full_name
    module.__path__ = [str(plugin_dir)]  # type: ignore[attr-defined]
    sys.modules[full_name] = module
    spec.loader.exec_module(module)
    return module


def test_repository_root_loads_as_a_directory_plugin(monkeypatch):
    for name in [n for n in sys.modules if n.startswith("hermes_plugins")]:
        monkeypatch.delitem(sys.modules, name)
    module = _load_like_hermes(ROOT, "nari_root_checkout")
    ctx = mock.Mock()
    module.register(ctx)
    assert ctx.register_tts_provider.call_args.args[0].name == "nari"


def test_package_directory_loads_as_a_directory_plugin(monkeypatch):
    for name in [n for n in sys.modules if n.startswith("hermes_plugins")]:
        monkeypatch.delitem(sys.modules, name)
    module = _load_like_hermes(ROOT / "hermes_nari", "nari_package_copy")
    ctx = mock.Mock()
    module.register(ctx)
    assert ctx.register_transcription_provider.call_args.args[0].name == "nari"


def test_init_files_avoid_hermes_kind_detection_markers():
    for path in (ROOT / "__init__.py", ROOT / "hermes_nari" / "__init__.py"):
        text = path.read_text(encoding="utf-8")
        assert not any(marker in text for marker in HERMES_KIND_MARKERS), path


def test_manifests_are_identical_and_versions_agree():
    root_manifest = (ROOT / "plugin.yaml").read_text(encoding="utf-8")
    assert root_manifest == (ROOT / "hermes_nari" / "plugin.yaml").read_text(encoding="utf-8")
    manifest = yaml.safe_load(root_manifest)
    assert manifest["name"] == "nari"
    assert manifest["requires_env"] == ["NARI_API_KEY"]
    assert manifest["kind"] == "backend"
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    version = re.search(r'^version = "([^"]+)"', pyproject, re.M).group(1)
    assert str(manifest["version"]) == version == hermes_nari.__version__


def test_entry_point_and_console_script_are_declared():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '[project.entry-points."hermes_agent.plugins"]' in pyproject
    assert re.search(r'^nari = "hermes_nari"$', pyproject, re.M)
    assert re.search(r'^hermes-nari = "hermes_nari\.cli:main"$', pyproject, re.M)
