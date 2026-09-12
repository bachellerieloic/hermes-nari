"""Guards on the repository itself: no dashes of the typographic kind, tidy manifests."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".pytest_cache", "__pycache__", "dist", "build", ".venv", "venv"}
TEXT_SUFFIXES = {".py", ".md", ".toml", ".yaml", ".yml", ".txt", ".cfg", ".ini", ""}
FORBIDDEN = {"\u2014": "em dash", "\u2013": "en dash"}


def _text_files():
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.is_file() and path.suffix in TEXT_SUFFIXES:
            yield path


def test_no_typographic_dashes_anywhere():
    offenders = []
    for path in _text_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        for char, label in FORBIDDEN.items():
            if char in text:
                line = next(i for i, l in enumerate(text.splitlines(), 1) if char in l)
                offenders.append(f"{path.relative_to(ROOT)}:{line} ({label})")
    assert not offenders, offenders


def test_readme_documents_the_essentials():
    raw = (ROOT / "README.md").read_text(encoding="utf-8")
    readme = "\n".join(line.strip() for line in raw.splitlines())
    for needle in ("provider: nari", "type: command", "voice:\nauto_tts: true", "plugins:\nenabled:",
                   "hermes plugins install bachellerieloic/hermes-nari", "git+https://github.com/bachellerieloic/hermes-nari",
                   "NARI_API_KEY", "founders@narilabs.com", "v0.16.0"):
        assert needle in readme, needle
