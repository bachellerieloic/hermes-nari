import pytest

from hermes_nari import cli
from tests.fake_nari import pcm_for


def run(argv):
    lines = []
    args = cli.build_parser().parse_args(argv)
    code = args.func(args, lines.append)
    return code, "\n".join(lines)


def test_version_flag():
    with pytest.raises(SystemExit) as excinfo:
        cli.main(["--version"])
    assert excinfo.value.code == 0


def test_doctor_reports_missing_pieces(no_ffmpeg):
    code, out = run(["doctor"])
    assert code == cli.EXIT_NOT_CONFIGURED
    assert "NARI_API_KEY: MISSING" in out and "ffmpeg: MISSING" in out
    assert "tts: model=qwen3-tts:free voice=leon output_format=mp3" in out
    assert "stt: model=qwen3-asr:free language=en" in out


def test_doctor_warns_about_shadowing_command_providers(monkeypatch, write_config, fake_ffmpeg):
    monkeypatch.setenv("NARI_API_KEY", "abcdefghijkl")
    write_config("tts:\n  providers:\n    nari:\n      type: command\n      command: old.py {input_path}\n")
    code, out = run(["doctor"])
    assert code == cli.EXIT_OK
    assert "NARI_API_KEY: abcd...kl" in out
    assert "tts.providers.nari is a command provider" in out


def test_voices_without_key():
    code, out = run(["voices"])
    assert code == cli.EXIT_NOT_CONFIGURED and "NARI_API_KEY" in out


def test_voices_lists_catalog(fake_nari):
    code, out = run(["voices", "--model", "qwen3-tts-fast:free"])
    assert code == cli.EXIT_OK
    assert "3 voices for qwen3-tts-fast:free" in out
    assert "leon" in out and "Leon" in out and "Warm and steady" in out


def test_voices_reports_api_errors(fake_nari):
    fake_nari.api_key = "rotated"
    code, out = run(["voices"])
    assert code == cli.EXIT_FAILED and "401" in out


def test_selftest_without_key():
    code, out = run(["selftest"])
    assert code == cli.EXIT_NOT_CONFIGURED


def test_selftest_round_trip(fake_nari, fake_ffmpeg, tmp_path):
    keep = tmp_path / "kept"
    code, out = run(["selftest", "--text", "Round trip please.", "--keep", str(keep)])
    assert code == cli.EXIT_OK, out
    assert "tts:        qwen3-tts:free/leon" in out and "stt:        qwen3-asr:free" in out
    assert "round trip:" in out and "audio kept in" in out
    kept_file = keep / cli.SELFTEST_FILE
    assert kept_file.exists()
    assert fake_nari.speech_requests[-1]["body"]["input"] == "Round trip please."
    assert fake_nari.ws_sessions[-1]["session"]["language"] == "en"


def test_selftest_cleans_up_by_default(fake_nari, fake_ffmpeg, monkeypatch, tmp_path):
    monkeypatch.setattr(cli.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "work"))
    code, out = run(["selftest"])
    assert code == cli.EXIT_OK, out
    assert not (tmp_path / "work" / cli.SELFTEST_FILE).exists()


def test_selftest_reports_tts_failure(fake_nari, fake_ffmpeg):
    fake_nari.next_http_error = (429, "FREE_DAILY_LIMIT_EXCEEDED", "done for today")
    code, out = run(["selftest"])
    assert code == cli.EXIT_FAILED and "TTS failed" in out and "00:00 UTC" in out


def test_selftest_reports_stt_failure(fake_nari, fake_ffmpeg):
    fake_nari.ws_error_after_configure = ("MODEL_NOT_FOUND", "no")
    code, out = run(["selftest"])
    assert code == cli.EXIT_FAILED and "STT failed" in out


def test_main_prints_nari_errors_instead_of_tracebacks(fake_nari, capsys):
    fake_nari.api_key = "rotated"
    assert cli.main(["voices"]) == cli.EXIT_FAILED
    assert "401" in capsys.readouterr().out
    assert pcm_for("x")  # keep the import honest
