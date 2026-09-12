import asyncio
import base64
import json

import pytest

from hermes_nari.api import (AUDIO_FRAME_PCM_BYTES, END_OF_INPUT_EVENT_ID, WS_MESSAGE_LIMIT_BYTES, NariClient,
                             audio_frames, run_sync)
from hermes_nari.errors import NariAuthError, NariConnectionError, NariRateLimitError, NariRequestError
from tests.support import FAST_TIMEOUTS

PCM = bytes(range(256)) * 700  # 179,200 bytes: several frames


def test_audio_frames_stay_under_the_message_cap_and_hold_whole_samples():
    frames = list(audio_frames(PCM))
    assert len(frames) == -(-len(PCM) // AUDIO_FRAME_PCM_BYTES)
    decoded = b""
    for frame in frames:
        assert len(frame.encode("utf-8")) < WS_MESSAGE_LIMIT_BYTES
        message = json.loads(frame)
        assert message["type"] == "input_audio_buffer.append"
        chunk = base64.b64decode(message["audio"])
        assert 0 < len(chunk) <= AUDIO_FRAME_PCM_BYTES and len(chunk) % 2 == 0
        decoded += chunk
    assert decoded == PCM


def test_audio_frames_drop_a_trailing_odd_byte_and_validate_frame_size():
    assert b"".join(base64.b64decode(json.loads(f)["audio"]) for f in audio_frames(b"abc")) == b"ab"
    with pytest.raises(ValueError):
        list(audio_frames(PCM, frame_bytes=3))


def test_transcribe_round_trip_follows_the_protocol(client, fake_nari):
    result = client.transcribe_pcm16(PCM, model="qwen3-asr:free", language="en")
    assert result.transcript == f"heard {len(PCM)} bytes"
    assert result.language == "en" and result.utterances == (result.transcript,)
    assert result.audio_seconds == pytest.approx(len(PCM) / 32_000)
    (session,) = fake_nari.ws_sessions
    assert session["first_message"] == {"type": "session.configure",
                                        "session": {"model": "qwen3-asr:free", "language": "en", "turn_detection": None}}
    assert session["headers"]["authorization"] == f"Bearer {fake_nari.api_key}"
    assert sum(session["frames"]) == len(PCM)
    assert all(size <= AUDIO_FRAME_PCM_BYTES for size in session["frames"])
    assert all(size < WS_MESSAGE_LIMIT_BYTES for size in session["raw_sizes"])
    assert session["commits"] == [END_OF_INPUT_EVENT_ID]


def test_language_none_and_prompt_are_encoded_as_documented(client, fake_nari):
    client.transcribe_pcm16(PCM[:4000], model="qwen3-asr-fast:free", language=None, prompt="Hermes, Nari")
    session = fake_nari.ws_sessions[-1]["session"]
    assert session["language"] is None and session["prompt"] == "Hermes, Nari"


def test_partials_are_reported_but_not_returned(client, fake_nari):
    seen = []
    result = client.transcribe_pcm16(PCM[:4000], model="qwen3-asr:free", on_partial=seen.append)
    assert len(seen) == fake_nari.partials_per_utterance
    assert all(result.transcript.startswith(p) for p in seen)
    assert result.partials == tuple(seen)
    assert result.transcript == "heard 4000 bytes"


def test_multiple_utterances_are_joined_in_order(client, fake_nari):
    fake_nari.utterance_bytes = 64_000
    fake_nari.transcript_for = lambda pcm: f"part{len(pcm)}"
    result = client.transcribe_pcm16(PCM, model="qwen3-asr:free")
    assert result.utterances == ("part64000", "part64000", "part51200")
    assert result.transcript == "part64000 part64000 part51200"
    assert result.audio_seconds == pytest.approx(len(PCM) / 32_000)


def test_empty_audio_is_rejected_locally(client, fake_nari):
    with pytest.raises(ValueError):
        client.transcribe_pcm16(b"", model="qwen3-asr:free")
    assert fake_nari.ws_sessions == []


def test_handshake_401_maps_to_auth_error(fake_nari):
    bad = NariClient("wrong", ws_url=fake_nari.ws_url, timeouts=FAST_TIMEOUTS)
    with pytest.raises(NariAuthError, match="NARI_API_KEY"):
        bad.transcribe_pcm16(PCM[:4000], model="qwen3-asr:free")


def test_handshake_429_maps_to_rate_limit(client, fake_nari):
    fake_nari.ws_reject = (429, "CONCURRENCY_LIMIT_EXCEEDED")
    with pytest.raises(NariRateLimitError, match="429"):
        client.transcribe_pcm16(PCM[:4000], model="qwen3-asr:free")


def test_error_event_after_configure_is_mapped(client, fake_nari):
    fake_nari.ws_error_after_configure = ("FREE_DAILY_LIMIT_EXCEEDED", "used up")
    with pytest.raises(NariRateLimitError, match="00:00 UTC"):
        client.transcribe_pcm16(PCM[:4000], model="qwen3-asr:free")
    fake_nari.ws_error_after_configure = ("MODEL_NOT_FOUND", "nope")
    with pytest.raises(NariRequestError, match="MODEL_NOT_FOUND"):
        client.transcribe_pcm16(PCM[:4000], model="qwen3-asr:free")


def test_early_close_reports_last_partial(client, fake_nari):
    fake_nari.close_early = True
    with pytest.raises(NariConnectionError) as excinfo:
        client.transcribe_pcm16(PCM[:4000], model="qwen3-asr:free")
    assert "closed" in str(excinfo.value) and "last partial" in str(excinfo.value)


def test_unreachable_websocket_is_a_connection_error():
    client = NariClient("key", ws_url="ws://127.0.0.1:9/", timeouts=FAST_TIMEOUTS)
    with pytest.raises(NariConnectionError):
        client.transcribe_pcm16(PCM[:4000], model="qwen3-asr:free")


def test_run_sync_works_inside_a_running_event_loop(client, fake_nari):
    async def inside_loop():
        return client.transcribe_pcm16(PCM[:4000], model="qwen3-asr:free")

    assert asyncio.run(inside_loop()).transcript == "heard 4000 bytes"
    assert run_sync(lambda: asyncio.sleep(0, result=7)) == 7
