import pytest

from hermes_nari.api import NariClient, parse_voices
from hermes_nari.errors import (NariAuthError, NariConnectionError, NariProtocolError, NariRateLimitError,
                                NariRequestError, NariServerError)
from hermes_nari.text import MAX_INPUT_CODE_POINTS
from tests.support import FAST_TIMEOUTS
from tests.fake_nari import pcm_for


def test_synthesize_sends_the_documented_request(client, fake_nari):
    pcm = client.synthesize_pcm("Hello there.", model="qwen3-tts:free", voice="leon")
    assert pcm == pcm_for("Hello there.")
    (request,) = fake_nari.speech_requests
    assert request["body"] == {"model": "qwen3-tts:free", "voice": "leon", "input": "Hello there.",
                               "response_format": "pcm", "stream": False}
    assert request["headers"]["Authorization"] == f"Bearer {fake_nari.api_key}"
    assert request["headers"]["Content-Type"] == "application/json"
    assert request["headers"]["User-Agent"].startswith("hermes-nari/")


def test_language_is_only_sent_when_given(client, fake_nari):
    client.synthesize_pcm("Hola.", model="qwen3-tts:free", voice="gabriel", language="es")
    assert fake_nari.speech_requests[-1]["body"]["language"] == "es"


def test_stream_yields_progressively_and_matches_the_full_audio(client, fake_nari):
    chunks = list(client.stream_pcm("Streamed sentence.", model="qwen3-tts:free", voice="leon"))
    assert len(chunks) > 1
    assert b"".join(chunks) == pcm_for("Streamed sentence.")
    assert fake_nari.speech_requests[-1]["body"]["stream"] is True


def test_long_text_is_split_and_concatenated(client, fake_nari):
    sentence = "The quick brown fox jumps over the lazy dog, again and again, without rest."
    text = " ".join(sentence for _ in range(40))
    assert len(text) > MAX_INPUT_CODE_POINTS
    pcm = client.synthesize_pcm(text, model="qwen3-tts:free", voice="leon")
    sent = [r["body"]["input"] for r in fake_nari.speech_requests]
    assert len(sent) > 1
    assert all(1 <= len(s) <= MAX_INPUT_CODE_POINTS for s in sent)
    assert " ".join(sent) == text
    assert pcm == b"".join(pcm_for(s) for s in sent)


def test_streaming_long_text_keeps_piece_order(client, fake_nari):
    text = " ".join(f"Piece number {i} of the story ends here." for i in range(120))
    pcm = b"".join(client.stream_pcm(text, model="qwen3-tts:free", voice="leon"))
    sent = [r["body"]["input"] for r in fake_nari.speech_requests]
    assert len(sent) > 1
    assert pcm == b"".join(pcm_for(s) for s in sent)


def test_empty_text_is_rejected_locally(client, fake_nari):
    with pytest.raises(ValueError):
        client.synthesize_pcm("   ", model="qwen3-tts:free", voice="leon")
    assert fake_nari.speech_requests == []


def test_list_voices_object_and_list_shapes(client, fake_nari):
    voices = client.list_voices("qwen3-tts:free")
    assert [v.id for v in voices] == ["leon", "diana", "gabriel"]
    assert voices[0].display == "Leon" and voices[0].language == "en" and voices[0].gender == "male"
    assert voices[0].preview_url == "https://example.invalid/leon.mp3"
    assert fake_nari.voice_requests[-1]["model"] == ["qwen3-tts:free"]
    fake_nari.voices_shape = "list"
    assert [v.id for v in client.list_voices("qwen3-tts:free")] == ["leon", "diana", "gabriel"]


def test_parse_voices_tolerates_unknown_wrappers_and_rejects_garbage():
    assert [v.id for v in parse_voices({"catalog_version": 3, "result": [{"id": "a"}, {"nope": 1}]})] == ["a"]
    assert parse_voices({"data": [{"id": "b", "display_name": "B"}]})[0].display == "B"
    with pytest.raises(NariProtocolError):
        parse_voices("not a list")


def test_401_becomes_auth_error_with_key_advice(fake_nari):
    bad = NariClient("wrong-key", base_url=fake_nari.base_url, timeouts=FAST_TIMEOUTS)
    with pytest.raises(NariAuthError) as excinfo:
        bad.synthesize_pcm("Hi.", model="qwen3-tts:free", voice="leon")
    message = str(excinfo.value)
    assert "401" in message and "INVALID_API_KEY" in message and "NARI_API_KEY" in message
    assert excinfo.value.status == 401 and excinfo.value.code == "INVALID_API_KEY"


def test_429_becomes_rate_limit_error_with_advice(client, fake_nari):
    fake_nari.next_http_error = (429, "FREE_DAILY_LIMIT_EXCEEDED", "Daily allowance exhausted")
    with pytest.raises(NariRateLimitError) as excinfo:
        client.synthesize_pcm("Hi.", model="qwen3-tts:free", voice="leon")
    assert "00:00 UTC" in str(excinfo.value) and excinfo.value.code == "FREE_DAILY_LIMIT_EXCEEDED"
    fake_nari.next_http_error = (429, "CONCURRENCY_LIMIT_EXCEEDED", "Too many in flight")
    with pytest.raises(NariRateLimitError, match="2 in-flight"):
        client.synthesize_pcm("Hi.", model="qwen3-tts:free", voice="leon")


def test_other_4xx_and_5xx_map_to_their_classes(client, fake_nari):
    fake_nari.next_http_error = (400, "INVALID_VOICE", "Unknown voice")
    with pytest.raises(NariRequestError, match="hermes-nari voices"):
        client.synthesize_pcm("Hi.", model="qwen3-tts:free", voice="nobody")
    fake_nari.next_http_error = (503, "UPSTREAM_UNAVAILABLE", "try later")
    with pytest.raises(NariServerError, match="503"):
        client.synthesize_pcm("Hi.", model="qwen3-tts:free", voice="leon")
    fake_nari.next_http_error = (402, "INSUFFICIENT_CREDITS", "no credits")
    with pytest.raises(NariRequestError, match=":free"):
        client.synthesize_pcm("Hi.", model="qwen3-tts:free", voice="leon")


def test_unreachable_host_is_a_connection_error():
    client = NariClient("key", base_url="http://127.0.0.1:9", timeouts=FAST_TIMEOUTS)
    with pytest.raises(NariConnectionError, match="could not reach"):
        client.list_voices("qwen3-tts:free")


def test_wav_request_format_is_never_used(client, fake_nari):
    client.synthesize_pcm("Hi.", model="qwen3-tts:free", voice="leon")
    assert all(r["body"]["response_format"] == "pcm" for r in fake_nari.speech_requests)
