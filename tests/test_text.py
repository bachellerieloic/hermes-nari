import pytest

from hermes_nari.text import MAX_INPUT_CODE_POINTS, split_text


def test_short_text_is_returned_trimmed():
    assert split_text("  Hello there.  ") == ("Hello there.",)


def test_whitespace_only_gives_nothing():
    assert split_text("   \n\t ") == ()


def test_limit_must_be_positive():
    with pytest.raises(ValueError):
        split_text("hi", limit=0)


def test_splits_at_sentence_boundaries_and_keeps_every_word():
    sentence = "This is a sentence that talks about nothing in particular, at some length."
    text = " ".join(sentence for _ in range(60))
    assert len(text) > MAX_INPUT_CODE_POINTS
    chunks = split_text(text)
    assert len(chunks) > 1
    assert all(1 <= len(c) <= MAX_INPUT_CODE_POINTS for c in chunks)
    assert all(c.endswith(".") for c in chunks)
    assert " ".join(chunks) == text


def test_sentence_longer_than_limit_is_split_at_whitespace():
    words = " ".join(f"word{i}" for i in range(50))
    chunks = split_text(words, limit=60)
    assert all(len(c) <= 60 for c in chunks)
    assert " ".join(chunks) == words
    assert all(not c.startswith(" ") and not c.endswith(" ") for c in chunks)


def test_single_oversized_word_is_cut_at_the_limit():
    word = "x" * 25
    chunks = split_text(word, limit=10)
    assert chunks == ("xxxxxxxxxx", "xxxxxxxxxx", "xxxxx")


def test_counts_code_points_not_bytes():
    accented = "é" * MAX_INPUT_CODE_POINTS
    assert split_text(accented) == (accented,)
    assert len(split_text(accented + "é")) == 2


def test_questions_and_exclamations_end_sentences():
    text = "Really? Yes! Fine."
    assert split_text(text, limit=8) == ("Really?", "Yes!", "Fine.")
