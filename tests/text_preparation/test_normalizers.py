#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test Normalizers
----------------------------------

Tests for Normalizers in the `text_preparation.normalizers` module.
"""
import pytest
from mindmeld.text_preparation.normalizers import (
    ASCIIFold,
    NFD,
    NFC,
    NFKD,
    NFKC,
    Lowercase,
    RegexNormalizerRuleFactory,
)

# TESTING NORMALIZER CLASSES

JA_SENTENCE_ONE = "紳士が過ぎ去った、 なぜそれが起こったのか誰にも分かりません！"
JA_SENTENCE_TWO = "株式会社ＫＡＤＯＫＡＷＡ Ｆｕｔｕｒｅ Ｐｕｂｌｉｓｈｉｎｇ"
JA_SENTENCE_THREE = "パピプペポ"
DE_SENTENCE_ONE = "Ein Gentleman ist vorbeigekommen, der weiß"
DE_SENTENCE_TWO = "Sie ist sehr kompetent,zuverlässig und vertrauenswürdig."
ES_SENTENCE_ONE = "Ha pasado un caballero, ¡quién sabe por qué pasó!"


@pytest.fixture
def ascii_fold_normalizer():
    return ASCIIFold()


@pytest.fixture
def nfd_normalizer():
    return NFD()


@pytest.fixture
def nfc_normalizer():
    return NFC()


@pytest.fixture
def nfkd_normalizer():
    return NFKD()


@pytest.fixture
def nfkc_normalizer():
    return NFKC()


def test_ascii_fold_ja(ascii_fold_normalizer):
    """ Ascii fold should not modify the Japanese sentence JA_SENTENCE_ONE"""
    normalized_text = ascii_fold_normalizer.normalize(JA_SENTENCE_ONE)
    assert normalized_text == JA_SENTENCE_ONE


def test_ascii_fold_de_one(ascii_fold_normalizer):
    """ Ascii fold should normalize the German weiss character."""
    normalized_text = ascii_fold_normalizer.normalize(DE_SENTENCE_ONE)
    assert normalized_text == "Ein Gentleman ist vorbeigekommen, der weiss"


def test_ascii_fold_de_two(ascii_fold_normalizer):
    """ Ascii fold should normalize the German unlauts (double dots over vowels)."""
    normalized_text = ascii_fold_normalizer.normalize(DE_SENTENCE_TWO)
    assert normalized_text == "Sie ist sehr kompetent,zuverlassig und vertrauenswurdig."


def test_ascii_fold_es_one(ascii_fold_normalizer):
    """ Ascii fold should normalize the accents in the Spanish sentence."""
    normalized_text = ascii_fold_normalizer.normalize(ES_SENTENCE_ONE)
    assert normalized_text == "Ha pasado un caballero, ¡quien sabe por que paso!"


@pytest.mark.parametrize(
    "sentence, expected_normalized_text",
    [
        (JA_SENTENCE_ONE, "紳士が過ぎ去った、 なぜそれが起こったのか誰にも分かりません！"),
        (JA_SENTENCE_TWO, "株式会社ＫＡＤＯＫＡＷＡ Ｆｕｔｕｒｅ Ｐｕｂｌｉｓｈｉｎｇ"),
        (JA_SENTENCE_THREE, "パピプペポ"),
        (DE_SENTENCE_ONE, "Ein Gentleman ist vorbeigekommen, der weiß"),
        (DE_SENTENCE_TWO, "Sie ist sehr kompetent,zuverlässig und vertrauenswürdig."),
        (ES_SENTENCE_ONE, "Ha pasado un caballero, ¡quién sabe por qué pasó!"),
    ],
)
def test_nfd_normalization(nfd_normalizer, sentence, expected_normalized_text):
    """ Testing NFD normalization."""
    normalized_text = nfd_normalizer.normalize(sentence)
    assert normalized_text == expected_normalized_text


@pytest.mark.parametrize(
    "sentence, expected_normalized_text",
    [
        (JA_SENTENCE_ONE, JA_SENTENCE_ONE),
        (JA_SENTENCE_TWO, JA_SENTENCE_TWO),
        (JA_SENTENCE_THREE, JA_SENTENCE_THREE),
        (DE_SENTENCE_ONE, DE_SENTENCE_ONE),
        (DE_SENTENCE_TWO, DE_SENTENCE_TWO),
        (ES_SENTENCE_ONE, ES_SENTENCE_ONE),
    ],
)
def test_nfc_normalization(nfc_normalizer, sentence, expected_normalized_text):
    """ Testing NFC normalization."""
    normalized_text = nfc_normalizer.normalize(sentence)
    assert normalized_text == expected_normalized_text


@pytest.mark.parametrize(
    "sentence, expected_normalized_text",
    [
        (JA_SENTENCE_ONE, "紳士が過ぎ去った、 なぜそれが起こったのか誰にも分かりません!"),
        (JA_SENTENCE_TWO, "株式会社KADOKAWA Future Publishing"),
        (JA_SENTENCE_THREE, "パピプペポ"),
        (DE_SENTENCE_ONE, DE_SENTENCE_ONE),
        (DE_SENTENCE_TWO, "Sie ist sehr kompetent,zuverlässig und vertrauenswürdig."),
        (ES_SENTENCE_ONE, "Ha pasado un caballero, ¡quién sabe por qué pasó!"),
    ],
)
def test_nfkd_normalization(nfkd_normalizer, sentence, expected_normalized_text):
    """ Testing NFKD normalization."""
    normalized_text = nfkd_normalizer.normalize(sentence)
    assert normalized_text == expected_normalized_text


@pytest.mark.parametrize(
    "sentence, expected_normalized_text",
    [
        (JA_SENTENCE_ONE, "紳士が過ぎ去った、 なぜそれが起こったのか誰にも分かりません!"),
        (JA_SENTENCE_TWO, "株式会社KADOKAWA Future Publishing"),
        (JA_SENTENCE_THREE, JA_SENTENCE_THREE),
        (DE_SENTENCE_ONE, DE_SENTENCE_ONE),
        (DE_SENTENCE_TWO, DE_SENTENCE_TWO),
        (ES_SENTENCE_ONE, ES_SENTENCE_ONE),
    ],
)
def test_nfkc_normalization(nfkc_normalizer, sentence, expected_normalized_text):
    """ Testing NFKC normalization."""
    normalized_text = nfkc_normalizer.normalize(sentence)
    assert normalized_text == expected_normalized_text


def test_lowercase_normalization():
    assert Lowercase().normalize("Hello, How Are You?") == "hello, how are you?"


def test_get_regex_normalizers():

    regex_norm_rule = {"pattern": ".*", "replacement": ""}
    regex_normalizer = RegexNormalizerRuleFactory.get_regex_normalizers(
        [regex_norm_rule]
    )[0]
    assert regex_normalizer.normalize("Cisco") == ""


# TESTING NORMALIZER CLASSES


def _check_match(text_preparation_pipeline, regex_norm_rule, input_text, expected_text):

    regex_normalizer = RegexNormalizerRuleFactory.get_default_regex_normalizer_rule(
        regex_norm_rule
    )
    text_preparation_pipeline.normalizers = [regex_normalizer]
    normalized_text = text_preparation_pipeline.normalize(input_text)
    assert normalized_text == expected_text


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("that's dennis' truck", "that's dennis truck"),
        ("where's luciens' cat?", "where's luciens cat?"),
        ("JAMES' CAR", "JAMES CAR"),
    ],
)
def test_remove_apos_at_end_of_possesive_form(
    text_preparation_pipeline, input_text, expected_text
):
    _check_match(
        text_preparation_pipeline,
        "RemoveAposAtEndOfPossesiveForm",
        input_text,
        expected_text,
    )


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("          MindMeld", "MindMeld"),
        ("      ", ""),
        ("      How are you?", "How are you?"),
        ("       わくわくしてます!", "わくわくしてます!")
    ],
)
def test_remove_beginning_space(text_preparation_pipeline, input_text, expected_text):
    _check_match(
        text_preparation_pipeline, "RemoveBeginningSpace", input_text, expected_text
    )


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("MindMeld           ", "MindMeld"),
        ("      ", ""),
        ("How are you?     ", "How are you?"),
        ("わくわくしてます!     ", "わくわくしてます!")
    ],
)
def test_remove_trailing_space(text_preparation_pipeline, input_text, expected_text):
    _check_match(
        text_preparation_pipeline, "RemoveTrailingSpace", input_text, expected_text
    )


@pytest.mark.parametrize(
    "input_text, expected_text",
    [("How    are    you?", "How are you?"), ("I          am   fine!", "I am fine!")],
)
def test_replace_spaces_with_space(
    text_preparation_pipeline, input_text, expected_text
):
    _check_match(
        text_preparation_pipeline, "ReplaceSpacesWithSpace", input_text, expected_text
    )


@pytest.mark.parametrize(
    "input_text, expected_text",
    [("How_are_you?", "How are you?"), ("I_am_fine", "I am fine")],
)
def test_replace_underscore_with_space(
    text_preparation_pipeline, input_text, expected_text
):
    _check_match(
        text_preparation_pipeline,
        "ReplaceUnderscoreWithSpace",
        input_text,
        expected_text,
    )


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("mindmeld's code", "mindmeld 's code"),
        ("k's", "k 's"),
        ("CARL'S PIZZA", "CARL 's PIZZA"),
    ],
)
def test_separate_apos_s(text_preparation_pipeline, input_text, expected_text):
    _check_match(text_preparation_pipeline, "SeparateAposS", input_text, expected_text)


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("****mindmeld", "mindmeld"),
        ("%#++=-=CISCO", "CISCO"),
        ("///// //// //// NLP", "NLP"),
    ],
)
def test_replace_punctuation_at_word_start_with_space(
    text_preparation_pipeline, input_text, expected_text
):
    _check_match(
        text_preparation_pipeline,
        "ReplacePunctuationAtWordStartWithSpace",
        input_text,
        expected_text,
    )


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("MindMeld*** is** the** best", "MindMeld is the best"),
        ("How%+=* are++- you^^%", "How are you"),
        ("I have $500 in my pocket!!!", "I have $500 in my pocket"),
    ],
)
def test_replace_punctuation_at_word_end_with_space(
    text_preparation_pipeline, input_text, expected_text
):
    _check_match(
        text_preparation_pipeline,
        "ReplacePunctuationAtWordEndWithSpace",
        input_text,
        expected_text,
    )


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("Lucien has//++=/1 cat", "Lucien has 1 cat"),
        ("Why!!!7 days?", "Why 7 days?"),
        ("Racing^^!#%%24 hours##%$7 days", "Racing 24 hours 7 days"),
    ],
)
def test_replace_special_chars_between_letters_and_digits_with_space(
    text_preparation_pipeline, input_text, expected_text
):
    _check_match(
        text_preparation_pipeline,
        "ReplaceSpecialCharsBetweenLettersAndDigitsWithSpace",
        input_text,
        expected_text,
    )


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("Lucien has 1//++=/cat", "Lucien has 1 cat"),
        ("Why 7!!!days?", "Why 7 days?"),
        ("Racing 24^^!#%%hours 7##%$days", "Racing 24 hours 7 days"),
    ],
)
def test_replace_special_chars_between_digits_and_letters_with_space(
    text_preparation_pipeline, input_text, expected_text
):
    _check_match(
        text_preparation_pipeline,
        "ReplaceSpecialCharsBetweenDigitsAndLettersWithSpace",
        input_text,
        expected_text,
    )


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("Lucien has one//++=/cat", "Lucien has one cat"),
        ("Why seven!!!days?", "Why seven days?"),
        ("Racing all^^!#%%hours seven##%$days", "Racing all hours seven days"),
    ],
)
def test_replace_special_chars_between_letters_with_space(
    text_preparation_pipeline, input_text, expected_text
):
    _check_match(
        text_preparation_pipeline,
        "ReplaceSpecialCharsBetweenLettersWithSpace",
        input_text,
        expected_text,
    )


# TESTING NORMALIZATION ON SAMPLE SENTENCES


@pytest.mark.parametrize(
    "raw_text, expected_text",
    [
        ("Test: Query for $500,000.", "test query for $500,000"),
        (
            "Test: Query for test.12.345..test,test",
            "test query for test 12.345 test test",
        ),
        ("Test: awesome band sigur rós.", "test awesome band sigur ros"),
        ("D'Angelo's new album", "d'angelo 's new album"),
        ("is s.o.b. ,, gonna be on at 8 p.m.?", "is s o b gonna be on at 8 p m"),
        ("join O'reilly's pmr", "join o'reilly 's pmr"),
    ],
)
def test_normalization_on_sample_sentences(
    text_preparation_pipeline, raw_text, expected_text
):
    assert expected_text == text_preparation_pipeline.normalize(raw_text)


# TESTING NORMALIZATION MAPPING


def test_mapping(text_preparation_pipeline):
    raw = "Test: 1. 2. 3."
    normalized = text_preparation_pipeline.normalize(raw)

    assert normalized == "test 1 2 3"

    forward, backward = text_preparation_pipeline.get_char_index_map(raw, normalized)

    assert forward == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 3,
        5: 4,
        6: 5,
        7: 5,
        8: 6,
        9: 7,
        10: 7,
        11: 8,
        12: 9,
        13: 9,
    }

    assert backward == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 5,
        5: 6,
        6: 8,
        7: 9,
        8: 11,
        9: 12,
    }


def test_mapping_2(text_preparation_pipeline):
    raw = "is s.o.b. ,, gonna be on at 8 p.m.?"
    normalized = text_preparation_pipeline.normalize(raw)

    assert normalized == "is s o b gonna be on at 8 p m"

    forward, backward = text_preparation_pipeline.get_char_index_map(raw, normalized)

    assert forward == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 7,
        9: 8,
        10: 8,
        11: 8,
        12: 8,
        13: 9,
        14: 10,
        15: 11,
        16: 12,
        17: 13,
        18: 14,
        19: 15,
        20: 16,
        21: 17,
        22: 18,
        23: 19,
        24: 20,
        25: 21,
        26: 22,
        27: 23,
        28: 24,
        29: 25,
        30: 26,
        31: 27,
        32: 28,
        33: 28,
        34: 28,
    }

    assert backward == {
        0: 0,
        1: 1,
        2: 2,
        3: 3,
        4: 4,
        5: 5,
        6: 6,
        7: 7,
        8: 9,
        9: 13,
        10: 14,
        11: 15,
        12: 16,
        13: 17,
        14: 18,
        15: 19,
        16: 20,
        17: 21,
        18: 22,
        19: 23,
        20: 24,
        21: 25,
        22: 26,
        23: 27,
        24: 28,
        25: 29,
        26: 30,
        27: 31,
        28: 32,
    }
