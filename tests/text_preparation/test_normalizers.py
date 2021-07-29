#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test Normalizers
----------------------------------

Tests for Normalizers in the `text_preparation.normalizers` module.
"""
import pytest

from mindmeld.text_preparation.normalizers import ASCIIFold, NFD, NFC, NFKD, NFKC

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


# TESTING NORMALIZATION ON SAMPLE SENTENCES

@pytest.mark.parametrize(
    "raw_text, expected_text",
    [
        ("Test: Query for $500,000.", "test query for $500,000"),
        ("Test: Query for test.12.345..test,test", "test query for test 12.345 test test"),
        ("Test: awesome band sigur rós.", "test awesome band sigur ros"),
        ("D'Angelo's new album", "d'angelo 's new album"),
        ("is s.o.b. ,, gonna be on at 8 p.m.?", "is s o b gonna be on at 8 p m"),
        ("join O'reilly's pmr", "join o'reilly 's pmr")
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
