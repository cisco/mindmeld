#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test Normalizers
----------------------------------

Tests for Normalizers in the `text_preparation.normalizers` module.
"""
import pytest

from mindmeld.text_preparation.normalizers import ASCIIFold, NFD, NFC, NFKD, NFKC


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
