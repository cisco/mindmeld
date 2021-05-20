#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test Tokenizers
----------------------------------

Tests for Tokenizers in the `text_preparation.tokenizers` module.
"""
import pytest


from mindmeld.text_preparation.tokenizers import (
    LetterTokenizer,
    WhiteSpaceTokenizer,
    SpacyTokenizer,
)

JA_SENTENCE_ONE = "紳士が過ぎ去った、 なぜそれが起こったのか誰にも分かりません！"
JA_SENTENCE_TWO = "株式会社ＫＡＤＯＫＡＷＡ Ｆｕｔｕｒｅ Ｐｕｂｌｉｓｈｉｎｇ"
JA_SENTENCE_THREE = "パピプペポ"
JA_SENTENCE_FOUR = "サウンドレベルアップして下さい"
DE_SENTENCE_ONE = "Ein Gentleman ist vorbeigekommen, der weiß"
ES_SENTENCE_ONE = "Ha pasado un caballero, ¡quién sabe por qué pasó!"
EN_SENTENCE_ONE = "Hello Sir. I'd like to tell you how much I like MindMeld."


@pytest.fixture
def white_space_tokenizer():
    return WhiteSpaceTokenizer()


@pytest.fixture
def spacy_tokenizer_ja():
    return SpacyTokenizer(language="ja", spacy_model_size="sm")


@pytest.fixture
def letter_tokenizer():
    return LetterTokenizer()


def test_white_space_tokenizer_en(white_space_tokenizer):
    tokenized_output = white_space_tokenizer.tokenize(EN_SENTENCE_ONE)
    expected_output = [
        {"start": 0, "text": "Hello", "end": 4},
        {"start": 6, "text": "Sir.", "end": 9},
        {"start": 11, "text": "I'd", "end": 13},
        {"start": 15, "text": "like", "end": 18},
        {"start": 20, "text": "to", "end": 21},
        {"start": 23, "text": "tell", "end": 26},
        {"start": 28, "text": "you", "end": 30},
        {"start": 32, "text": "how", "end": 34},
        {"start": 36, "text": "much", "end": 39},
        {"start": 41, "text": "I", "end": 41},
        {"start": 43, "text": "like", "end": 46},
        {"start": 48, "text": "MindMeld.", "end": 56},
    ]
    assert tokenized_output == expected_output


def test_white_space_tokenizer_es(white_space_tokenizer):
    tokenized_output = white_space_tokenizer.tokenize(ES_SENTENCE_ONE)
    expected_output = [
        {"start": 0, "text": "Ha", "end": 1},
        {"start": 3, "text": "pasado", "end": 8},
        {"start": 10, "text": "un", "end": 11},
        {"start": 13, "text": "caballero,", "end": 22},
        {"start": 24, "text": "¡quién", "end": 29},
        {"start": 31, "text": "sabe", "end": 34},
        {"start": 36, "text": "por", "end": 38},
        {"start": 40, "text": "qué", "end": 42},
        {"start": 44, "text": "pasó!", "end": 48},
    ]
    assert tokenized_output == expected_output


def test_white_space_tokenizer_de(white_space_tokenizer):
    tokenized_output = white_space_tokenizer.tokenize(DE_SENTENCE_ONE)
    expected_output = [
        {"start": 0, "text": "Ein", "end": 2},
        {"start": 4, "text": "Gentleman", "end": 12},
        {"start": 14, "text": "ist", "end": 16},
        {"start": 18, "text": "vorbeigekommen,", "end": 32},
        {"start": 34, "text": "der", "end": 36},
        {"start": 38, "text": "weiß", "end": 41},
    ]
    assert tokenized_output == expected_output


def test_character_tokenizer_ja(letter_tokenizer):
    tokenized_output = letter_tokenizer.tokenize(JA_SENTENCE_ONE)
    expected_output = [
        {"start": 0, "text": "紳", "end": 0},
        {"start": 1, "text": "士", "end": 1},
        {"start": 2, "text": "が", "end": 2},
        {"start": 3, "text": "過", "end": 3},
        {"start": 4, "text": "ぎ", "end": 4},
        {"start": 5, "text": "去", "end": 5},
        {"start": 6, "text": "っ", "end": 6},
        {"start": 7, "text": "た", "end": 7},
        {"start": 8, "text": "、", "end": 8},
        {"start": 10, "text": "な", "end": 10},
        {"start": 11, "text": "ぜ", "end": 11},
        {"start": 12, "text": "そ", "end": 12},
        {"start": 13, "text": "れ", "end": 13},
        {"start": 14, "text": "が", "end": 14},
        {"start": 15, "text": "起", "end": 15},
        {"start": 16, "text": "こ", "end": 16},
        {"start": 17, "text": "っ", "end": 17},
        {"start": 18, "text": "た", "end": 18},
        {"start": 19, "text": "の", "end": 19},
        {"start": 20, "text": "か", "end": 20},
        {"start": 21, "text": "誰", "end": 21},
        {"start": 22, "text": "に", "end": 22},
        {"start": 23, "text": "も", "end": 23},
        {"start": 24, "text": "分", "end": 24},
        {"start": 25, "text": "か", "end": 25},
        {"start": 26, "text": "り", "end": 26},
        {"start": 27, "text": "ま", "end": 27},
        {"start": 28, "text": "せ", "end": 28},
        {"start": 29, "text": "ん", "end": 29},
        {"start": 30, "text": "！", "end": 30},
    ]
    assert tokenized_output == expected_output


def test_spacy_tokenizer_ja_one(spacy_tokenizer_ja):
    tokenized_output = spacy_tokenizer_ja.tokenize(JA_SENTENCE_ONE)
    expected_output = [
        {"start": 0, "text": "紳士", "end": 1},
        {"start": 2, "text": "が", "end": 2},
        {"start": 3, "text": "過ぎ", "end": 4},
        {"start": 5, "text": "去っ", "end": 6},
        {"start": 7, "text": "た", "end": 7},
        {"start": 8, "text": "、", "end": 8},
        {"start": 9, "text": "なぜ", "end": 10},
        {"start": 11, "text": "それ", "end": 12},
        {"start": 13, "text": "が", "end": 13},
        {"start": 14, "text": "起こっ", "end": 16},
        {"start": 17, "text": "た", "end": 17},
        {"start": 18, "text": "の", "end": 18},
        {"start": 19, "text": "か", "end": 19},
        {"start": 20, "text": "誰", "end": 20},
        {"start": 21, "text": "に", "end": 21},
        {"start": 22, "text": "も", "end": 22},
        {"start": 23, "text": "分かり", "end": 25},
        {"start": 26, "text": "ませ", "end": 27},
        {"start": 28, "text": "ん", "end": 28},
        {"start": 29, "text": "！", "end": 29},
    ]
    assert tokenized_output == expected_output


def test_spacy_tokenizer_ja_two(spacy_tokenizer_ja):
    tokenized_output = spacy_tokenizer_ja.tokenize(JA_SENTENCE_TWO)
    expected_output = [
        {"start": 0, "text": "株式", "end": 1},
        {"start": 2, "text": "会社", "end": 3},
        {"start": 4, "text": "ＫＡＤＯＫＡＷＡ", "end": 11},
        {"start": 12, "text": "Ｆｕｔｕｒｅ", "end": 17},
        {"start": 18, "text": "Ｐｕｂｌｉｓｈｉｎｇ", "end": 27},
    ]
    assert tokenized_output == expected_output


def test_spacy_tokenizer_ja_four(spacy_tokenizer_ja):
    tokenized_output = spacy_tokenizer_ja.tokenize(JA_SENTENCE_FOUR)
    expected_output = [
        {"start": 0, "text": "サウンド", "end": 3},
        {"start": 4, "text": "レベル", "end": 6},
        {"start": 7, "text": "アップ", "end": 9},
        {"start": 10, "text": "し", "end": 10},
        {"start": 11, "text": "て", "end": 11},
        {"start": 12, "text": "下さい", "end": 14},
    ]
    assert tokenized_output == expected_output
