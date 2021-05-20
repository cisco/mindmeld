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
        {"start": 0, "text": "Hello"},
        {"start": 6, "text": "Sir."},
        {"start": 11, "text": "I'd"},
        {"start": 15, "text": "like"},
        {"start": 20, "text": "to"},
        {"start": 23, "text": "tell"},
        {"start": 28, "text": "you"},
        {"start": 32, "text": "how"},
        {"start": 36, "text": "much"},
        {"start": 41, "text": "I"},
        {"start": 43, "text": "like"},
        {"start": 48, "text": "MindMeld."},
    ]
    assert tokenized_output == expected_output


def test_white_space_tokenizer_es(white_space_tokenizer):
    tokenized_output = white_space_tokenizer.tokenize(ES_SENTENCE_ONE)
    expected_output = [
        {"start": 0, "text": "Ha"},
        {"start": 3, "text": "pasado"},
        {"start": 10, "text": "un"},
        {"start": 13, "text": "caballero,"},
        {"start": 24, "text": "¡quién"},
        {"start": 31, "text": "sabe"},
        {"start": 36, "text": "por"},
        {"start": 40, "text": "qué"},
        {"start": 44, "text": "pasó!"},
    ]
    assert tokenized_output == expected_output


def test_white_space_tokenizer_de(white_space_tokenizer):
    tokenized_output = white_space_tokenizer.tokenize(DE_SENTENCE_ONE)
    expected_output = [
        {"start": 0, "text": "Ein"},
        {"start": 4, "text": "Gentleman"},
        {"start": 14, "text": "ist"},
        {"start": 18, "text": "vorbeigekommen,"},
        {"start": 34, "text": "der"},
        {"start": 38, "text": "weiß"},
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
        {"start": 0, "text": "紳士"},
        {"start": 2, "text": "が"},
        {"start": 3, "text": "過ぎ"},
        {"start": 5, "text": "去っ"},
        {"start": 7, "text": "た"},
        {"start": 8, "text": "、"},
        {"start": 9, "text": "なぜ"},
        {"start": 11, "text": "それ"},
        {"start": 13, "text": "が"},
        {"start": 14, "text": "起こっ"},
        {"start": 17, "text": "た"},
        {"start": 18, "text": "の"},
        {"start": 19, "text": "か"},
        {"start": 20, "text": "誰"},
        {"start": 21, "text": "に"},
        {"start": 22, "text": "も"},
        {"start": 23, "text": "分かり"},
        {"start": 26, "text": "ませ"},
        {"start": 28, "text": "ん"},
        {"start": 29, "text": "！"},
    ]
    assert tokenized_output == expected_output


def test_spacy_tokenizer_ja_two(spacy_tokenizer_ja):
    tokenized_output = spacy_tokenizer_ja.tokenize(JA_SENTENCE_TWO)
    expected_output = [
        {"start": 0, "text": "株式"},
        {"start": 2, "text": "会社"},
        {"start": 4, "text": "ＫＡＤＯＫＡＷＡ"},
        {"start": 12, "text": "Ｆｕｔｕｒｅ"},
        {"start": 18, "text": "Ｐｕｂｌｉｓｈｉｎｇ"},
    ]
    assert tokenized_output == expected_output


def test_spacy_tokenizer_ja_four(spacy_tokenizer_ja):
    tokenized_output = spacy_tokenizer_ja.tokenize(JA_SENTENCE_FOUR)
    expected_output = [
        {"start": 0, "text": "サウンド"},
        {"start": 4, "text": "レベル"},
        {"start": 7, "text": "アップ"},
        {"start": 10, "text": "し"},
        {"start": 11, "text": "て"},
        {"start": 12, "text": "下さい"},
    ]
    assert tokenized_output == expected_output
