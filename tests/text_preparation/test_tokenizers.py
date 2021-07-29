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
    CharacterTokenizer,
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


@pytest.fixture
def character_tokenizer():
    return CharacterTokenizer()


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


def test_letter_tokenizer_ja(letter_tokenizer):
    tokenized_output = letter_tokenizer.tokenize(JA_SENTENCE_ONE)
    expected_output = [
        {"start": 0, "text": "紳"},
        {"start": 1, "text": "士"},
        {"start": 2, "text": "が"},
        {"start": 3, "text": "過"},
        {"start": 4, "text": "ぎ"},
        {"start": 5, "text": "去"},
        {"start": 6, "text": "っ"},
        {"start": 7, "text": "た"},
        {"start": 8, "text": "、"},
        {"start": 10, "text": "な"},
        {"start": 11, "text": "ぜ"},
        {"start": 12, "text": "そ"},
        {"start": 13, "text": "れ"},
        {"start": 14, "text": "が"},
        {"start": 15, "text": "起"},
        {"start": 16, "text": "こ"},
        {"start": 17, "text": "っ"},
        {"start": 18, "text": "た"},
        {"start": 19, "text": "の"},
        {"start": 20, "text": "か"},
        {"start": 21, "text": "誰"},
        {"start": 22, "text": "に"},
        {"start": 23, "text": "も"},
        {"start": 24, "text": "分"},
        {"start": 25, "text": "か"},
        {"start": 26, "text": "り"},
        {"start": 27, "text": "ま"},
        {"start": 28, "text": "せ"},
        {"start": 29, "text": "ん"},
        {"start": 30, "text": "！"},
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


def test_character_tokenizer_de_one(character_tokenizer):
    tokenized_output = character_tokenizer.tokenize(DE_SENTENCE_ONE)
    expected_output = [
        {"start": 0, "text": "E"},
        {"start": 1, "text": "i"},
        {"start": 2, "text": "n"},
        {"start": 4, "text": "G"},
        {"start": 5, "text": "e"},
        {"start": 6, "text": "n"},
        {"start": 7, "text": "t"},
        {"start": 8, "text": "l"},
        {"start": 9, "text": "e"},
        {"start": 10, "text": "m"},
        {"start": 11, "text": "a"},
        {"start": 12, "text": "n"},
        {"start": 14, "text": "i"},
        {"start": 15, "text": "s"},
        {"start": 16, "text": "t"},
        {"start": 18, "text": "v"},
        {"start": 19, "text": "o"},
        {"start": 20, "text": "r"},
        {"start": 21, "text": "b"},
        {"start": 22, "text": "e"},
        {"start": 23, "text": "i"},
        {"start": 24, "text": "g"},
        {"start": 25, "text": "e"},
        {"start": 26, "text": "k"},
        {"start": 27, "text": "o"},
        {"start": 28, "text": "m"},
        {"start": 29, "text": "m"},
        {"start": 30, "text": "e"},
        {"start": 31, "text": "n"},
        {"start": 32, "text": ","},
        {"start": 34, "text": "d"},
        {"start": 35, "text": "e"},
        {"start": 36, "text": "r"},
        {"start": 38, "text": "w"},
        {"start": 39, "text": "e"},
        {"start": 40, "text": "i"},
        {"start": 41, "text": "ß"},
    ]
    assert tokenized_output == expected_output


def test_character_tokenizer_ja_three(character_tokenizer):
    tokenized_output = character_tokenizer.tokenize(JA_SENTENCE_THREE)
    expected_output = [
        {"start": 0, "text": "パ"},
        {"start": 1, "text": "ピ"},
        {"start": 2, "text": "プ"},
        {"start": 3, "text": "ペ"},
        {"start": 4, "text": "ポ"},
    ]
    assert tokenized_output == expected_output


def test_tokenize(text_preparation_pipeline):
    raw_tokens = text_preparation_pipeline.tokenize("Test: Query for $500,000. Chyea!")
    tokens = text_preparation_pipeline._normalize_tokens(raw_tokens)

    assert len(tokens)
    assert tokens[0]["entity"] == "test"
    assert tokens[0]["raw_entity"] == "Test:"
    assert tokens[1]["raw_start"] == 6
    assert tokens[3]["raw_entity"] == "$500,000."
    assert tokens[3]["raw_start"] == 16
    assert tokens[3]["entity"] == "$500,000"
    assert tokens[4]["entity"] == "chyea"
    assert tokens[4]["raw_entity"] == "Chyea!"
    assert tokens[4]["raw_start"] == 26


def test_tokenize_none(white_space_tokenizer, spacy_tokenizer_ja, letter_tokenizer, character_tokenizer):
    assert white_space_tokenizer.tokenize(None) == []
    assert spacy_tokenizer_ja.tokenize(None) == []
    assert letter_tokenizer.tokenize(None) == []
    assert character_tokenizer.tokenize(None) == []
