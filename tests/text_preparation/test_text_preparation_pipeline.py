#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test TextPreparationPipeline
----------------------------------

Tests for TextPreparationPipeline in the `text_preparation.text_preparation_pipeline` module.
"""
import pytest

from mindmeld.components._config import ENGLISH_LANGUAGE_CODE
from mindmeld.text_preparation.normalizers import (
    ASCIIFold,
    Normalizer,
    NoOpNormalizer,
    RegexNormalizerRule,
    Lowercase,
)
from mindmeld.text_preparation.preprocessors import NoOpPreprocessor, Preprocessor
from mindmeld.text_preparation.stemmers import EnglishNLTKStemmer
from mindmeld.text_preparation.tokenizers import NoOpTokenizer, SpacyTokenizer
from mindmeld.text_preparation.text_preparation_pipeline import (
    TextPreparationPipeline,
    TextPreparationPipelineError,
    TextPreparationPipelineFactory,
)
from mindmeld.text_preparation.tokenizers import WhiteSpaceTokenizer


def test_text_preparation_pipeline_tokenizer_not_none():
    with pytest.raises(TextPreparationPipelineError):
        TextPreparationPipeline(tokenizer=None)


def test_offset_token_start_values():
    sample_token = {"start": 3}
    TextPreparationPipeline.offset_token_start_values(tokens=[sample_token], offset=5)
    assert sample_token["start"] == 8


def test_filter_out_space_text_tokens():
    input_tokens = [
        {"text": "How"},
        {"text": "   "},
        {"text": "are"},
        {"text": " "},
        {"text": "you"},
        {"text": "?"},
    ]
    expected_output_tokens = [
        {"text": "How"},
        {"text": "are"},
        {"text": "you"},
        {"text": "?"},
    ]
    output_tokens = TextPreparationPipeline.filter_out_space_text_tokens(input_tokens)
    assert expected_output_tokens == output_tokens


def test_find_mindmeld_annotation_re_matches():
    sentence = "Hello {Lucien|sys_person|employee}. Do you have {1|sys_number} cat?"
    matches = TextPreparationPipeline.find_mindmeld_annotation_re_matches(sentence)
    assert len(matches) == 2
    first_match, second_match = matches
    assert first_match.group(1) == "Lucien"
    assert first_match.span() == (6, 34)
    assert second_match.group(1) == "1"
    assert second_match.span() == (48, 62)


def test_normalize_around_annoations():
    sentence = "HELLO {LUCIEN|PERSON_NAME}, HOW ARE YOU?"
    normalized_sentence = TextPreparationPipeline.modify_around_annotations(
        text=sentence, function=Lowercase().normalize
    )
    expected_normalized_sentence = "hello {lucien|PERSON_NAME}, how are you?"
    assert normalized_sentence == expected_normalized_sentence


def test_tokenize_around_annoations():
    text_preparation_pipeline = (
        TextPreparationPipelineFactory.create_default_text_preparation_pipeline()
    )
    sentence = "HELLO {LUCIEN|PERSON_NAME}, HOW ARE YOU?"
    raw_tokens = text_preparation_pipeline.tokenize(sentence)

    expected_raw_tokens = [
        {"start": 0, "text": "HELLO"},
        {"start": 7, "text": "LUCIEN"},
        {"start": 26, "text": ","},
        {"start": 28, "text": "HOW"},
        {"start": 32, "text": "ARE"},
        {"start": 36, "text": "YOU?"},
    ]
    assert raw_tokens == expected_raw_tokens


def test_create_text_preparation_pipeline():
    text_preparation_pipeline = (
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            language=ENGLISH_LANGUAGE_CODE,
            preprocessors=[],
            regex_norm_rules=[{"pattern": ".*", "replacement": "cisco"}],
            normalizers=["Lowercase", "ASCIIFold"],
            tokenizer="WhiteSpaceTokenizer",
            stemmer=None,
        )
    )

    assert text_preparation_pipeline.language == ENGLISH_LANGUAGE_CODE
    assert isinstance(text_preparation_pipeline.preprocessors[0], NoOpPreprocessor)
    assert isinstance(text_preparation_pipeline.normalizers[0], RegexNormalizerRule)
    assert isinstance(text_preparation_pipeline.normalizers[1], Lowercase)
    assert isinstance(text_preparation_pipeline.normalizers[2], ASCIIFold)
    assert isinstance(text_preparation_pipeline.tokenizer, WhiteSpaceTokenizer)
    assert isinstance(text_preparation_pipeline.stemmer, EnglishNLTKStemmer)


def test_text_preparation_pipeline_hash():
    text_preparation_pipeline = (
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            language=ENGLISH_LANGUAGE_CODE,
            preprocessors=["NoOpPreprocessor"],
            regex_norm_rules=[{"pattern": ".*", "replacement": "cisco"}],
            normalizers=["Lowercase", "ASCIIFold"],
            tokenizer="WhiteSpaceTokenizer",
            stemmer=None,
        )
    )

    original_hash = text_preparation_pipeline.get_hashid()

    # Change order of normalizers
    text_preparation_pipeline = (
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            language=ENGLISH_LANGUAGE_CODE,
            preprocessors=["NoOpPreprocessor"],
            regex_norm_rules=[{"pattern": ".*", "replacement": "cisco"}],
            normalizers=["ASCIIFold", "Lowercase"],
            tokenizer="WhiteSpaceTokenizer",
            stemmer=None,
        )
    )
    order_changed_hash = text_preparation_pipeline.get_hashid()

    # Change RegexNormalizer pattern
    text_preparation_pipeline = (
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            language=ENGLISH_LANGUAGE_CODE,
            preprocessors=["NoOpPreprocessor"],
            regex_norm_rules=[{"pattern": ".*", "replacement": "cisc0"}],
            normalizers=["ASCIIFold", "Lowercase"],
            tokenizer="WhiteSpaceTokenizer",
            stemmer=None,
        )
    )

    regex_changed_hash = text_preparation_pipeline.get_hashid()

    # Change Tokenizer type
    text_preparation_pipeline = (
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            language=ENGLISH_LANGUAGE_CODE,
            preprocessors=["NoOpPreprocessor"],
            regex_norm_rules=[{"pattern": ".*", "replacement": "cisco"}],
            normalizers=["ASCIIFold", "Lowercase"],
            tokenizer="LetterTokenizer",
            stemmer=None,
        )
    )

    tokenizer_changed_hash = text_preparation_pipeline.get_hashid()

    assert original_hash != order_changed_hash
    assert original_hash != regex_changed_hash
    assert original_hash != tokenizer_changed_hash
    assert tokenizer_changed_hash != regex_changed_hash
    assert tokenizer_changed_hash != order_changed_hash
    assert regex_changed_hash != order_changed_hash


def test_construct_pipeline_components_valid_input():
    text_preparation_pipeline = (
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            preprocessors=("NoOpPreprocessor", NoOpPreprocessor()),
            normalizers=(
                "RemoveBeginningSpace",
                NoOpNormalizer(),
                "ReplaceSpacesWithSpace",
                Lowercase(),
            ),
            tokenizer="SpacyTokenizer",
            stemmer=None,
        )
    )

    assert text_preparation_pipeline.language == ENGLISH_LANGUAGE_CODE
    for preprocessor in text_preparation_pipeline.preprocessors:
        assert isinstance(preprocessor, Preprocessor)
    for normalizer in text_preparation_pipeline.normalizers:
        assert isinstance(normalizer, Normalizer)
    assert isinstance(text_preparation_pipeline.tokenizer, SpacyTokenizer)
    assert isinstance(text_preparation_pipeline.stemmer, EnglishNLTKStemmer)


def test_construct_pipeline_components_invalid_input():
    with pytest.raises(TypeError):
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            preprocessors=None,
            normalizers=("SpacyTokenizer"),
            tokenizer=None,
            stemmer=None,
        )

    with pytest.raises(TypeError):
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            preprocessors=None,
            normalizers=(NoOpTokenizer(), "NoOpTokenizer"),
            tokenizer=None,
            stemmer=None,
        )

    with pytest.raises(TypeError):
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            preprocessors=None,
            normalizers=None,
            tokenizer="NoOpNormalizer",
            stemmer=None,
        )

    with pytest.raises(TypeError):
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            preprocessors=None,
            normalizers=None,
            tokenizer=None,
            stemmer="NoOpPreprocessor",
        )

    with pytest.raises(TypeError):
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            preprocessors=("NoOpNormalizer"),
            normalizers=None,
            tokenizer=None,
            stemmer=None,
        )

    with pytest.raises(TypeError):
        TextPreparationPipelineFactory.create_text_preparation_pipeline(
            preprocessors=(NoOpNormalizer(), NoOpTokenizer()),
            normalizers=None,
            tokenizer=None,
            stemmer=None,
        )
