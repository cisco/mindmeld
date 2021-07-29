#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test Regex Normalizers
----------------------------------

Tests for Regex Normalizers in the `text_preparation.normalizers` module.
"""
import pytest
from mindmeld.text_preparation.normalizers import DEFAULT_REGEX_NORM_RULES


def _check_match(text_preparation_pipeline, regex_norm_rule, input_text, expected_text):
    text_preparation_pipeline.normalizers = [DEFAULT_REGEX_NORM_RULES[regex_norm_rule]]
    normalized_text = text_preparation_pipeline.normalize(input_text)
    assert normalized_text == expected_text


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("that's dennis' truck", "that's dennis truck"),
        ("where's luciens' cat?", "where's luciens cat?"),
        ("JAMES' CAR", "JAMES CAR")
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


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("Lucien has one//++=/|sys_number cat", "Lucien has one|sys_number cat"),
        ("Why seven!!!|sys_number", "Why seven|sys_number"),
        ("Racing all^^!#%%|custom_entity hours", "Racing all|custom_entity hours"),
    ],
)
def test_replace_special_chars_before_pipe(
    text_preparation_pipeline, input_text, expected_text
):
    _check_match(
        text_preparation_pipeline,
        "RemoveSpecialCharsBeforePipe",
        input_text,
        expected_text,
    )


@pytest.mark.parametrize(
    "input_text, expected_text",
    [
        ("Lucien}+=-+s cat", "Lucien s cat"),
        ("John]+=-+s dog", "John s dog"),
    ],
)
def test_replace_end_bracket_and_following_special_chars_before_s_with_space(
    text_preparation_pipeline, input_text, expected_text
):
    _check_match(
        text_preparation_pipeline,
        "ReplaceEndBracketAndFollowingSpecialCharsBeforeSWithSpace",
        input_text,
        expected_text,
    )
