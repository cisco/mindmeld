#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test SpacyModelFactory
----------------------------------

Tests for SpacyModelFactory in the `text_preparation.spacy_model_factory` module.
"""
import pytest

from mindmeld.text_preparation.spacy_model_factory import SpacyModelFactory


def test_validate_spacy_language():
    with pytest.raises(ValueError):
        SpacyModelFactory.validate_spacy_language("zz")
    assert SpacyModelFactory.validate_spacy_model_size("lg") is None


@pytest.mark.parametrize(
    "language_code, model_size, expected_model_name",
    [
        ("fr", "lg", "fr_core_news_lg"),
        ("zh", "md", "zh_core_web_md"),
        ("es", "lg", "es_core_news_lg"),
        ("en", "sm", "en_core_web_sm"),
    ],
)
def test_get_spacy_model_name(language_code, model_size, expected_model_name):
    model_name = SpacyModelFactory._get_spacy_model_name(language_code, model_size)
    assert model_name == expected_model_name


def test_import_spacy_model():
    model_name = "en_core_web_sm"
    SpacyModelFactory._download_spacy_model(model_name)
    imported_module = SpacyModelFactory._import_spacy_model(model_name)
    assert imported_module.__name__ == model_name
