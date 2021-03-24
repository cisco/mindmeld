#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_augmentation
----------------------------------

Tests the ``augmentation`` module.
"""

import pytest

from mindmeld.augmentation import UnsupportedLanguageError
from mindmeld.components._config import get_augmentation_config
from mindmeld.models.helpers import create_augmentor
from mindmeld.resource_loader import ResourceLoader

NUM_PARAPHRASES = 10


@pytest.fixture(scope="module")
def english_paraphraser(kwik_e_mart_app_path):
    config = get_augmentation_config(app_path=kwik_e_mart_app_path)
    language = "en"
    resource_loader = ResourceLoader.create_resource_loader(kwik_e_mart_app_path)
    augmentor = create_augmentor(
        config=config,
        language=language,
        resource_loader=resource_loader,
    )
    return augmentor


@pytest.mark.extras
@pytest.mark.parametrize(
    "query, value",
    [
        ("some text", 10),
        ("another text", 8),
        ("yet another text", 10),
    ],
)
def test_num_paraphrases(english_paraphraser, query, value):
    paraphrases = english_paraphraser.augment_queries([query])
    assert len(paraphrases) == value


@pytest.mark.extras
def test_unsupported_language(kwik_e_mart_app_path):
    config = get_augmentation_config(app_path=kwik_e_mart_app_path)
    resource_loader = ResourceLoader.create_resource_loader(kwik_e_mart_app_path)

    with pytest.raises(UnsupportedLanguageError):
        create_augmentor(
            config=config,
            language="de",
            resource_loader=resource_loader,
        )


@pytest.fixture(scope="module")
def multilingual_paraphraser(kwik_e_mart_app_path):
    config = get_augmentation_config(app_path=kwik_e_mart_app_path)
    config["augmentor_class"] = "MultiLingualParaphraser"
    language = "es"
    resource_loader = ResourceLoader.create_resource_loader(kwik_e_mart_app_path)
    augmentor = create_augmentor(
        config=config,
        language=language,
        resource_loader=resource_loader,
    )
    return augmentor


@pytest.mark.extras
@pytest.mark.parametrize(
    "query",
    [
        ("aumenta el volumen"),
    ],
)
def test_spanish_paraphrases(multilingual_paraphraser, query):
    paraphrases = multilingual_paraphraser.augment_queries([query])
    assert "aumentar el volumen" in paraphrases
