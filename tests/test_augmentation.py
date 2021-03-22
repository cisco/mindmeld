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
    config["params"]["fwd_params"]["num_return_sequences"] = NUM_PARAPHRASES
    lang = "en"
    num_augmentations = 10
    resource_loader = ResourceLoader.create_resource_loader(kwik_e_mart_app_path)
    augmentor = create_augmentor(
        config=config,
        lang=lang,
        num_augmentations=num_augmentations,
        resource_loader=resource_loader,
    )
    return augmentor


@pytest.mark.extras
@pytest.mark.parametrize(
    "query",
    [
        ("some text"),
        ("another text"),
        ("yet another text"),
    ],
)
def test_num_paraphrases(english_paraphraser, query):
    paraphrases = english_paraphraser.augment_queries([query])
    assert len(paraphrases) == 10


@pytest.mark.extras
def test_unsupported_language(kwik_e_mart_app_path):
    config = get_augmentation_config(app_path=kwik_e_mart_app_path)
    config["params"]["fwd_params"]["num_return_sequences"] = NUM_PARAPHRASES
    resource_loader = ResourceLoader.create_resource_loader(kwik_e_mart_app_path)

    with pytest.raises(UnsupportedLanguageError):
        create_augmentor(
            config=config,
            lang="de",
            num_augmentations=10,
            resource_loader=resource_loader,
        )
