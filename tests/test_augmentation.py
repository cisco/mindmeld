#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_augmentation
----------------------------------

Tests the ``augmentation`` module.
"""

import pytest

from mindmeld.components._config import get_augmentation_config
from mindmeld.models.helpers import create_augmentor
from mindmeld.resource_loader import ResourceLoader

NUM_PARAPHRASES = 10


@pytest.fixture(scope="module")
def english_paraphraser(kwik_e_mart_app_path):
    config = get_augmentation_config(app_path=kwik_e_mart_app_path)
    config['params']['num_return_sequences'] = NUM_PARAPHRASES
    lang = 'en'
    resource_loader = ResourceLoader.create_resource_loader(kwik_e_mart_app_path)
    augmentor = create_augmentor(
        lang=lang, config=config, resource_loader=resource_loader
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
    paraphrases = english_paraphraser.augment_queries(
        [query]
    )
    assert len(paraphrases) == 10
