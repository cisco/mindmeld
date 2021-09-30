#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_augmentation
----------------------------------

Tests the ``augmentation`` module.
"""

import pytest

from mindmeld.augmentation import AugmentorFactory, UnsupportedLanguageError
from mindmeld.components._config import get_augmentation_config
from mindmeld.resource_loader import ResourceLoader
from mindmeld.query_factory import QueryFactory

NUM_PARAPHRASES = 10


@pytest.fixture(scope="module")
def english_paraphraser_retain_entities(kwik_e_mart_app_path):
    config = get_augmentation_config(app_path=kwik_e_mart_app_path)
    language = "en"
    config['augmentor_class'] = "EnglishParaphraser"
    config['retain_entities'] = True
    query_factory = QueryFactory.create_query_factory(app_path=kwik_e_mart_app_path, duckling=True)
    resource_loader = ResourceLoader.create_resource_loader(app_path=kwik_e_mart_app_path,
                                                            query_factory=query_factory)
    augmentor = AugmentorFactory(
        config=config,
        language=language,
        resource_loader=resource_loader,
    ).create_augmentor()
    return augmentor


@pytest.mark.extras
@pytest.mark.parametrize(
    "query, entity_types",
    [
        ("some text that contains no entities", [],),
        (
            "can you tell me if {springfield|store_name} is possibly open at this time on {friday|sys_time}",
            ['store_name', 'sys_time'],
        ),
        (
            "Open the {china town|store_name} at {1 pm|sys_time|opening_time}",
            ['store_name', 'sys_time|opening_time'],
        ),
    ],
)
def test_paraphrases_with_entities(english_paraphraser_retain_entities, query, entity_types):
    paraphrases = english_paraphraser_retain_entities.augment_queries([query])
    for p in paraphrases:
        for entity in entity_types:
            assert entity in p


@pytest.fixture(scope="module")
def english_paraphraser(kwik_e_mart_app_path):
    config = get_augmentation_config(app_path=kwik_e_mart_app_path)
    config['retain_entities'] = False
    language = "en"
    resource_loader = ResourceLoader.create_resource_loader(kwik_e_mart_app_path)
    augmentor = AugmentorFactory(
        config=config,
        language=language,
        resource_loader=resource_loader,
    ).create_augmentor()
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
        AugmentorFactory(
            config=config,
            language="de",
            resource_loader=resource_loader,
        ).create_augmentor()


@pytest.mark.skip
@pytest.mark.extras
@pytest.mark.parametrize(
    "query",
    [
        ("aumenta el volumen"),
    ],
)
def test_spanish_paraphrases(kwik_e_mart_app_path, query):
    config = get_augmentation_config(app_path=kwik_e_mart_app_path)
    config["augmentor_class"] = "MultiLingualParaphraser"
    language = "es"
    resource_loader = ResourceLoader.create_resource_loader(kwik_e_mart_app_path)
    multilingual_paraphraser = AugmentorFactory(
        config=config,
        language=language,
        resource_loader=resource_loader,
    ).create_augmentor()

    paraphrases = multilingual_paraphraser.augment_queries([query])
    assert "aumentar el volumen" in paraphrases
