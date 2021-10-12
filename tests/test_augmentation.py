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
from mindmeld.markup import load_query

NUM_PARAPHRASES = 10


@pytest.fixture(scope="class")
def english_paraphraser_retain_entities(kwik_e_mart_app_path, request):
    config = get_augmentation_config(app_path=kwik_e_mart_app_path)
    language = "en"
    config['augmentor_class'] = "EnglishParaphraser"
    config['retain_entities'] = True
    query_factory = QueryFactory.create_query_factory(app_path=kwik_e_mart_app_path, duckling=True)
    resource_loader = ResourceLoader.create_resource_loader(app_path=kwik_e_mart_app_path,
                                                            query_factory=query_factory)
    request.cls.query_factory = query_factory
    request.cls.augmentor = AugmentorFactory(
        config=config,
        language=language,
        resource_loader=resource_loader,
    ).create_augmentor()
    yield None
    request.cls.augmentor = None
    request.cls.query_factor = None


@pytest.mark.skip
@pytest.mark.extras
@pytest.mark.usefixtures("english_paraphraser_retain_entities")
class TestEnglishParaphraserWithEntities:
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
    def test_paraphrases_with_entities(self, query, entity_types):
        processed_query = load_query(query, query_factory=self.query_factory)
        paraphrases = self.augmentor.augment_queries([processed_query])
        for p in paraphrases:
            for entity in entity_types:
                assert entity in p


@pytest.fixture(scope="class")
def english_paraphraser(kwik_e_mart_app_path, request):
    config = get_augmentation_config(app_path=kwik_e_mart_app_path)
    config['retain_entities'] = False
    language = "en"
    resource_loader = ResourceLoader.create_resource_loader(kwik_e_mart_app_path)
    request.cls.augmentor = AugmentorFactory(
        config=config,
        language=language,
        resource_loader=resource_loader,
    ).create_augmentor()
    yield None
    request.cls.augmentor = None


@pytest.mark.extras
@pytest.mark.usefixtures("english_paraphraser")
class TestDefaultEnglishParaphraser:
    @pytest.mark.parametrize(
        "query, value",
        [
            ("some text", 10),
            ("another text", 8),
            ("yet another text", 10),
        ],
    )
    def test_num_paraphrases(self, query, value):
        processed_query = load_query(query)
        paraphrases = self.augmentor.augment_queries([processed_query])
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

    paraphrases = multilingual_paraphraser.augment_queries([load_query(query)])
    multilingual_paraphraser=None
    assert "aumentar el volumen" in paraphrases
