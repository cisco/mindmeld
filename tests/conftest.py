# -*- coding: utf-8 -*-

"""
conftest
----------------------------------

Configurations for tests. Include shared fixtures here.
"""
# pylint: disable=locally-disabled,redefined-outer-name
from __future__ import unicode_literals

import os
import warnings

import pytest
import codecs

from mmworkbench.tokenizer import Tokenizer
from mmworkbench.query_factory import QueryFactory
from mmworkbench.resource_loader import ResourceLoader
from mmworkbench.components import NaturalLanguageProcessor, Preprocessor

warnings.filterwarnings("module", category=DeprecationWarning,
                        module="sklearn.preprocessing.label")

APP_NAME = 'kwik_e_mart'
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), APP_NAME)
FOOD_ORDERING_APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'food_ordering')
AENEID_FILE = 'aeneid.txt'
AENEID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), AENEID_FILE)


@pytest.fixture
def lstm_entity_config():
    return {
        'model_type': 'tagger',
        'label_type': 'entities',

        'model_settings': {
            'classifier_type': 'lstm',

            'tag_scheme': 'IOB',
            'feature_scaler': 'max-abs'

        },
        'params': {
            'number_of_epochs': 1
        },
        'features': {
            'in-gaz-span-seq': {},
        }
    }


@pytest.fixture(scope='session')
def kwik_e_mart_app_path():
    return APP_PATH


@pytest.fixture(scope='session')
def food_ordering_app_path():
    return FOOD_ORDERING_APP_PATH


@pytest.fixture(scope='session')
def kwik_e_mart_nlp(kwik_e_mart_app_path):
    """Provides a built processor instance"""
    nlp = NaturalLanguageProcessor(app_path=kwik_e_mart_app_path)
    nlp.build()
    return nlp


class GhostPreprocessor(Preprocessor):
    """A simple preprocessor that removes all instances of the word `ghost`"""
    def process(self, text):
        while 'ghost' in text:
            text = text.replace('ghost', '')
        return text

    def get_char_index_map(self, raw_text, processed_text):
        return {}, {}


@pytest.fixture
def tokenizer():
    """A tokenizer for normalizing text"""
    return Tokenizer()


@pytest.fixture
def preprocessor():
    """A simple preprocessor"""
    return GhostPreprocessor()


@pytest.fixture
def query_factory(tokenizer, preprocessor):
    """For creating queries"""
    return QueryFactory(tokenizer=tokenizer, preprocessor=preprocessor)


@pytest.fixture
def resource_loader(query_factory):
    """A resource loader"""
    return ResourceLoader(APP_PATH, query_factory)


@pytest.fixture
def aeneid_path():
    return AENEID_PATH


@pytest.fixture
def aeneid_content(aeneid_path):
    with codecs.open(aeneid_path, mode='r', encoding='utf-8') as f:
        return f.read()
