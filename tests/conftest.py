# -*- coding: utf-8 -*-

"""
conftest
----------------------------------

Configurations for tests. Include shared fixtures here.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import asyncio
import codecs
import os
import warnings

import pytest

from mmworkbench.components import NaturalLanguageProcessor, Preprocessor
from mmworkbench.query_factory import QueryFactory
from mmworkbench.resource_loader import ResourceLoader
from mmworkbench.tokenizer import Tokenizer

warnings.filterwarnings("module", category=DeprecationWarning,
                        module="sklearn.preprocessing.label")


APP_NAME = 'kwik_e_mart'
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), APP_NAME)
FOOD_ORDERING_APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'food_ordering')
AENEID_FILE = 'aeneid.txt'
AENEID_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), AENEID_FILE)
HOME_ASSISTANT_APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'home_assistant')


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
def home_assistant_app_path():
    return HOME_ASSISTANT_APP_PATH


@pytest.fixture(scope='session')
def home_assistant_nlp(home_assistant_app_path):
    """Provides a built processor instance"""
    nlp = NaturalLanguageProcessor(app_path=home_assistant_app_path)
    nlp.build()
    nlp.dump()
    return nlp


@pytest.fixture(scope='session')
def kwik_e_mart_nlp(kwik_e_mart_app_path):
    """Provides a built processor instance"""
    nlp = NaturalLanguageProcessor(app_path=kwik_e_mart_app_path)
    nlp.build()
    nlp.dump()
    return nlp


@pytest.fixture(scope='session')
def async_kwik_e_mart_app(kwik_e_mart_nlp):
    from .kwik_e_mart import app_async
    app = app_async.app
    app.lazy_init()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(app.app_manager.load())
    return app


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
