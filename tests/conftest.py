# -*- coding: utf-8 -*-

"""
conftest
----------------------------------

Configurations for tests. Include shared fixtures here.
"""
# pylint: disable=locally-disabled,redefined-outer-name
from __future__ import unicode_literals

import os

import pytest

from mmworkbench.tokenizer import Tokenizer
from mmworkbench.query_factory import QueryFactory
from mmworkbench.resource_loader import ResourceLoader
from mmworkbench.components import Preprocessor

APP_NAME = 'kwik_e_mart'
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), APP_NAME)


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
