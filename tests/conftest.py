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
from mmworkbench.components.resource_loader import ResourceLoader

APP_NAME = 'kwik-e-mart'
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), APP_NAME)


@pytest.fixture
def tokenizer():
    """A tokenizer for normalizing text"""
    return Tokenizer()


@pytest.fixture
def query_factory(tokenizer):
    """For creating queries"""
    return QueryFactory(tokenizer)


@pytest.fixture
def resource_loader(query_factory):
    return ResourceLoader(APP_PATH, query_factory)
