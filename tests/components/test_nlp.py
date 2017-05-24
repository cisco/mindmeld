#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_nlp
----------------------------------

Tests for NaturalLanguageProcessor module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
from __future__ import unicode_literals

import os

import pytest

from mmworkbench.exceptions import ProcessorError
from mmworkbench.components import NaturalLanguageProcessor

APP_NAME = 'kwik_e_mart'
APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), APP_NAME)


@pytest.fixture
def empty_nlp():
    """Provides an empty, unbuilt processor instance"""
    return NaturalLanguageProcessor(APP_PATH)


@pytest.fixture(scope='module')
def nlp():
    """Provides an empty processor instance"""
    nlp = NaturalLanguageProcessor(APP_PATH)
    nlp.build()
    return nlp


def test_instantiate():
    """Tests creating an NLP instance"""
    nlp = NaturalLanguageProcessor(APP_PATH)
    assert nlp


def test_build(empty_nlp):
    """Tests building a processor with default config.

    This is a basic sanity check to make sure there are no exceptions.
    """
    nlp = empty_nlp
    nlp.build()


def test_dump(nlp):
    """Test dump method of nlp"""
    nlp.dump()


def test_early_process(empty_nlp):
    """Tests that attempting to process a message without first loading or
    building models will raise an exception"""
    with pytest.raises(ProcessorError):
        empty_nlp.process('Hello')


@pytest.mark.skip
def test_load(nlp):
    """Tests loading a processor from disk"""
    nlp.load()


def test_process(nlp):
    """Tests a basic call to process"""
    response = nlp.process('Hello')

    assert response == {
        'text': 'Hello',
        'domain': 'store_info',
        'intent': 'greet',
        'entities': []
    }
