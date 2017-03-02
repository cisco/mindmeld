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

from mmworkbench.processor.nlp import NaturalLanguageProcessor

APP_NAME = 'kwik-e-mart'
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), APP_NAME)


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


# def test_load(nlp):
#     nlp.load()


def test_process(nlp):
    response = nlp.process('Hello')

    assert response == {
        'text': 'Hello',
        'domain': 'store_info',
        'intent': 'greet',
        'entities': []
    }
