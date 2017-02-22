#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_nlp
----------------------------------

Tests for NaturalLanguageProcessor module.
"""
from __future__ import unicode_literals

import os

import pytest


from mmworkbench.processor.nlp import NaturalLanguageProcessor

APP_NAME = 'kwik-e-mart'
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), APP_NAME)


@pytest.fixture
def empty_nlp():
    return NaturalLanguageProcessor(APP_PATH)


@pytest.fixture(scope='module')
def nlp():
    nlp = NaturalLanguageProcessor(APP_PATH)
    nlp.build()
    return nlp


def test_instantiate():
    nlp = NaturalLanguageProcessor(APP_PATH)
    assert nlp


def test_build(empty_nlp):
    # Basic sanity check to make sure there are no exceptions
    nlp = empty_nlp
    nlp.build()


def test_dump(nlp):
    nlp.dump()


# def test_load(nlp):
#     nlp.load()


def test_process(nlp):
    response = nlp.process('Hello')

    assert response == {
        'query_text': 'Hello',
        'domain': 'store_info',
        'intent': 'greet',
        'entities': []
    }
