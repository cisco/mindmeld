#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_standard_models
----------------------------------

Tests for `standard_models` module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import os

import pytest

from mmworkbench import markup
from mmworkbench.models import ModelConfig, CLASS_LABEL_TYPE, QUERY_EXAMPLE_TYPE
from mmworkbench.models.text_models import TextModel
from mmworkbench.tokenizer import Tokenizer
from mmworkbench.query_factory import QueryFactory
from mmworkbench.resource_loader import ResourceLoader

APP_NAME = 'kwik_e_mart'
APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), APP_NAME)


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
    """A resource loader"""
    return ResourceLoader(APP_PATH, query_factory)


class TestTextModel:

    @classmethod
    def setup_class(cls):
        data_dict = {
            'greet': [
                'Hello',
                'Hello!',
                'hey',
                "what's up",
                'greetings',
                'yo',
                'hi',
                'hey, how are you?',
                'hola',
                'start',
            ],
            'exit': [
                'bye',
                'goodbye',
                'until next time',
                'see ya later',
                'ttyl',
                'talk to you later'
                'later',
                'have a nice day',
                'finish',
                'gotta go'
                "I'm leaving",
                "I'm done",
                "that's all"
            ]
        }
        labeled_data = []
        for intent in data_dict:
            for text in data_dict[intent]:
                labeled_data.append(markup.load_query(text, intent=intent))
        cls.labeled_data = labeled_data

    def test_fit(self, resource_loader):
        """Tests that a basic fit succeeds"""
        config = ModelConfig(**{
            'model_type': 'text',
            'example_type': QUERY_EXAMPLE_TYPE,
            'label_type': CLASS_LABEL_TYPE,
            'model_settings': {
                'classifier_type': 'logreg'
            },
            'params': {
                'fit_intercept': True,
                'C': 100
            },
            'features': {
                'bag-of-words': {
                    'lengths': [1]
                },
                'freq': {'bins': 5},
                'length': {}
            }
        })
        model = TextModel(config)
        examples = [q.query for q in self.labeled_data]
        labels = [q.intent for q in self.labeled_data]
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model._current_params == {'fit_intercept': True, 'C': 100}

    def test_fit_cv(self, resource_loader):
        """Tests fitting with param selection"""
        config = ModelConfig(**{
            'model_type': 'text',
            'example_type': QUERY_EXAMPLE_TYPE,
            'label_type': CLASS_LABEL_TYPE,
            'model_settings': {
                'classifier_type': 'logreg'
            },
            'param_selection': {
                'type': 'k-fold',
                'k': 10,
                'grid': {
                    'C': [10, 100, 1000],
                    'fit_intercept': [True, False]
                },
            },
            'features': {
                'bag-of-words': {
                    'lengths': [1]
                },
                'freq': {'bins': 5},
                'length': {}
            }
        })
        model = TextModel(config)
        examples = [q.query for q in self.labeled_data]
        labels = [q.intent for q in self.labeled_data]
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model._current_params

    def test_fit_predict(self, resource_loader):
        """Tests prediction after a fit"""
        config = ModelConfig(**{
            'model_type': 'text',
            'example_type': QUERY_EXAMPLE_TYPE,
            'label_type': CLASS_LABEL_TYPE,
            'model_settings': {
                'classifier_type': 'logreg'
            },
            'params': {
                'fit_intercept': True,
                'C': 100
            },
            'features': {
                'bag-of-words': {
                    'lengths': [1]
                },
                'freq': {'bins': 5},
                'length': {}
            }
        })
        model = TextModel(config)
        examples = [q.query for q in self.labeled_data]
        labels = [q.intent for q in self.labeled_data]
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query('hi').query]) == 'greet'
        assert model.predict([markup.load_query('bye').query]) == 'exit'

    def test_extract_features(self, resource_loader):
        """Tests extracted features after a fit"""
        config = ModelConfig(**{
            'model_type': 'text',
            'example_type': QUERY_EXAMPLE_TYPE,
            'label_type': CLASS_LABEL_TYPE,
            'model_settings': {
                'classifier_type': 'logreg'
            },
            'params': {
                'fit_intercept': True,
                'C': 100
            },
            'features': {
                'bag-of-words': {
                    'lengths': [1]
                },
            }
        })
        model = TextModel(config)
        examples = [q.query for q in self.labeled_data]
        labels = [q.intent for q in self.labeled_data]
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        expected_features = {'bag_of_words|length:1|ngram:hi': 1,
                             'bag_of_words|length:1|ngram:there': 1}
        extracted_features = model.view_extracted_features(markup.load_query('hi there').query)
        assert extracted_features == expected_features
