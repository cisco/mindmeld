#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for `sequence_classification` submodule of nn_utils
"""

import os

import pytest

from mindmeld import markup
from mindmeld.models import QUERY_EXAMPLE_TYPE, CLASS_LABEL_TYPE, ModelFactory, ModelConfig
from mindmeld.resource_loader import ResourceLoader, ProcessedQueryList

APP_NAME = "kwik_e_mart"
APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), APP_NAME
)


@pytest.fixture
def resource_loader(query_factory):
    """A resource loader"""
    return ResourceLoader(APP_PATH, query_factory)


class TestSequenceClassification:
    @classmethod
    def setup_class(cls):
        data_dict = {
            "greet": [
                "Hello",
                "Hello!",
                "hey",
                "what's up",
                "greetings",
                "yo",
                "hi",
                "hey, how are you?",
                "hola",
                "start",
            ],
            "exit": [
                "bye",
                "goodbye",
                "until next time",
                "see ya later",
                "ttyl",
                "talk to you later" "later",
                "have a nice day",
                "finish",
                "gotta go" "I'm leaving",
                "I'm done",
                "that's all",
            ],
        }
        labeled_data = []
        for intent in data_dict:
            for text in data_dict[intent]:
                labeled_data.append(markup.load_query(text, intent=intent))
        cls.labeled_data = ProcessedQueryList.from_in_memory_list(labeled_data)

    @pytest.mark.extras
    @pytest.mark.torch
    def test_default_embedder(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {"emb_dim": 5},
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.torch
    def test_glove_embedder(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {"embedder_type": "glove", "emb_dim": 5},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()

        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

        config = {**config, "params": {"embedder_type": "glove"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.bert
    def test_bert_embedder(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {"embedder_type": "bert"},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()

        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

        config = {
            **config,
            "params": {
                "embedder_type": "bert",
                "pretrained_model_name_or_path": "bert-base-cased",
                "add_terminals": True
            }
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.torch
    def test_char_cnn(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "cnn"},
            "params": {"emb_dim": 30, "tokenizer_type": "char-tokenizer"},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.torch
    @pytest.mark.transformers_tokenizers
    def test_bpe_cnn(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "cnn"},
            "params": {"emb_dim": 30, "tokenizer_type": "bpe-tokenizer", "add_terminals": True},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.torch
    @pytest.mark.transformers_tokenizers
    def test_wordpiece_cnn(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "cnn"},
            "params": {
                "emb_dim": 30, "tokenizer_type": "wordpiece-tokenizer", "add_terminals": True
            },
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.torch
    def test_word_cnn(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "cnn"},
            "params": {"emb_dim": 30},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.torch
    def test_glove_cnn(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "cnn"},
            "params": {"embedder_type": "glove"},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.torch
    def test_bert_cnn(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "cnn"},
            "params": {"embedder_type": "bert"},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()

        # To use a embedder_type 'bert', classifier_type must be 'embedder'.
        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

    @pytest.mark.extras
    @pytest.mark.torch
    def test_char_lstm(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm"},
            "params": {"emb_dim": 30, "tokenizer_type": "char-tokenizer"},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.torch
    @pytest.mark.transformers_tokenizers
    def test_bpe_lstm(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm"},
            "params": {"emb_dim": 30, "tokenizer_type": "bpe-tokenizer", "add_terminals": True},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.torch
    @pytest.mark.transformers_tokenizers
    def test_wordpiece_lstm(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm"},
            "params": {
                "emb_dim": 30, "tokenizer_type": "wordpiece-tokenizer", "add_terminals": True
            },
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.torch
    def test_word_lstm(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm"},
            "params": {"emb_dim": 30},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.extras
    @pytest.mark.torch
    def test_glove_lstm(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm"},
            "params": {"embedder_type": "glove"},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]
