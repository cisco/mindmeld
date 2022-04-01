#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for `sequence_classification` submodule of nn_utils
"""

import os
import shutil

import pytest

from mindmeld import markup
from mindmeld.models import QUERY_EXAMPLE_TYPE, CLASS_LABEL_TYPE, ModelFactory, ModelConfig
from mindmeld.models.nn_utils.helpers import get_num_weights_of_model
from mindmeld.resource_loader import ResourceLoader, ProcessedQueryList

APP_NAME = "kwik_e_mart"
APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), APP_NAME
)
GENERATED_TMP_FOLDER = os.path.join(APP_PATH, ".generated/pytorch_module")


@pytest.fixture
def resource_loader(query_factory):
    """A resource loader"""
    return ResourceLoader(APP_PATH, query_factory)


@pytest.mark.extras
@pytest.mark.torch
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
                "talk to you later later",
                "have a nice day",
                "finish",
                "gotta go I'm leaving",
                "I'm done",
                "that's all",
            ],
        }
        labeled_data = []
        for intent in data_dict:
            for text in data_dict[intent]:
                labeled_data.append(markup.load_query(text, intent=intent))
        cls.labeled_data = ProcessedQueryList.from_in_memory_list(labeled_data)

    def test_default_embedder(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {"emb_dim": 5},  # default embedder_output_pooling_type is "mean"
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()

        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "first"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "last"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    def test_glove_embedder(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {  # default embedder_output_pooling_type is "mean"
                "embedder_type": "glove", "emb_dim": 5
            },
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

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "max"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.transformers
    def test_bpe_embedder(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {  # default embedder_output_pooling_type is "mean"
                "emb_dim": 30, "tokenizer_type": "bpe-tokenizer", "add_terminals": True
            },
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()

        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "first"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "last"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "max"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {
            **config["params"], "embedder_output_pooling_type": "mean_sqrt"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.transformers
    def test_wordpiece_embedder(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {  # default embedder_output_pooling_type is "mean"
                "emb_dim": 30, "tokenizer_type": "wordpiece-tokenizer", "add_terminals": True
            },
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()

        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "first"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "last"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

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

    @pytest.mark.transformers
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

    @pytest.mark.transformers
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

    @pytest.mark.transformers
    def test_bpe_lstm(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm"},
            "params": {  # default lstm_output_pooling_type is "last"
                "emb_dim": 30, "tokenizer_type": "bpe-tokenizer", "add_terminals": True
            },
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()

        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "lstm_output_pooling_type": "first"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "lstm_output_pooling_type": "mean"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "lstm_output_pooling_type": "max"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {
            **config["params"], "lstm_output_pooling_type": "mean_sqrt"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.transformers
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

        config = {**config, "params": {**config["params"], "add_terminals": "True"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.xfail(strict=False)
    @pytest.mark.transformers
    @pytest.mark.bert
    def test_bert_embedder(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {  # default embedder_output_pooling_type for bert is "first"
                "embedder_type": "bert"
            },
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()

        """ test different configurations for bert-base-cased model"""

        config = {**config, "params": {
            **config["params"],
            "pretrained_model_name_or_path": "bert-base-cased"
        }}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "mean"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "last"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "max"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {
            **config["params"], "embedder_output_pooling_type": "mean_sqrt"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        """ test for different pretrained transformers"""

        config = {
            **config,
            "params": {"pretrained_model_name_or_path": "distilbert-base-uncased", }
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {
            **config,
            "params": {"pretrained_model_name_or_path": "roberta-base"}
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {
            **config,
            "params": {"pretrained_model_name_or_path": "albert-base-v2"}
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {
            **config,
            "params": {"pretrained_model_name_or_path": "sentence-transformers/all-mpnet-base-v2"}
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

        config = {**config, "params": {**config["params"], "embedder_output_pooling_type": "mean"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    @pytest.mark.skip(reason="dumping of torch module state dict occupies disk space")
    @pytest.mark.xfail(strict=False)
    @pytest.mark.transformers
    @pytest.mark.bert
    def test_bert_embedder_frozen_params(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {  # default embedder_output_pooling_type for bert is "first"
                "embedder_type": "bert",
                "pretrained_model_name_or_path": "distilbert-base-uncased",
                "embedder_output_pooling_type": "mean",
                "update_embeddings": False
            },
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()

        # fit the model
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)

        # assert only some weights are trainable
        clf = model._clf
        n_requires_grad, n_total = get_num_weights_of_model(clf)
        assert n_requires_grad < n_total, print(n_requires_grad, n_total)

        # check if dumping and loading partial state dict logs required messages & throws no errors
        os.makedirs(GENERATED_TMP_FOLDER, exist_ok=True)
        clf.dump(GENERATED_TMP_FOLDER)
        new_clf = clf.load(GENERATED_TMP_FOLDER)
        shutil.rmtree(GENERATED_TMP_FOLDER)

        # do predictions with loaded model
        model._clf = new_clf
        assert model.predict([markup.load_query("hi").query])[0] in ["greet", "exit"]

    def test_bert_cnn(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "cnn"},
            "params": {"embedder_type": "bert", "pretrained_model_name_or_path": "bert-base-cased"},
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()

        # To use a embedder_type 'bert', classifier_type must be 'embedder'.
        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

    def test_bert_lstm(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "text",
            "example_type": QUERY_EXAMPLE_TYPE,
            "label_type": CLASS_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm"},
            "params": {"embedder_type": "bert", "pretrained_model_name_or_path": "bert-base-cased"},
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.intents()

        # To use a embedder_type 'bert', classifier_type must be 'embedder'.
        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)
