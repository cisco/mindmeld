#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for `token_classification` submodule of nn_utils
"""

import os
import shutil

import pytest

from mindmeld import markup
from mindmeld.models import ENTITY_EXAMPLE_TYPE, ENTITIES_LABEL_TYPE, ModelFactory, ModelConfig
from mindmeld.models.nn_utils.helpers import get_num_weights_of_model
from mindmeld.query_factory import QueryFactory
from mindmeld.resource_loader import ResourceLoader, ProcessedQueryList

APP_NAME = "kwik_e_mart"
APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), APP_NAME
)
GENERATED_TMP_FOLDER = os.path.join(APP_PATH, ".generated/pytorch_module")
QUERY_FACTORY = QueryFactory.create_query_factory(app_path=None, duckling=True)


@pytest.fixture
def resource_loader():
    """A resource loader"""
    return ResourceLoader(app_path=None, query_factory=QUERY_FACTORY)


def model_predictions_assertions(model):
    """Conducts assertions on model predictions; common checks across multiple unittests"""
    predictions = model.predict([
        markup.load_query("Medium Beers pizza from oz pizza",
                          query_factory=QUERY_FACTORY).query])[0]
    assert len(predictions) <= 6
    for prediction in predictions:
        if prediction:  # non entities are predicted as NoneType
            assert prediction.entity.type in {
                'category', 'cuisine', 'dish', 'option', 'restaurant', 'sys_number'
            }


@pytest.mark.extras
@pytest.mark.torch
class TestSequenceClassification:
    @classmethod
    def setup_class(cls):
        examples = [
            "make that {one|sys_number|num_orders} plate of {vegetable lumpia|dish} and "
            "{ten|sys_number|num_orders} orders of {pork sasig|dish}",
            "{lamb curry|dish} with a {side of rice|option} from the {burmese cafe|restaurant}",
            "i'll get a {normal cheese pizza|dish} with {extra cheese|option} and "
            "{extra sauce|option}",
            "What do you have for {dinner|category}?",
            "i want a {half ramen and omusubi|dish} pork",
            "{appetizers|category}",
            "can i get a {regular poki bowl|dish} with {tuna|option} and {yellow tail|option}",
            "I'd like the {Tabouleh|dish} with {feta cheese|option}, {dolmas|dish} "
            "and a {prawn kebab plate|dish} from {saffron 685|restaurant}.",
            "add an order of {tea|dish}",
            "{Half tofu barbeque pork|dish} and a {coconut water|dish}",
            "add an order of {tea|dish} from {Sallys|restaurant}",
            "i feel like having some {Kale Caesar Salad|dish}",
            "i would like to order from {k-oz restaurant|restaurant}",
            "Can I get some {south indian|cuisine} food delivered in 30 min?",
            "{Chorizo con huevos|dish} {without beans|option}",
            "Can I get a {lamb curry|dish} with a side of {brown rice|option} and {raita|option}?",
            "can you recommend some {Japan|cuisine} restaurants which are highly reviewed but not "
            "that expensive",
            "add the {spicy poke sauce|option}",
            "{basil eggplant with chicken and prawns|dish}",
            "{blt|dish} with {egg|option}, make it with {wheat bread|option}",
            "{blackened catfish|dish} with {no sweet potatoes|category}",
            "{1|sys_number|num_orders} order of {samosa|dish}, {1|sys_number|num_orders} "
            "{baingan bartha|dish}, {1|sys_number|num_orders} {chicken achar|dish}, "
            "{2|sys_number|num_orders} {onion naans|dish} and {1|sys_number|num_orders} "
            "{kheema naan|dish}",
            "can i order a {mimosa|dish} {brunch|category}?",
            "get me something from {Dinosaurs Vietnamese Sandwiches|restaurant}",
            "what kind of {fish|option} dishes are there",
            "From {burma cafe|restaurant}, I'll get the {pumpkin curry|dish} with "
            "{coconut rice|option}",
            "Can I add get {3|sys_number|num_orders} {guaranas|dish} to that?",
            "Could you add a {dozen|sys_number|num_orders} {sesame cookies|dish}?",
            "Give me some of that {pork sisig|dish}",
            "Give me some food please",
            "add that",
            "no expensive food please",
        ]
        labeled_data = [markup.load_query(ex, query_factory=QUERY_FACTORY) for ex in examples]
        cls.labeled_data = ProcessedQueryList.from_in_memory_list(labeled_data)

    def test_default_embedder(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {"emb_dim": 5},
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "use_crf_layer": False}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    def test_glove_embedder(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {"embedder_type": "glove", "emb_dim": 5},
        }

        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()

        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

        config = {**config, "params": {"embedder_type": "glove"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "use_crf_layer": False}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    @pytest.mark.transformers
    def test_bpe_embedder(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {  # default token_spans_pooling_type is "first"
                "emb_dim": 30, "tokenizer_type": "bpe-tokenizer", "add_terminals": True
            },
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "token_spans_pooling_type": "mean"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "token_spans_pooling_type": "max"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "token_spans_pooling_type": "mean_sqrt"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "token_spans_pooling_type": "last"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {
            **config["params"], "use_crf_layer": False, "token_spans_pooling_type": "first"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    @pytest.mark.transformers
    def test_wordpiece_embedder(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {
                "emb_dim": 30, "tokenizer_type": "wordpiece-tokenizer", "add_terminals": True
            },
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "use_crf_layer": False}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    def test_char_embedder(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {  # default token_spans_pooling_type is "first"
                "emb_dim": 30, "tokenizer_type": "char-tokenizer"},
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()

        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "add_terminals": "True"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "token_spans_pooling_type": "mean"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "use_crf_layer": False}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    def test_word_lstm(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm-pytorch"},
            "params": {"emb_dim": 30},
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()

        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "add_terminals": "True"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "use_crf_layer": False}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    def test_glove_lstm(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm-pytorch"},
            "params": {"embedder_type": "glove"},
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "use_crf_layer": False}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    @pytest.mark.transformers
    def test_bpe_lstm(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm-pytorch"},
            "params": {  # default token_spans_pooling_type is "first"
                "emb_dim": 30, "tokenizer_type": "bpe-tokenizer", "add_terminals": True
            },
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()

        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "use_crf_layer": False}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "token_spans_pooling_type": "mean"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "token_spans_pooling_type": "max"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "token_spans_pooling_type": "mean_sqrt"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "token_spans_pooling_type": "last"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {
            **config["params"], "use_crf_layer": False, "token_spans_pooling_type": "first"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    @pytest.mark.transformers
    def test_wordpiece_lstm(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm-pytorch"},
            "params": {
                "emb_dim": 30, "tokenizer_type": "wordpiece-tokenizer", "add_terminals": True
            },
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "use_crf_layer": False}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    def test_char_lstm(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm-pytorch"},
            "params": {  # default token_spans_pooling_type is "first"
                "embedder_type": "glove", "emb_dim": 30, "tokenizer_type": "char-tokenizer"},
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()

        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

        config = {**config, "params": {
            "embedder_type": None, "emb_dim": 30, "tokenizer_type": "char-tokenizer"}
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "add_terminals": "True"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "token_spans_pooling_type": "mean"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "use_crf_layer": False}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    def test_char_lstm_word_lstm(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "lstm-lstm"},
            "params": {"emb_dim": 5},
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()

        incorrect_config = {**config, "params": {**config["params"], "add_terminals": True}}
        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**incorrect_config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

        incorrect_config = {
            **config, "params": {**config["params"], "tokenizer_type": "char-tokenizer"}}
        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**incorrect_config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

        incorrect_config = {
            **config, "params": {
                **config["params"], "embedder_type": "bert",
                "pretrained_model_name_or_path": "bert-base-cased"}
        }
        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**incorrect_config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "use_crf_layer": False}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "char_lstm_output_pooling_type": "mean"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        glove_config = {**config, "params": {"embedder_type": "glove"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**glove_config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    def test_char_cnn_word_lstm(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "cnn-lstm"},
            "params": {"emb_dim": 5},
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()

        incorrect_config = {**config, "params": {**config["params"], "add_terminals": True}}
        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**incorrect_config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

        incorrect_config = {
            **config, "params": {**config["params"], "tokenizer_type": "char-tokenizer"}}
        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**incorrect_config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

        incorrect_config = {
            **config, "params": {
                **config["params"], "embedder_type": "bert",
                "pretrained_model_name_or_path": "bert-base-cased"}
        }
        with pytest.raises(ValueError):
            model = ModelFactory.create_model_from_config(ModelConfig(**incorrect_config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)

        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {**config, "params": {**config["params"], "use_crf_layer": False}}
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        glove_config = {**config, "params": {"embedder_type": "glove"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**glove_config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

    @pytest.mark.xfail(strict=False)
    @pytest.mark.transformers
    def test_bert_embedder(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {"embedder_type": "bert"},
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()

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
        model_predictions_assertions(model)

        new_config = {**config, "params": {**config["params"], "token_spans_pooling_type": "mean"}}
        model = ModelFactory.create_model_from_config(ModelConfig(**new_config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        """ test for different pretrained transformers"""

        config = {
            **config,
            "params": {
                "embedder_type": "bert",
                "pretrained_model_name_or_path": "distilbert-base-uncased",
            }
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {
            **config,
            "params": {
                "embedder_type": "bert",
                "pretrained_model_name_or_path": "albert-base-v2",
            }
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {
            **config,
            "params": {
                "embedder_type": "bert",
                "pretrained_model_name_or_path": "sentence-transformers/all-mpnet-base-v2",
            }
        }
        model = ModelFactory.create_model_from_config(ModelConfig(**config))
        model.initialize_resources(resource_loader, examples, labels)
        model.fit(examples, labels)
        model_predictions_assertions(model)

        config = {
            **config,
            "params": {
                "embedder_type": "bert",
                "pretrained_model_name_or_path": "roberta-base",
            }
        }
        with pytest.raises(NotImplementedError):
            model = ModelFactory.create_model_from_config(ModelConfig(**config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)
            model_predictions_assertions(model)

    @pytest.mark.skip(reason="dumping of torch module state dict occupies disk space")
    @pytest.mark.xfail(strict=False)
    @pytest.mark.transformers
    @pytest.mark.bert
    def test_bert_embedder_frozen_params(self, resource_loader):
        """Tests that a fit succeeds"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {  # default embedder_output_pooling_type for bert is "first"
                "embedder_type": "bert",
                "pretrained_model_name_or_path": "distilbert-base-uncased",
                "update_embeddings": False
            },
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()

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
        model_predictions_assertions(model)

    @pytest.mark.xfail(strict=False)
    @pytest.mark.transformers
    def test_bert_embedder_unsupported(self, resource_loader):
        """Tests that a fit succeeds w/ and w/o crf layer"""
        config = {
            "model_type": "tagger",
            "example_type": ENTITY_EXAMPLE_TYPE,
            "label_type": ENTITIES_LABEL_TYPE,
            "model_settings": {"classifier_type": "embedder"},
            "params": {
                "embedder_type": "bert",
                "pretrained_model_name_or_path": "distilroberta-base",
                "add_terminals": True
            },
        }
        examples = self.labeled_data.queries()
        labels = self.labeled_data.entities()

        with pytest.raises(NotImplementedError):
            model = ModelFactory.create_model_from_config(ModelConfig(**config))
            model.initialize_resources(resource_loader, examples, labels)
            model.fit(examples, labels)
            model_predictions_assertions(model)
