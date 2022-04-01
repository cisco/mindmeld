#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pytest

from mindmeld.models import (
    ModelFactory,
    create_model,
    CLASS_LABEL_TYPE,
    ENTITIES_LABEL_TYPE,
    QUERY_EXAMPLE_TYPE,
    ENTITY_EXAMPLE_TYPE,
    ModelConfig
)
from mindmeld.models.tagger_models import PytorchTaggerModel
from mindmeld.models.text_models import TextModel, PytorchTextModel


def test_create_model_from_config_dict():
    config = {
        "model_type": "text",
        "example_type": QUERY_EXAMPLE_TYPE,
        "label_type": CLASS_LABEL_TYPE,
        "model_settings": {"classifier_type": "logreg"},
        "params": {"fit_intercept": True, "C": 100},
        "features": {
            "bag-of-words": {"lengths": [1]},
            "freq": {"bins": 5},
            "length": {},
        },
    }
    model = ModelFactory.create_model_from_config(model_config=config)
    assert isinstance(model, TextModel)


@pytest.mark.extras
@pytest.mark.torch
def test_create_model_from_config_object():
    config = {
        "model_type": "text",
        "example_type": QUERY_EXAMPLE_TYPE,
        "label_type": CLASS_LABEL_TYPE,
        "model_settings": {"classifier_type": "lstm"},
    }
    model = ModelFactory.create_model_from_config(model_config=ModelConfig(**config))
    assert isinstance(model, PytorchTextModel)


@pytest.mark.extras
@pytest.mark.torch
def test_create_model_from_helpers():
    config = {
        "model_type": "tagger",
        "example_type": ENTITY_EXAMPLE_TYPE,
        "label_type": ENTITIES_LABEL_TYPE,
        "model_settings": {"classifier_type": "lstm-pytorch"},
        "params": {},
    }
    model = create_model(config=ModelConfig(**config))
    assert isinstance(model, PytorchTaggerModel)


@pytest.mark.extras
@pytest.mark.torch
def test_create_model_from_helpers_without_input_type():
    config = {
        "model_type": "tagger",
        "model_settings": {"classifier_type": "lstm-pytorch"},
        "params": {},
    }
    with pytest.raises(TypeError):
        model = create_model(config=ModelConfig(**config))
        del model


def test_create_model_from_incomplete_config_without_model_settings():
    incomplete_config = {
        "model_type": "text",
        "example_type": QUERY_EXAMPLE_TYPE,
        "label_type": CLASS_LABEL_TYPE,
    }
    with pytest.raises(TypeError):
        model = ModelFactory.create_model_from_config(model_config=incomplete_config)
        del model


@pytest.mark.extras
@pytest.mark.torch
def test_create_model_from_incomplete_config_without_params():
    incomplete_config = {
        "model_type": "tagger",
        "example_type": QUERY_EXAMPLE_TYPE,
        "label_type": CLASS_LABEL_TYPE,
        "model_settings": {"classifier_type": "lstm"},
    }
    with pytest.raises(ValueError):
        model = ModelFactory.create_model_from_config(model_config=incomplete_config)
        del model
