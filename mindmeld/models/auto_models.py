# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Contains classes that can resolve model types and create/load appropriate models
"""

import logging
import os
from typing import Union

from .helpers import register_model
from .model import ModelConfig, BaseModel
from .tagger_models import AutoTaggerModel
from .text_models import AutoTextModel

logger = logging.getLogger(__name__)


class AutoModel:
    """Auto class that identifies appropriate text/tagger model from text_models.py/tagger_models.py
    to load based on the inputted configs or from the loaded configs file.

    The .from_config() methods allows to load the appropriate model when passed in with ModelConfig.
    The .from_path() method uses BaseModel's load method to load a dumped config, which is them used
    to load appropriate model and return it through a metadata dictionary object.

    """
    ALLOWED_MODEL_TYPES = ["text", "tagger"]

    # method kept for backwards compatability
    def __new__(cls, config: Union[dict, ModelConfig]):
        return cls.from_config(config)

    @classmethod
    def from_config(cls, model_config: Union[dict, ModelConfig]):

        if not (model_config and isinstance(model_config, (ModelConfig, dict))):
            msg = f"Need a valid model config to create a text/tagger model in AutoModel. " \
                  f"Found model_config={model_config} of type({type(model_config)})"
            raise ValueError(msg)

        # get model type upon validation
        model_config = cls._get_model_config(model_config)
        model_type = cls._get_model_type(model_config)

        # load metadata and return
        if model_type == "text":
            model_class = AutoTextModel.get_model_class(model_config)
        elif model_type == "tagger":
            model_class = AutoTaggerModel.get_model_class(model_config)

        return model_class(model_config)

    @classmethod
    def from_path(cls, path: str):

        if not (path and isinstance(path, str)):
            msg = f"Need a valid path to load a text/tagger model in AutoModel. " \
                  f"Found path={path} of type({type(path)})"
            raise ValueError(msg)

        if not path.endswith(".pkl"):
            msg = "Model Path must end with .pkl for AutoModel to be able to identify the model"
            raise ValueError(msg)

        if not os.path.exists(path):
            msg = f"Inputted path '{path}' is not found while trying to load model in AutoModel. " \
                  f"Make sure the inputted path is a valid pickle file path ending with .pkl"
            raise ValueError(msg)

        # if loading from a path, determine the xxxModel type and return after xxxModel.load()
        metadata = BaseModel.load(path)
        model_config = metadata["model_config"]

        # get model type upon validation
        model_config = cls._get_model_config(model_config)
        model_type = cls._get_model_type(model_config)

        # load metadata and return
        if model_type == "text":
            model_class = AutoTextModel.get_model_class(model_config)
        elif model_type == "tagger":
            model_class = AutoTaggerModel.get_model_class(model_config)

        return model_class.load(path)

    @staticmethod
    def _get_model_config(model_config: Union[dict, ModelConfig]):

        # format configs
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)

        # validate configs
        if not isinstance(model_config, ModelConfig):
            msg = f"Expected input config to be either a valid dictionary or an instance of " \
                  f"ModelConfig class, but found of type {type(model_config)}"
            raise ValueError(msg)

        return model_config

    @staticmethod
    def _get_model_type(model_config: ModelConfig):

        # identify the model_type
        try:
            model_type = model_config.model_type
        except KeyError as e:
            msg = f"Invalid model configuration: Unknown model type {model_type}. " \
                  f"Known types are: {AutoModel.ALLOWED_MODEL_TYPES}"
            raise ValueError(msg) from e

        return model_type

    @classmethod
    def register_models(cls):
        for model_type in AutoModel.ALLOWED_MODEL_TYPES:
            register_model(model_type, AutoModel)
        register_model("auto", AutoModel)


AutoModel.register_models()
