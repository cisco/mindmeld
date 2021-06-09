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
from .model import ModelConfig, AbstractModel
from .tagger_models import AutoTaggerModel
from .text_models import AutoTextModel

logger = logging.getLogger(__name__)


class AutoModel:
    KNOWN_MODELS_TYPES = ["text", "tagger"]

    def __new__(cls, config_or_path: Union[str, dict, ModelConfig]):

        if not config_or_path:
            msg = f"Need a valid model config or model path to create/load a text/tagger model " \
                  f"in AutoModel. Found config_or_path={config_or_path} "
            raise ValueError(msg)

        if isinstance(config_or_path, (ModelConfig, dict)):
            model_config = config_or_path

            # get model type upon validation
            model_type = cls._get_model_type(model_config)

            # create the model and return it
            if model_type == "text":
                model = AutoTextModel.from_config(model_config)
            elif model_type == "tagger":
                model = AutoTaggerModel.from_config(model_config)

            return model

        # if loading from a path, determine the xxxModel type and return xxxModel.load()
        if isinstance(config_or_path, str):
            path = str(config_or_path)

            if not path.endswith(".pkl"):
                msg = "Model Path must end with .pkl for AutoModel to be able to identify the model"
                raise ValueError(msg)

            if not os.path.exists(path):
                msg = f"Inputted path '{path}' is not found while trying to create model in " \
                      f"AutoModel. Make sure the inputted path is a valid pickle file " \
                      f"path ending with .pkl"
                raise ValueError(msg)

            metadata = AbstractModel.load(path)
            model_config = metadata["model_config"]

            # get model type upon validation
            model_type = cls._get_model_type(model_config)

            # load metadata and return
            if model_type == "text":
                model_class = AutoTextModel.get_model_class(model_config)
            elif model_type == "tagger":
                model_class = AutoTaggerModel.get_model_class(model_config)

            return model_class.load(path)

    @staticmethod
    def _get_model_type(model_config: Union[dict, ModelConfig]):

        # format configs
        if isinstance(model_config, dict):
            model_config = ModelConfig(**model_config)

        # validate configs
        if not isinstance(model_config, ModelConfig):
            msg = f"Expected input config to be either a valid dictionary or an instance of " \
                  f"ModelConfig class, but found of type {type(model_config)}"
            raise ValueError(msg)

        # identify the model_type
        try:
            model_type = model_config.model_type
        except KeyError as e:
            msg = f"Invalid model configuration: Unknown model type {model_type}. " \
                  f"Known types are: {AutoModel.KNOWN_MODELS_TYPES}"
            raise ValueError(msg) from e

        return model_type

    @classmethod
    def from_path(cls, path: str):
        return cls(path)

    @classmethod
    def from_config(cls, config: Union[dict, ModelConfig]):
        return cls(config)

    @classmethod
    def register_models(cls):
        for model_type in AutoModel.KNOWN_MODELS_TYPES:
            register_model(model_type, AutoModel)
        register_model("auto", AutoModel)


AutoModel.register_models()
