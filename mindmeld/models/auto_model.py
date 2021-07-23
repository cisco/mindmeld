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
from typing import Union, Type

from .helpers import register_model
from .model import ModelConfig, AbstractModel
from .tagger_models import AutoTaggerModel
from .text_models import AutoTextModel

logger = logging.getLogger(__name__)


class AutoModel:
    """Auto class that identifies appropriate text/tagger model from text_models.py/tagger_models.py
    to load one based on the inputted configs or from the loaded configs file.

    The .from_config() methods allows to load the appropriate model when a ModelConfig is passed.
    The .from_path() method uses AbstractModel's load method to load a dumped config, which is then
    used to load appropriate model and return it through a metadata dictionary object.

    """
    ALLOWED_MODEL_TYPES = ["text", "tagger"]

    def __new__(cls, config: Union[dict, ModelConfig]) -> Type[AbstractModel]:
        # method for backwards compatability in ./helpers/create_model()
        return cls.from_config(config)

    @classmethod
    def from_config(cls, model_config: Union[dict, ModelConfig]) -> Type[AbstractModel]:
        """
        Loads a valid model from the specified model configs
        """

        if not (model_config and isinstance(model_config, (ModelConfig, dict))):
            msg = f"Need a valid model config to create a text/tagger model in AutoModel. " \
                  f"Found model_config={model_config} of type({type(model_config)})"
            raise ValueError(msg)

        # get model type upon validation
        model_config = cls._resolve_model_config(model_config)
        model_type = cls._get_model_type(model_config)

        # load metadata and return
        if model_type == "text":
            model_class = AutoTextModel.get_model_class(model_config)
        elif model_type == "tagger":
            model_class = AutoTaggerModel.get_model_class(model_config)

        return model_class(model_config)

    @classmethod
    def from_path(cls, path: str) -> Union[None, Type[AbstractModel]]:
        """
        Loads a valid model from the specified path wherein the model was previously dumped.
        Returns None when the specified path is not found or if the model loaded from the
        specified path is a NoneType.
        """

        if not (path and isinstance(path, str)):
            msg = f"Need a valid path to load a text/tagger model in AutoModel. " \
                  f"Found path={path} of type({type(path)})"
            raise ValueError(msg)

        if not path.endswith(".pkl"):
            msg = "Model Path must end with .pkl for AutoModel to be able to identify the model"
            raise ValueError(msg)

        try:
            # if loading from path, determine the ABCModel type & return after doing xxxModel.load()
            model_config = AbstractModel.load_model_config(path)

            # get model type upon validation
            model_config = cls._resolve_model_config(model_config)
            model_type = cls._get_model_type(model_config)

            # load metadata and return
            if model_type == "text":
                model_class = AutoTextModel.get_model_class(model_config)
            elif model_type == "tagger":
                model_class = AutoTaggerModel.get_model_class(model_config)

            return model_class.load(path)

        except FileNotFoundError:
            # sometimes a model (and its config file) might not be dumped, eg. in role classifiers
            # or even if dumped, can be of NoneType enclosed in a dictionary
            return None

    @staticmethod
    def _resolve_model_config(model_config: Union[dict, ModelConfig]) -> ModelConfig:

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
    def _get_model_type(model_config: ModelConfig) -> str:

        # identify the model_type
        try:
            model_type = model_config.model_type
            assert model_type in AutoModel.ALLOWED_MODEL_TYPES
        except (KeyError, AssertionError) as e:
            msg = f"Invalid model configuration: Unknown model type {model_type}. " \
                  f"Known types are: {AutoModel.ALLOWED_MODEL_TYPES}"
            raise ValueError(msg) from e

        return model_type

    @staticmethod
    def register_models() -> None:
        for model_type in AutoModel.ALLOWED_MODEL_TYPES:
            register_model(model_type, AutoModel)
        register_model("auto", AutoModel)


AutoModel.register_models()
