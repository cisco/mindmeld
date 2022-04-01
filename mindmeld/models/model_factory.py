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
This module contains the ModelFactory class that can resolve model class names and create/load
appropriate models
"""

import logging
from typing import Union, Type

from .helpers import register_model, ModelType
from .model import ModelConfig, AbstractModel, AbstractModelFactory
from .tagger_models import TaggerModelFactory
from .text_models import TextModelFactory

logger = logging.getLogger(__name__)


class ModelFactory:
    """Auto class that identifies appropriate text/tagger model from text_models.py/tagger_models.py
    to load one based on the inputted configs or from the loaded configs file.

    The .create_model_from_config() methods allows to load the appropriate model when a ModelConfig
    is passed. The .create_model_from_path() method uses AbstractModel's load method to load a
    dumped config, which is then used to load appropriate model and return it through a metadata
    dictionary object.
    """

    def __new__(cls, config: Union[dict, ModelConfig]) -> Type[AbstractModel]:
        # method for backwards compatibility in ./helpers/create_model()
        return cls.create_model_from_config(config)

    @classmethod
    def create_model_from_config(
        cls,
        model_config: Union[dict, ModelConfig]
    ) -> Type[AbstractModel]:
        """
        Instantiates and returns a valid model from the specified model configs

        Args:
            model_config (Union[dict, ModelConfig]): Model configs inputted either as dict or an
                instance of ModelConfig

        Returns:
            model (Type[AbstractModel]): A text/tagger model instance

        Raises:
            ValueError: When the configs are invalid
        """

        is_valid_config = model_config and isinstance(model_config, (ModelConfig, dict))
        if not is_valid_config:
            msg = f"Need a valid model config to create a text/tagger model in ModelFactory. " \
                  f"Found model_config={model_config} of type({type(model_config)})"
            raise ValueError(msg)

        model_config = cls._resolve_model_config(model_config)
        model_type = cls._get_model_type(model_config)
        model_class = cls._get_model_factory(model_type).get_model_cls(model_config)
        return model_class(model_config)

    @classmethod
    def create_model_from_path(cls, path: str) -> Union[None, Type[AbstractModel]]:
        """
        Loads and returns a model from the specified path

        Args:
            path (str): A pickle file path from where a model can be loaded

        Returns:
            model (Union[None, Type[AbstractModel]]): Returns None when the specified path is not
                found or if the model loaded from the specified path is a NoneType. If found a valid
                config and a valid model, the model is load by calling .load() method and returned

        Raises:
            ValueError: When the path is invalid
        """

        if not (path and isinstance(path, str)):
            msg = f"Need a valid path to load a text/tagger model in ModelFactory. " \
                  f"Found path={path} of type({type(path)})"
            raise ValueError(msg)

        if not path.endswith(".pkl"):
            msg = "Model Path must end with .pkl for ModelFactory to be able to identify the model"
            raise ValueError(msg)

        try:
            # if loading from path, determine the ABCModel type & return after doing XxxModel.load()
            model_config = AbstractModel.load_model_config(path)
            model_config = cls._resolve_model_config(model_config)
            model_type = cls._get_model_type(model_config)
            model_class = cls._get_model_factory(model_type).get_model_cls(model_config)
            return model_class.load(path)

        except FileNotFoundError:
            # sometimes a model (and its config file) might not be dumped, eg. in role classifiers
            # or even if dumped, can be of NoneType enclosed in a dictionary
            msg = f"No model file found while trying to load model from path: {path}. It might " \
                  f"be the case that the classifier didn't need a model due to one or no classes."
            logger.warning(msg)
            return None

    @staticmethod
    def _resolve_model_config(model_config: Union[dict, ModelConfig]) -> ModelConfig:
        """
        Resolves and returns model configs.

        Args:
            model_config (Union[dict, ModelConfig]): If inputted as a dict, a new instance of
                ModelConfig is created with that dict and returned.

        Returns:
            model_config (ModelConfig): An instance of ModelConfig

        Raises:
            ValueError: When the model config is of an invalid type
        """

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
    def _get_model_type(model_config: ModelConfig) -> ModelType:
        """
        Returns model type from the model config and validates if it is a valid type or not.

        Args:
            model_config (ModelConfig): An instance of ModelConfig

        Returns:
            model_type (ModelType): The model type obtained from configs

        Raises:
            ValueError: When the model type is invalid
        """

        model_type = model_config.model_type

        try:
            return ModelType(model_type)
        except ValueError as e:
            msg = f"Invalid model configuration: Unknown model type {model_type}. " \
                  f"Known types are: {[v.value for v in ModelType.__members__.values()]}"
            raise ValueError(msg) from e

    @staticmethod
    def _get_model_factory(model_type: ModelType) -> Type[AbstractModelFactory]:
        """
        Returns a factory based on the provided model type

        Args:
            model_type (ModelType): An object of ModelType specifying the type of model to create

        Returns:
            model_factory (Type[AbstractModelFactory]): A model factory for specified model_type
        """

        return {
            ModelType.TEXT_MODEL: TextModelFactory,
            ModelType.TAGGER_MODEL: TaggerModelFactory,
        }[model_type]

    @staticmethod
    def register_models() -> None:
        for model_type in [v.value for v in ModelType.__members__.values()]:
            register_model(model_type, ModelFactory)
        register_model("auto", ModelFactory)


ModelFactory.register_models()
