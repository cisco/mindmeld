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

# Need to import the following so models, label encoders, features are registered.
# See helpers module
from . import (
    model_factory,
    embedder_models,
    labels
)
from .embedder_models import Embedder
from .features import (
    entity_features,
    query_features
)
from .helpers import (
    CLASS_LABEL_TYPE,
    ENTITIES_LABEL_TYPE,
    ENTITY_EXAMPLE_TYPE,
    QUERY_EXAMPLE_TYPE,
    create_model,
    load_model,
    create_embedder_model,
    register_embedder,
)
from .model import ModelConfig
from .model_factory import ModelFactory

__all__ = [
    "model_factory",
    "embedder_models",
    "labels",
    "ModelFactory",
    "Embedder",
    "query_features",
    "entity_features",
    "CLASS_LABEL_TYPE",
    "ENTITIES_LABEL_TYPE",
    "ENTITY_EXAMPLE_TYPE",
    "QUERY_EXAMPLE_TYPE",
    "create_model",
    "load_model",
    "create_embedder_model",
    "register_embedder",
    "ModelConfig",
]
