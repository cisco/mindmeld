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

# Need to import the following so models and features are registered. See helpers module
from . import (
    entity_features,
    query_features,
    tagger_models,
    text_models,
    embedder_models,
)
from .helpers import (
    CLASS_LABEL_TYPE,
    ENTITIES_LABEL_TYPE,
    ENTITY_EXAMPLE_TYPE,
    QUERY_EXAMPLE_TYPE,
    create_model,
    create_embedder_model,
    register_embedder,
)
from .model import ModelConfig
from .embedder_models import Embedder

__all__ = [
    "ModelConfig",
    "text_models",
    "tagger_models",
    "embedder_models",
    "query_features",
    "entity_features",
    "create_model",
    "QUERY_EXAMPLE_TYPE",
    "ENTITY_EXAMPLE_TYPE",
    "CLASS_LABEL_TYPE",
    "ENTITIES_LABEL_TYPE",
    "create_embedder_model",
    "Embedder",
    "register_embedder",
]
