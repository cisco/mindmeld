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

from .classification import EMBEDDER_TYPE_TO_ALLOWED_TOKENIZER_TYPES
from .sequence_classification import (
    EmbedderForSequenceClassification,
    CnnForSequenceClassification,
    LstmForSequenceClassification,
    BertForSequenceClassification
)
from .token_classification import (
    EmbedderForTokenClassification,
    LstmForTokenClassification,
    CharCnnWithWordLstmForTokenClassification,
    CharLstmWithWordLstmForTokenClassification,
    BertForTokenClassification
)

ALLOWED_EMBEDDER_TYPES = [None, *EMBEDDER_TYPE_TO_ALLOWED_TOKENIZER_TYPES]

__all__ = [
    "ALLOWED_EMBEDDER_TYPES",
    "EmbedderForSequenceClassification",
    "CnnForSequenceClassification",
    "LstmForSequenceClassification",
    "BertForSequenceClassification",
    "EmbedderForTokenClassification",
    "LstmForTokenClassification",
    "CharCnnWithWordLstmForTokenClassification",
    "CharLstmWithWordLstmForTokenClassification",
    "BertForTokenClassification",
]
