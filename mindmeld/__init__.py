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

try:
    # register test module for assertion rewrite
    import pytest

    pytest.register_assert_rewrite("mindmeld.test")
except ImportError:
    pass  # no worries

from ._util import blueprint, configure_logs
from ._version import current
from .app import Application
from .components import Conversation, DialogueResponder, NaturalLanguageProcessor, QuestionAnswerer

__all__ = [
    "blueprint",
    "configure_logs",
    "Application",
    "Conversation",
    "DialogueResponder",
    "NaturalLanguageProcessor",
    "QuestionAnswerer",
]

__author__ = "Cisco Systems, Inc."
__email__ = "mindmeld@cisco.com"
__version__ = current
