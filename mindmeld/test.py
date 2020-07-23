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

"""This module contains test utilities for MindMeld.
"""
from .components.dialogue import Conversation


class ConversationTestHelper(Conversation):
    """
    This class is used during testing to assert that we identify the correct domain and intent,
    receive the right text as a response, and navigate to the correct frame.
    """

    def assert_text(self, expected_text, *, text_index=0, history_index=0):
        history_entry = self.history[history_index]
        texts = [self._follow_directive(d) for d in history_entry["directives"]]
        if isinstance(expected_text, list):
            assert texts[text_index] in expected_text
        else:
            assert texts[text_index] == expected_text

    def assert_domain(self, expected_domain, *, history_index=0):
        history_entry = self.history[history_index]
        assert history_entry["request"]["domain"] == expected_domain

    def assert_intent(self, expected_intent, *, history_index=0):
        history_entry = self.history[history_index]
        assert history_entry["request"]["intent"] == expected_intent

    def assert_frame(self, frame):
        assert self.frame == frame
