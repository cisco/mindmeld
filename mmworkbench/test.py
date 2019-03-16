# -*- coding: utf-8 -*-
"""This module contains test utilities for workbench.
"""
from .components.dialogue import Conversation


class TestConversation(Conversation):
    """
    This class is used during testing to assert that we identify the correct domain and intent,
    receive the right text as a response, and navigate to the correct frame.
    """
    def assert_text(self, expected_text, *, text_index=0, history_index=0):
        history_entry = self.history[history_index]
        texts = [self._follow_directive(d) for d in history_entry["directives"]]
        assert texts[text_index] == expected_text

    def assert_domain(self, expected_domain, *, history_index=0):
        history_entry = self.history[history_index]
        assert history_entry['request']['domain'] == expected_domain

    def assert_intent(self, expected_intent, *, history_index=0):
        history_entry = self.history[history_index]
        assert history_entry['request']['intent'] == expected_intent

    def assert_frame(self, frame):
        assert self.frame == frame

