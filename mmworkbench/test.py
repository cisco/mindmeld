import os
from .path import get_current_app_path
from .components import NaturalLanguageProcessor
from .components.dialogue import Conversation


class ConversationTest(object):
    LASTEST_HISTORY_INDEX = 0

    @classmethod
    def setup_class(cls):
        app_path = get_current_app_path(os.path.dirname(os.path.realpath(__file__)))
        nlp = NaturalLanguageProcessor(app_path=app_path)
        nlp.load()
        cls.conv = Conversation(nlp=nlp, app_path=app_path)

    @staticmethod
    def assert_text(texts, expected_text, index=0):
        assert texts[index] == expected_text

    @staticmethod
    def assert_intent(conv, expected_intent):
        last_history = conv.history[ConversationTest.LASTEST_HISTORY_INDEX]
        assert last_history['intent'] == expected_intent

    def setup_method(self, method):
        self.conv.reset()

    def say(self, text):
        return self.conv.say(text)
