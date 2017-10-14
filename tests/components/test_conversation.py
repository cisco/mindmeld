import os
import pytest

from mmworkbench.components.nlp import NaturalLanguageProcessor
from mmworkbench.components.dialogue import Conversation

APP_NAME = 'kwik_e_mart'
APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), APP_NAME)


@pytest.fixture
def setup_class():
    nlp = NaturalLanguageProcessor(app_path=APP_PATH)
    nlp.build()
    nlp.dump()
    nlp.load()
    return nlp, Conversation(nlp=nlp, app_path=APP_PATH)


def test_params_are_cleared():
    """Tests that the params are cleared in one trip from app to wb."""
    _, conv = setup_class()

    conv.params = {
        'allowed_intents': ['store_info.find_nearest_store'],
        'target_dialogue_state': 'greeting'
    }
    conv.say('close door')

    assert conv.params == {}
