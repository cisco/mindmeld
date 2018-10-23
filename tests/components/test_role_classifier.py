
import pytest
import os

from mmworkbench.components import NaturalLanguageProcessor

HOME_ASSISTANT_APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       '../home_assistant')


@pytest.fixture
def home_assistant_app_path():
    return HOME_ASSISTANT_APP_PATH


@pytest.fixture
def home_assistant_nlp(home_assistant_app_path):
    """Provides a built processor instance"""
    nlp = NaturalLanguageProcessor(app_path=home_assistant_app_path)
    nlp.build()
    nlp.dump()
    return nlp


test_data_7 = [
    (
        ['change alarm from 2 pm to 4 pm', 'times_and_dates', 'change_alarm', 'sys_time'],
        ['old_time', 'new_time'],
        ['new_time', 'old_time']
     )
]


@pytest.mark.parametrize("example,role_order_0,role_order_1", test_data_7)
def test_role_classifier(home_assistant_nlp, example, role_order_0, role_order_1):

    intent = home_assistant_nlp.domains[example[1]].intents[example[2]]
    entity_recognizer = intent.entity_recognizer
    role_classifier = intent.entities[example[3]].role_classifier

    entities = entity_recognizer.predict(example[0])
    probs = role_classifier.predict_proba(example[0], entities, 0)
    assert([tup[0] for tup in probs] == role_order_0)

    probs = role_classifier.predict_proba(example[0], entities, 1)
    assert ([tup[0] for tup in probs] == role_order_1)
