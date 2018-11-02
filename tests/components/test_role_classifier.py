
import pytest
import os
import shutil

from mmworkbench.path import MODEL_CACHE_PATH, get_role_model_paths, get_entity_model_paths
from mmworkbench.exceptions import FileNotFoundError
from mmworkbench.components import NaturalLanguageProcessor

HOME_ASSISTANT_APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                       'home_assistant')


@pytest.fixture
def home_assistant_nlp():
    """Provides a built processor instance"""
    nlp = NaturalLanguageProcessor(app_path=HOME_ASSISTANT_APP_PATH)
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


def test_model_accuracies_are_similar_before_and_after_caching():
    # clear model cache
    model_cache_path = MODEL_CACHE_PATH.format(app_path=HOME_ASSISTANT_APP_PATH)
    try:
        shutil.rmtree(MODEL_CACHE_PATH.format(app_path=HOME_ASSISTANT_APP_PATH))
    except FileNotFoundError:
        pass

    # Make sure no cache exists
    assert os.path.exists(model_cache_path) is False
    nlp = NaturalLanguageProcessor(HOME_ASSISTANT_APP_PATH)
    nlp.build()

    entity_eval = \
        nlp.domains['times_and_dates'].intents['change_alarm'].entity_recognizer.evaluate()
    role_eval = \
        nlp.domains[
            'times_and_dates'].intents[
            'change_alarm'].entities['sys_time'].role_classifier.evaluate()
    entity_accuracy_no_cache = entity_eval.get_accuracy()
    role_accuracy_no_cache = role_eval.get_accuracy()

    # dump cache and use it's data
    nlp = NaturalLanguageProcessor(HOME_ASSISTANT_APP_PATH)
    nlp.build(incremental=True)
    nlp.dump()
    nlp = NaturalLanguageProcessor(HOME_ASSISTANT_APP_PATH)
    nlp.build(incremental=True)

    # make sure cache exists
    assert os.path.exists(model_cache_path) is True

    entity_eval = \
        nlp.domains[
            'times_and_dates'].intents[
            'change_alarm'].entity_recognizer.evaluate()
    role_eval = \
        nlp.domains[
            'times_and_dates'].intents[
            'change_alarm'].entities['sys_time'].role_classifier.evaluate()
    entity_accuracy_cached = entity_eval.get_accuracy()
    role_accuracy_cached = role_eval.get_accuracy()

    assert role_accuracy_no_cache == role_accuracy_cached
    assert entity_accuracy_no_cache == entity_accuracy_cached


def test_entity_and_role_hashes_are_unique_for_incremental_builds():
    nlp = NaturalLanguageProcessor(HOME_ASSISTANT_APP_PATH)
    nlp.build(incremental=True)

    example_cache = os.listdir(MODEL_CACHE_PATH.format(app_path=HOME_ASSISTANT_APP_PATH))[0]
    unique_hashs = set()

    for domain in nlp.domains:
        for intent in nlp.domains[domain].intents:
            _, cached_path = get_entity_model_paths(
                HOME_ASSISTANT_APP_PATH, domain, intent, timestamp=example_cache)
            hash_val = open(cached_path + '.hash', 'r').read()
            assert hash_val not in unique_hashs
            unique_hashs.add(hash_val)

            for entity in nlp.domains[domain].intents[intent].entity_recognizer.entity_types:
                _, cached_path = get_role_model_paths(
                    HOME_ASSISTANT_APP_PATH, domain, intent, entity, timestamp=example_cache)
                hash_val = open(cached_path + '.hash', 'r').read()
                assert hash_val not in unique_hashs
                unique_hashs.add(hash_val)
