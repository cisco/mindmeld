
import pytest
import os
import shutil

from mmworkbench.path import MODEL_CACHE_PATH, get_role_model_paths, get_entity_model_paths
from mmworkbench.exceptions import FileNotFoundError
from mmworkbench.components import NaturalLanguageProcessor


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


test_data_8 = [
    (
        ['change alarm from 2 pm to 4 pm', 'times_and_dates', 'change_alarm', 'sys_time'],
        ['old_time', 'new_time']
     )
]


@pytest.mark.parametrize("example,role_type", test_data_8)
def test_nlp_process_for_roles(home_assistant_nlp, example, role_type):
    result = home_assistant_nlp.process(example)

    assert(result['entities'][0]['role'] == role_type[0])
    assert (result['entities'][1]['role'] == role_type[1])

    result = home_assistant_nlp.process(example, verbose=True)

    assert (result['entities'][0]['role'] == role_type[0])
    assert (result['entities'][1]['role'] == role_type[1])
    assert (result['confidences']['roles'][0])
    assert (result['confidences']['roles'][1])


def test_model_accuracies_are_similar_before_and_after_caching(home_assistant_app_path):
    # clear model cache
    model_cache_path = MODEL_CACHE_PATH.format(app_path=home_assistant_app_path)
    try:
        shutil.rmtree(MODEL_CACHE_PATH.format(app_path=home_assistant_app_path))
    except FileNotFoundError:
        pass

    # Make sure no cache exists
    assert os.path.exists(model_cache_path) is False
    nlp = NaturalLanguageProcessor(home_assistant_app_path)
    nlp.build(incremental=True)
    nlp.dump()

    entity_eval = nlp.domains[
        'times_and_dates'].intents['change_alarm'].entity_recognizer.evaluate()

    role_eval = nlp.domains[
        'times_and_dates'].intents[
        'change_alarm'].entities['sys_time'].role_classifier.evaluate()

    entity_accuracy_no_cache = entity_eval.get_accuracy()
    role_accuracy_no_cache = role_eval.get_accuracy()

    example_cache = os.listdir(MODEL_CACHE_PATH.format(app_path=home_assistant_app_path))[0]
    nlp = NaturalLanguageProcessor(home_assistant_app_path)
    nlp.load(example_cache)

    # make sure cache exists
    assert os.path.exists(model_cache_path) is True

    entity_eval = nlp.domains[
        'times_and_dates'].intents[
        'change_alarm'].entity_recognizer.evaluate()

    role_eval = nlp.domains[
        'times_and_dates'].intents[
        'change_alarm'].entities['sys_time'].role_classifier.evaluate()

    entity_accuracy_cached = entity_eval.get_accuracy()
    role_accuracy_cached = role_eval.get_accuracy()

    assert role_accuracy_no_cache == role_accuracy_cached
    assert entity_accuracy_no_cache == entity_accuracy_cached


def test_all_classifier_are_unique_for_incremental_builds(home_assistant_app_path):
    nlp = NaturalLanguageProcessor(home_assistant_app_path)
    nlp.build(incremental=True)

    example_cache = os.listdir(MODEL_CACHE_PATH.format(app_path=home_assistant_app_path))[0]
    unique_hashs = set()

    for domain in nlp.domains:
        for intent in nlp.domains[domain].intents:
            _, cached_path = get_entity_model_paths(
                home_assistant_app_path, domain, intent, timestamp=example_cache)
            hash_val = open(cached_path + '.hash', 'r').read()
            assert hash_val not in unique_hashs
            unique_hashs.add(hash_val)

            for entity in nlp.domains[domain].intents[intent].entity_recognizer.entity_types:
                _, cached_path = get_role_model_paths(
                    home_assistant_app_path, domain, intent, entity, timestamp=example_cache)
                hash_val = open(cached_path + '.hash', 'r').read()
                assert hash_val not in unique_hashs
                unique_hashs.add(hash_val)
