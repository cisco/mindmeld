#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_nlp
----------------------------------

Tests for NaturalLanguageProcessor module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import pytest

from mmworkbench.exceptions import ProcessorError, AllowedNlpClassesKeyError
from mmworkbench.components import NaturalLanguageProcessor


@pytest.fixture
def empty_nlp(kwik_e_mart_app_path):
    """Provides an empty, unbuilt processor instance"""
    return NaturalLanguageProcessor(app_path=kwik_e_mart_app_path)


def test_instantiate(kwik_e_mart_app_path):
    """Tests creating an NLP instance"""
    nlp = NaturalLanguageProcessor(kwik_e_mart_app_path)
    assert nlp


def test_build(empty_nlp):
    """Tests building a processor with default config.

    This is a basic sanity check to make sure there are no exceptions.
    """
    nlp = empty_nlp
    nlp.build()


def test_dump(kwik_e_mart_nlp):
    """Test dump method of nlp"""
    kwik_e_mart_nlp.dump()


def test_early_process(empty_nlp):
    """Tests that attempting to process a message without first loading or
    building models will raise an exception"""
    with pytest.raises(ProcessorError):
        empty_nlp.process('Hello')


@pytest.mark.skip
def test_load(kwik_e_mart_nlp):
    """Tests loading a processor from disk"""
    kwik_e_mart_nlp.load()


def test_process(kwik_e_mart_nlp):
    """Tests a basic call to process"""
    response = kwik_e_mart_nlp.process('Hello')

    assert response == {
        'text': 'Hello',
        'domain': 'store_info',
        'intent': 'greet',
        'entities': []
    }


test_data_1 = [
    (['store_info.find_nearest_store'], 'store near MG Road',
     'store_info', 'find_nearest_store'),
    (['store_info.find_nearest_store', 'store_info.greet'], 'hello!',
     'store_info', 'greet'),
    (['store_info.find_nearest_store'], 'hello!',
     'store_info', 'find_nearest_store'),
    (['store_info.*'], 'hello!', 'store_info', 'greet')
]


@pytest.mark.parametrize("allowed_intents,query,expected_domain,expected_intent", test_data_1)
def test_nlp_hierarchy_bias_for_user_bias(kwik_e_mart_nlp, allowed_intents, query,
                                          expected_domain, expected_intent):
    """Tests user specified domain and intent biases"""
    extracted_intents = kwik_e_mart_nlp.extract_allowed_intents(allowed_intents)
    response = kwik_e_mart_nlp.process(query, extracted_intents)

    assert response == {
        'text': query,
        'domain': expected_domain,
        'intent': expected_intent,
        'entities': []
    }


test_data_2 = [
    (['store_info.*', 'store_info.greet'],
     'hello!', 'store_info', 'greet'),
    (['store_info.find_nearest_store'], 'hello!', 'store_info',
     'find_nearest_store'),
    (['store_info.*'], 'hello!', 'store_info', 'greet'),
    (['store_info.*', 'store_info.find_nearest_store'], 'store near MG Road',
     'store_info', 'find_nearest_store')
]


@pytest.mark.parametrize("allowed_intents,query,expected_domain,expected_intent", test_data_2)
def test_nlp_hierarchy_using_domains_intents(kwik_e_mart_nlp, allowed_intents,
                                             query, expected_domain, expected_intent):
    """Tests user specified allowable domains and intents"""
    extracted_intents = kwik_e_mart_nlp.extract_allowed_intents(allowed_intents)
    response = kwik_e_mart_nlp.process(query, extracted_intents)

    assert response == {
        'text': query,
        'domain': expected_domain,
        'intent': expected_intent,
        'entities': []
    }


test_data_3 = [
    "what mythical scottish town appears for one day every 100 years",
    "lets run 818m",
    "Get me the product id ws-c2950t-24-24 tomorrow",
    ""
]


@pytest.mark.parametrize("query", test_data_3)
def test_nlp_hierarchy_for_queries_mallard_fails_on(kwik_e_mart_nlp, query):
    """Tests user specified allowable domains and intents"""
    response = kwik_e_mart_nlp.process(query)
    assert response['text'] == query


def test_validate_and_extract_allowed_intents(kwik_e_mart_nlp):
    """Tests user specified allowable domains and intents"""
    with pytest.raises(ValueError):
        kwik_e_mart_nlp.extract_allowed_intents(['store_info'])
    with pytest.raises(AllowedNlpClassesKeyError):
        kwik_e_mart_nlp.extract_allowed_intents(['unrelated_domain.*'])
    with pytest.raises(AllowedNlpClassesKeyError):
        kwik_e_mart_nlp.extract_allowed_intents(['store_info.unrelated_intent'])


test_data_4 = [
    (
        ['when is the 23rd helm street quickie mart open?',
         'when is the 23rd elm st kwik-e-mart open?',
         'when is the 23 elm street quicky mart open?'],
        'store_info',
        'get_store_hours',
        [['23rd helm street'], ['23rd elm st'], ['23 elm street']],
        [['23rd helm street', '23rd elm st', '23 elm street']],
     ),
    (
        ['is the 104 first street store open this sunday',
         'is the first street store open this sunday',
         'is the 10 4 street store open this sunday'],
        'store_info',
        'get_store_hours',
        [['104 first street', 'this sunday'], ['first street', 'this sunday'],
         ['10 4 street', 'this sunday']],
        [['104 first street', 'first street', '10 4 street'],
         ['this sunday', 'this sunday', 'this sunday']]
    )
]


@pytest.mark.parametrize("queries,expected_domain,expected_intent,expected_nbest_entities,"
                         "expected_aligned_entities", test_data_4)
def test_process_nbest(kwik_e_mart_nlp, queries, expected_domain, expected_intent,
                       expected_nbest_entities, expected_aligned_entities):
    """Tests a call to process with n-best transcripts passed in."""
    response = kwik_e_mart_nlp.process(queries)
    response['entities_text'] = [e['text'] for e in response['entities']]
    response.pop('entities')
    response['nbest_transcripts_entities_text'] = [[e['text'] for e in n_entities]
                                                   for n_entities in
                                                   response['nbest_transcripts_entities']]
    response.pop('nbest_transcripts_entities')
    response['nbest_aligned_entities_text'] = [[e['text'] for e in n_entities]
                                               for n_entities in response['nbest_aligned_entities']]
    response.pop('nbest_aligned_entities')

    assert response == {
        'text': queries[0],
        'domain': expected_domain,
        'intent': expected_intent,
        'entities_text': expected_nbest_entities[0],
        'nbest_transcripts_text': queries,
        'nbest_transcripts_entities_text': expected_nbest_entities,
        'nbest_aligned_entities_text': expected_aligned_entities
    }


test_data_5 = [
    (
        ['hi there', 'hi bear', 'high chair'],
        'store_info',
        'greet'
     )
]


@pytest.mark.parametrize("queries,expected_domain,expected_intent", test_data_5)
def test_process_nbest_unspecified_intent(kwik_e_mart_nlp, queries,
                                          expected_domain, expected_intent):
    """Tests a basic call to process with n-best transcripts passed in
    for an intent where n-best processing is unavailable.
    """
    response = kwik_e_mart_nlp.process(queries)

    assert response == {
        'text': queries[0],
        'domain': expected_domain,
        'intent': expected_intent,
        'entities': []
    }


test_data_6 = [
    ([])
]


@pytest.mark.parametrize("queries", test_data_6)
def test_process_empty_nbest_unspecified_intent(kwik_e_mart_nlp, queries):
    """Tests a basic call to process with n-best transcripts passed in
    for an intent where n-best is an empty list.
    """
    response = kwik_e_mart_nlp.process(queries)
    assert response['text'] == ''


def test_parallel_processing(kwik_e_mart_nlp):
    nlp = kwik_e_mart_nlp
    import mmworkbench.components.nlp as nlp_module
    import os
    import time
    if nlp_module.executor:
        input_list = ['A', 'B', 'C']
        parent = os.getpid()

        # test adding a new function to an instance
        def test_function(self, item):
            item = item.lower()
            if os.getpid() == parent:
                item = item + '-parent'
            else:
                item = item + '-child'
            return item
        prev_executor = nlp_module.executor
        nlp._test_function = test_function.__get__(nlp)
        processed = nlp._process_list(input_list, '_test_function')
        # verify the process pool was restarted
        # (the function doesn't exist yet in the subprocesses)
        assert nlp_module.executor != prev_executor
        # verify the list was processed by main process
        assert processed == ('a-parent', 'b-parent', 'c-parent')

        prev_executor = nlp_module.executor
        processed = nlp._process_list(input_list, '_test_function')
        # verify the process pool was not restarted
        assert prev_executor == nlp_module.executor
        # verify the list was processed by subprocesses
        assert processed == ('a-child', 'b-child', 'c-child')

        # test that the timeout works properly
        def slow_function(self, item):
            item = item.lower()
            if os.getpid() == parent:
                item = item + '-parent'
            else:
                # sleep enough to trigger a timeout in the child process
                time.sleep(nlp_module.SUBPROCESS_WAIT_TIME + 0.1)
                item = item + '-child'
            return item
        nlp._test_function = slow_function.__get__(nlp)
        nlp_module.restart_subprocesses()
        prev_executor = nlp_module.executor
        processed = nlp._process_list(input_list, '_test_function')
        # verify the process pool was restarted due to timeout
        assert prev_executor != nlp_module.executor
        # verify the list was processed by main process
        assert processed == ('a-parent', 'b-parent', 'c-parent')


def test_custom_data(kwik_e_mart_nlp):
    store_info_processor = kwik_e_mart_nlp.domains.store_info

    store_hours_ent_rec = store_info_processor.intents.get_store_hours.entity_recognizer
    assert store_hours_ent_rec._model_config.train_label_set == 'testtrain.*\\.txt'
    assert store_hours_ent_rec._model_config.test_label_set == 'testtrain.*\\.txt'

    # make sure another intent doesn't have the same custom data specs
    exit_ent_rec = store_info_processor.intents.exit.entity_recognizer
    assert exit_ent_rec._model_config.train_label_set != 'testtrain.*\\.txt'
    assert exit_ent_rec._model_config.test_label_set != 'testtrain.*\\.txt'
