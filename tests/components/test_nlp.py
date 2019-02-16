#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_nlp
----------------------------------

Tests for NaturalLanguageProcessor module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
import pytest
import math

from mmworkbench.exceptions import ProcessorError, AllowedNlpClassesKeyError
from mmworkbench.components import NaturalLanguageProcessor
from mmworkbench.query_factory import QueryFactory


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


test_data_dyn = [
    ('kadubeesanahalli', None, 'store_info', 'help', ''),
    ('45 Fifth', None, 'store_info', 'get_store_hours', '45 Fifth'),
    ('kadubeesanahalli', {'gazetteers': {'store_name': {'kadubeesanahalli': 1}}},
     'store_info', 'get_store_hours', 'kadubeesanahalli'),
    ('45 Fifth', None, 'store_info', 'get_store_hours', '45 Fifth')
]


@pytest.mark.parametrize("query,dyn_gaz,expected_domain,expected_intent,expected_entity",
                         test_data_dyn)
def test_nlp_hierarchy_using_dynamic_gazetteer(kwik_e_mart_nlp, query, dyn_gaz,
                                               expected_domain, expected_intent, expected_entity):
    """Tests user specified allowable domains and intents"""
    response = kwik_e_mart_nlp.process(query, dynamic_resource=dyn_gaz)

    if dyn_gaz:
        assert query not in kwik_e_mart_nlp.resource_loader.get_gazetteer('store_name')['entities']
        in_gaz_tokens = '45 Fifth'.lower()
        assert in_gaz_tokens in kwik_e_mart_nlp.resource_loader.get_gazetteer(
            'store_name')['entities']

    assert response['domain'] == expected_domain

    if expected_intent != 'get_store_hours':
        assert response['intent'] != 'get_store_hours'
    else:
        assert response['intent'] == expected_intent

    if expected_entity == '':
        assert response['entities'] == []
    else:
        assert expected_entity in [entity['text'] for entity in response['entities']]


test_data_3 = [
    "what mythical scottish town appears for one day every 100 years",
    "lets run 818m",
    "Get me the product id ws-c2950t-24-24 tomorrow",
    ""
]


@pytest.mark.parametrize("query", test_data_3)
def test_nlp_hierarchy_for_queries_duckling_fails_on(kwik_e_mart_nlp, query):
    """Tests user specified allowable domains and intents"""
    response = kwik_e_mart_nlp.process(query)
    assert response['text'] == query


test_data_not_stemmed = [
    "airliner",
    "gyroscopic",
    "adjustable",
    "defensible",
    "irritant",
    "replacement",
    "adjustment",
    "dependent",
    "adoption",
    "communism",
    "activate",
    "effective",
    "bowdlerize",
    "manager",
    "proceed",
    "exceed",
    "succeed",
    "outing",
    "inning",
    "news",
    "sky"
]


@pytest.mark.parametrize("query", test_data_not_stemmed)
def test_nlp_for_non_stemmed_queries(kwik_e_mart_nlp, query):
    """Tests queries that are NOT in the training data but have their stemmed
     versions in the training data"""
    query_factory = QueryFactory.create_query_factory()
    stemmed_tokens = query_factory.create_query(text=query).stemmed_tokens
    assert query == stemmed_tokens[0]


test_data_need_stemming = [
    ("cancelled", "cancel"),
    ("aborted", "abort"),
    ("backwards", "backward"),
    ("exitted", "exit"),
    ("finished", "finish")
]


@pytest.mark.parametrize("query,stemmed_query", test_data_need_stemming)
def test_nlp_for_stemmed_queries(kwik_e_mart_nlp, query, stemmed_query):
    """Tests queries that are NOT in the training data but have their stemmed
     versions in the training data"""
    query_factory = QueryFactory.create_query_factory()
    stemmed_tokens = query_factory.create_query(text=query).stemmed_tokens
    assert stemmed_query == stemmed_tokens[0]


test_data_stemmed = [
    "cancelled",
    "exited",
    "aborted"
]


@pytest.mark.parametrize("query", test_data_stemmed)
def test_nlp_hierarchy_for_stemmed_queries(kwik_e_mart_nlp, query):
    """Tests queries that are NOT in the training data but have their stemmed
     versions in the training data"""
    response = kwik_e_mart_nlp.process(query)
    assert response['text'] == query
    assert response['domain'] == 'store_info'
    assert response['intent'] == 'exit'


def test_validate_and_extract_allowed_intents(kwik_e_mart_nlp):
    """Tests user specified allowable domains and intents"""
    with pytest.raises(ValueError):
        kwik_e_mart_nlp.extract_allowed_intents(['store_info'])
    with pytest.raises(AllowedNlpClassesKeyError):
        kwik_e_mart_nlp.extract_allowed_intents(['unrelated_domain.*'])
    with pytest.raises(AllowedNlpClassesKeyError):
        kwik_e_mart_nlp.extract_allowed_intents(['store_info.unrelated_intent'])


def test_process_verbose_no_entity(kwik_e_mart_nlp):
    """Test basic processing without metadata"""
    response = kwik_e_mart_nlp.process('Hello', verbose=True)

    assert response['domain'] == 'store_info'
    assert response['intent'] == 'greet'
    assert response['entities'] == []
    assert isinstance(response['confidences']['domains']['store_info'], float)
    assert isinstance(response['confidences']['intents']['greet'], float)


def test_process_verbose(kwik_e_mart_nlp):
    """Test basic processing with metadata"""
    response = kwik_e_mart_nlp.process('is the elm street store open', verbose=True)

    assert response['domain'] == 'store_info'
    assert response['intent'] == 'get_store_hours'
    assert response['entities'][0]['text'] == 'elm street'
    assert isinstance(response['confidences']['entities'][0][response['entities'][0]['type']],
                      float)
    assert isinstance(response['confidences']['domains']['store_info'], float)
    assert isinstance(response['confidences']['intents']['get_store_hours'], float)


def test_process_verbose_long_tokens(kwik_e_mart_nlp):
    """Test confidence for entities that have lower raw tokens indices than normalized tokens"""
    text = 'Is the Kwik-E-Mart open tomorrow?'
    response = kwik_e_mart_nlp.process(text, verbose=True)

    tokenizer = kwik_e_mart_nlp.resource_loader.query_factory.tokenizer
    raw_tokens = [t['text'] for t in tokenizer.tokenize_raw(text)]
    normalized_tokens = [t['entity'] for t in tokenizer.tokenize(text)]

    assert raw_tokens == ['Is', 'the', 'Kwik-E-Mart', 'open', 'tomorrow?']
    assert normalized_tokens == ['is', 'the', 'kwik', 'e', 'mart', 'open', 'tomorrow']

    assert response['domain'] == 'store_info'
    assert response['intent'] == 'get_store_hours'
    assert response['entities'][0]['text'] == 'tomorrow'
    assert isinstance(response['confidences']['entities'][0][response['entities'][0]['type']],
                      float)


def test_process_verbose_short_tokens(kwik_e_mart_nlp):
    """Test confidence for entities that have higher raw tokens indices than normalized tokens"""
    text = 'when ** open -- tomorrow?'
    response = kwik_e_mart_nlp.process(text, verbose=True)

    tokenizer = kwik_e_mart_nlp.resource_loader.query_factory.tokenizer
    raw_tokens = [t['text'] for t in tokenizer.tokenize_raw(text)]
    normalized_tokens = [t['entity'] for t in tokenizer.tokenize(text)]

    assert raw_tokens == ['when', '**', 'open', '--', 'tomorrow?']
    assert normalized_tokens == ['when', 'open', 'tomorrow']

    assert response['domain'] == 'store_info'
    assert response['intent'] == 'get_store_hours'
    assert response['entities'][0]['text'] == 'tomorrow'
    assert isinstance(response['confidences']['entities'][0][response['entities'][0]['type']],
                      float)


test_nbest = [
    (
        ['when is the 23rd elm street quickie mart open?',
         'when is the 23rd elm st kwik-e-mart open?',
         'when is the 23 elm street quicky mart open?'],
        'store_info',
        'get_store_hours'
     )]


@pytest.mark.parametrize("queries,expected_domain,expected_intent", test_nbest)
def test_nbest_process_verbose(kwik_e_mart_nlp, queries, expected_domain, expected_intent):
    response = kwik_e_mart_nlp.process(queries, verbose=True)
    response['entities_text'] = [e['text'] for e in response['entities']]
    for i, e in enumerate(response['entities']):
        assert isinstance(response['confidences']['entities'][i][e['type']], float)
    assert isinstance(response['confidences']['domains'][expected_domain], float)
    assert isinstance(response['confidences']['intents'][expected_intent], float)


test_data_4 = [
    (
        ['when is the 23rd elm street quickie mart open?',
         'when is the 23rd elm st kwik-e-mart open?',
         'when is the 23 elm street quicky mart open?'],
        'store_info',
        'get_store_hours',
        [['23rd elm street'], ['23rd elm st'], ['23 elm street']],
        [['23rd elm street', '23rd elm st', '23 elm street']],
    ),
    (
        ['is the 104 first street store open this sunday',
         'is the first street store open this sunday',
         'is the 10 4 street store open this sunday'],
        'store_info',
        'get_store_hours',
        [['104 first street', 'sunday'], ['first street', 'sunday'],
         ['10 4 street', 'sunday']],
        [['104 first street', 'first street', '10 4 street'],
         ['sunday', 'sunday']]
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


def test_dynamic_gazetteer_case_sensitiveness(kwik_e_mart_nlp):
    response = kwik_e_mart_nlp.process(
        "find me ala bazaar",
        dynamic_resource={'gazetteers': {'store_name': {"aLA bazAAr": 1000.0}}})
    assert response['entities'][0]['text'] == "ala bazaar"


def test_word_shape_feature(kwik_e_mart_nlp):
    ic = kwik_e_mart_nlp.domains['store_info'].intent_classifier
    features = ic.view_extracted_features("is the 104 first street store open")

    word_shape_features = {}
    for key in features:
        if 'word_shape' in key:
            word_shape_features[key] = features[key]

    shape_1_value = math.log(2, 2)/7
    expected_features = {
        'bag_of_words|length:1|word_shape:xx': shape_1_value,
        'bag_of_words|length:1|word_shape:xxx': shape_1_value,
        'bag_of_words|length:1|word_shape:xxxx': shape_1_value,
        'bag_of_words|length:1|word_shape:xxxxx': math.log(3, 2)/7,
        'bag_of_words|length:1|word_shape:xxxxx+': shape_1_value,
        'bag_of_words|length:1|word_shape:ddd': shape_1_value
    }

    assert expected_features == word_shape_features


def test_sys_entity_feature(kwik_e_mart_nlp):
    ic = kwik_e_mart_nlp.domains['store_info'].intent_classifier
    features = ic.view_extracted_features("is the 104 1st street store open")

    sys_candidate_features = {}
    for key in features:
        if 'sys_candidate' in key:
            sys_candidate_features[key] = features[key]

    expected_features = {
        'sys_candidate|type:sys_number': 2,
        'sys_candidate|type:sys_number|granularity:None': 2,
        'sys_candidate|type:sys_ordinal': 1,
        'sys_candidate|type:sys_ordinal|granularity:None': 1
    }

    assert expected_features == sys_candidate_features
