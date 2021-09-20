#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_nlp
----------------------------------

Tests for NaturalLanguageProcessor module.
"""
import math

# pylint: disable=locally-disabled,redefined-outer-name
import pytest

from mindmeld.components import NaturalLanguageProcessor
from mindmeld.exceptions import ProcessorError
from marshmallow.exceptions import ValidationError
from mindmeld.components.domain_classifier import DomainClassifier


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
        empty_nlp.process("Hello")


@pytest.mark.skip
def test_load(kwik_e_mart_nlp):
    """Tests loading a processor from disk"""
    kwik_e_mart_nlp.load()


def test_process(kwik_e_mart_nlp):
    """Tests a basic call to process"""
    response = kwik_e_mart_nlp.process("Hello")

    assert response == {
        "text": "Hello",
        "domain": "store_info",
        "intent": "greet",
        "entities": [],
    }


def test_query_info_contains_language_information(kwik_e_mart_nlp):
    query_info = kwik_e_mart_nlp.resource_loader.file_to_query_info
    file_to_test = None
    for file in query_info.keys():
        if "train.txt" in file:
            file_to_test = file
            break
    row_id = query_info[file_to_test]["query_ids"][0]
    query = kwik_e_mart_nlp.resource_loader.query_cache.get(row_id).query
    assert query.language == "en"
    assert query.locale == "en_CA"


def test_process_contains_language_information(kwik_e_mart_nlp):
    # Timestamp is set to Jan 2nd 2020 so that the resolve thanksgiving date will be in 2020
    result = kwik_e_mart_nlp.process(
        "Is the Main Street location open for Thanksgiving?", timestamp=1578025558
    )
    for entity in result["entities"]:
        if entity["text"] == "Thanksgiving":
            # American thanksgiving is 2020-11-26T00:00:00.000-08:00,
            # which is different from Canada's.
            assert entity["value"][0]["value"] == "2020-10-12T00:00:00.000-07:00"


def test_inspect_contains_language_information(kwik_e_mart_nlp, mocker):
    mock = mocker.patch.object(DomainClassifier, "inspect", return_value=[])
    kwik_e_mart_nlp.inspect("When is main street open tomorrow", domain="store_info")
    query = mock.call_args_list[0][0][0]
    assert query.language == "en"
    assert query.locale == "en_CA"


def test_role_classification(home_assistant_nlp):
    allowed_intents = ["times_and_dates.change_alarm.sys_time.new_time"]
    extracted_intents = home_assistant_nlp.extract_nlp_masked_components_list(allowed_intents)
    response = home_assistant_nlp.process("5:30am", extracted_intents)
    assert response['entities'][0]['role'] == "new_time"

    allowed_intents = ["times_and_dates.change_alarm.sys_time.old_time"]
    extracted_intents = home_assistant_nlp.extract_nlp_masked_components_list(allowed_intents)
    response = home_assistant_nlp.process("5:30am", extracted_intents)
    assert response['entities'][0]['role'] == "old_time"

    allowed_intents = ["times_and_dates.change_alarm.sys_time.*"]
    extracted_intents = home_assistant_nlp.extract_nlp_masked_components_list(allowed_intents)
    response = home_assistant_nlp.process("5:30am", extracted_intents)
    assert response['entities'][0]['role'] == "new_time"

    allowed_intents = ["times_and_dates.change_alarm.*.*"]
    extracted_intents = home_assistant_nlp.extract_nlp_masked_components_list(allowed_intents)
    response = home_assistant_nlp.process("5:30am", extracted_intents)
    assert response['entities'][0]['role'] == "new_time"


test_data_1 = [
    (
        ["store_info.find_nearest_store"],
        "store near MG Road",
        "store_info",
        "find_nearest_store", []
    ),
    (
        ["store_info.find_nearest_store", "store_info.greet"],
        "hello!",
        "store_info",
        "greet", []
    ),
    (["store_info.find_nearest_store"], "hello!", "store_info", "find_nearest_store", []),
    (["store_info.*"], "hello!", "store_info", "greet", []),
    (["store_info.get_store_hours.store_name"],
     "will springfield mall be open and what time will it close",
     "store_info", "get_store_hours", 'springfield mall')
]


@pytest.mark.parametrize(
    "allowed_intents,query,expected_domain,expected_intent,expected_entities", test_data_1
)
def test_nlp_hierarchy_bias_for_user_bias(
    kwik_e_mart_nlp, allowed_intents, query, expected_domain, expected_intent, expected_entities
):
    """Tests user specified domain and intent biases"""
    extracted_intents = kwik_e_mart_nlp.extract_nlp_masked_components_list(allowed_intents)
    response = kwik_e_mart_nlp.process(query, extracted_intents)
    assert response['text'] == query
    assert response['domain'] == expected_domain
    assert response['intent'] == expected_intent
    if expected_entities:
        assert response['entities'][0]['text'] == expected_entities


test_data_10 = [
    (
        ["times_and_dates.change_alarm"],
        {'times_and_dates': {'change_alarm': {'sys_time': {
            'new_time': {}, 'old_time': {}}, 'sys_interval': {'old_time': {}}}}},
    ),
    (
        ["times_and_dates.*.sys_time.new_time"],
        {'times_and_dates': {'change_alarm': {'sys_time': {'new_time': {}}}}},
    ),
    (
        ["smart_home.set_thermostat.sys_temperature"],
        {'smart_home': {'set_thermostat': {'sys_temperature': {'room_temperature': {}}}}},
    ),
    (
        ["smart_home.*.sys_temperature"],
        {'smart_home': {'set_thermostat': {'sys_temperature': {'room_temperature': {}}}}},
    ),
    (
        ["times_and_dates.*.sys_time.new_time",
         "smart_home.set_thermostat.sys_temperature.room_temperature"],
        {'smart_home': {'set_thermostat': {'sys_temperature': {'room_temperature': {}}}},
         'times_and_dates': {'change_alarm': {'sys_time': {'new_time': {}}}}},
    ),
    (
        ["times_and_dates.*.sys_time.new_time", "times_and_dates.change_alarm.sys_time.old_time"],
        {'times_and_dates': {'change_alarm': {'sys_time': {'old_time': {}, 'new_time': {}}}}},
    ),
    (
        ["times_and_dates.*.*.new_time", "times_and_dates.change_alarm.sys_time.*"],
        {'times_and_dates': {'change_alarm': {'sys_time': {'old_time': {}, 'new_time': {}}}}},
    ),
    (
        ["times_and_dates.change_alarm.*.new_time"],
        {'times_and_dates': {'change_alarm': {'sys_time': {'new_time': {}}}}},
    ),
    (
        ["times_and_dates.change_alarm.sys_time.*"],
        {'times_and_dates': {'change_alarm': {'sys_time': {'new_time': {}, 'old_time': {}}}}},
    ),
    (
        ["times_and_dates.change_alarm.sys_time"],
        {'times_and_dates': {'change_alarm': {'sys_time': {'old_time': {}, 'new_time': {}}}}},
    ),
    (
        ["times_and_dates.*"],
        {'times_and_dates': {'change_alarm': {'sys_time': {'new_time': {}, 'old_time': {}},
                                              'sys_interval': {'old_time': {}}}}}
    ),
    (
        ["smart_home.*"],
        {'smart_home': {'set_thermostat': {'all': {}, 'sys_time': {}, 'sys_interval': {},
                                           'location': {}, 'sys_temperature':
                                               {'room_temperature': {}}}}}
    ),
    (
        ["smart_home.*.sys_temperature", "times_and_dates.change_alarm.*"],
        {'times_and_dates': {'change_alarm': {'sys_interval': {'old_time': {}},
                                              'sys_time': {'old_time': {}, 'new_time': {}}}},
         'smart_home': {'set_thermostat': {'sys_temperature': {'room_temperature': {}}}}},
    ),
    (
        ["feedback.compliment", "times_and_dates.change_alarm.*"],
        {'times_and_dates': {'change_alarm': {'sys_interval': {'old_time': {}},
                                              'sys_time': {'old_time': {}, 'new_time': {}}}},
         'feedback': {'compliment': {}}},
    ),
]


@pytest.mark.parametrize(
    "allowed_intents,expected_nlp_hierarchy", test_data_10
)
def test_nlp_hierarchy_for_allowed_intents(
    home_assistant_nlp, allowed_intents, expected_nlp_hierarchy
):
    """Tests user specified domain and intent biases"""
    extracted_intents = home_assistant_nlp.extract_nlp_masked_components_list(allowed_intents)
    assert extracted_intents == expected_nlp_hierarchy


test_data_110 = [
    (
        ["times_and_dates.change_alarm"], ["times_and_dates.change_alarm.sys_time.new_time"],
        {'times_and_dates': {'change_alarm': {'sys_time': {'old_time': {}},
                                              'sys_interval': {'old_time': {}}}}},
    ),
    (
        ["smart_home.set_thermostat"], ["smart_home.set_thermostat.sys_temperature",
                                        "smart_home.set_thermostat.location"],
        {'smart_home': {'set_thermostat': {'sys_interval': {}, 'all': {}, 'sys_time': {}}}},
    ),
    (
        [], ["smart_home.set_thermostat"],
        {'times_and_dates': {'change_alarm': {'sys_time': {'new_time': {}, 'old_time': {}},
                                              'sys_interval': {'old_time': {}}}},
         'feedback': {'insult': {}, 'compliment': {}}},
    ),
    (
        [], ["smart_home.set_thermostat.sys_temperature"],
        {'smart_home': {'set_thermostat': {'location': {}, 'all': {}, 'sys_interval': {},
                                           'sys_time': {}}},
         'feedback': {'insult': {}, 'compliment': {}},
         'times_and_dates': {'change_alarm': {'sys_interval': {'old_time': {}},
                                              'sys_time': {'new_time': {}, 'old_time': {}}}}},
    ),
]


@pytest.mark.parametrize(
    "allow_nlp,deny_nlp,expected_nlp_hierarchy", test_data_110
)
def test_nlp_hierarchy_for_allow_deny_nlp(
    home_assistant_nlp, allow_nlp, deny_nlp, expected_nlp_hierarchy
):
    """Tests user specified domain and intent biases"""
    extracted_nlp = home_assistant_nlp.extract_nlp_masked_components_list(allow_nlp, deny_nlp)
    assert extracted_nlp == expected_nlp_hierarchy


test_data_2 = [
    (["store_info.*", "store_info.greet"], "hello!", "store_info", "greet"),
    (["store_info.find_nearest_store"], "hello!", "store_info", "find_nearest_store"),
    (["store_info.*"], "hello!", "store_info", "greet"),
    (
        ["store_info.*", "store_info.find_nearest_store"],
        "store near MG Road",
        "store_info",
        "find_nearest_store",
    ),
]


@pytest.mark.parametrize(
    "allowed_intents,query,expected_domain,expected_intent", test_data_2
)
def test_nlp_hierarchy_using_domains_intents(
    kwik_e_mart_nlp, allowed_intents, query, expected_domain, expected_intent
):
    """Tests user specified allowable domains and intents"""
    extracted_intents = kwik_e_mart_nlp.extract_nlp_masked_components_list(allowed_intents)
    response = kwik_e_mart_nlp.process(query, extracted_intents)
    assert response == {
        "text": query,
        "domain": expected_domain,
        "intent": expected_intent,
        "entities": [],
    }


test_data_dyn = [
    ("kadubeesanahalli", None, "store_info", "help", ""),
    ("45 Fifth", None, "store_info", "get_store_hours", "45 Fifth"),
    (
        "kadubeesanahalli",
        {"gazetteers": {"store_name": {"kadubeesanahalli": 10.0}}},
        "store_info",
        "get_store_hours",
        "kadubeesanahalli",
    ),
]


@pytest.mark.parametrize(
    "query,dyn_gaz,expected_domain,expected_intent,expected_entity", test_data_dyn
)
def test_nlp_hierarchy_using_dynamic_gazetteer(
    kwik_e_mart_nlp, query, dyn_gaz, expected_domain, expected_intent, expected_entity
):
    """Tests user specified allowable domains and intents"""
    response = kwik_e_mart_nlp.process(query, dynamic_resource=dyn_gaz)

    if dyn_gaz:
        assert (
            query
            not in kwik_e_mart_nlp.resource_loader.get_gazetteer("store_name")[
                "entities"
            ]
        )
        in_gaz_tokens = "45 Fifth".lower()
        assert (
            in_gaz_tokens
            in kwik_e_mart_nlp.resource_loader.get_gazetteer("store_name")["entities"]
        )

    assert response["domain"] == expected_domain

    if expected_intent != "get_store_hours":
        assert response["intent"] != "get_store_hours"
    else:
        assert response["intent"] == expected_intent

    if expected_entity == "":
        assert response["entities"] == []
    else:
        assert expected_entity in [entity["text"] for entity in response["entities"]]


def test_allowed_entities(kwik_e_mart_nlp):
    res = kwik_e_mart_nlp.process("peanut",
                                  allowed_intents=["store_info.get_store_number.store_name"])
    assert res['entities'][0]['type'] == 'store_name'
    assert res['entities'][0]['text'] == 'peanut'

    res = kwik_e_mart_nlp.process("xyz",
                                  allowed_intents=["store_info.get_store_number.store_name"])
    assert res['entities'] == []

    res = kwik_e_mart_nlp.process("xyz",
                                  allowed_intents=["store_info.get_store_number.store_name"],
                                  dynamic_resource={'gazetteers': {'store_name': {'xyz': 1.0}}})
    assert res['entities'][0]['type'] == 'store_name'
    assert res['entities'][0]['text'] == 'xyz'


def test_disallowed_entities(kwik_e_mart_nlp):
    res = kwik_e_mart_nlp.process("hello")
    assert res['intent'] == 'greet'
    res = kwik_e_mart_nlp.process("hello", deny_nlp=["store_info.greet"])
    assert res['intent'] != 'greet'

    res = kwik_e_mart_nlp.process("transfer $200 from checking to savings")
    assert res['intent'] == 'transfer_money'
    assert res['entities'][0]['type'] == 'sys_amount-of-money'
    assert res['entities'][1]['type'] == 'account_type'
    assert res['entities'][1]['role'] == 'origin'
    assert res['entities'][2]['type'] == 'account_type'
    assert res['entities'][2]['role'] == 'dest'

    res = kwik_e_mart_nlp.process("transfer $200 from checking to savings",
                                  deny_nlp=["banking.transfer_money"])
    assert res['intent'] != 'transfer_money'

    res = kwik_e_mart_nlp.process("transfer $200 from checking to savings",
                                  deny_nlp=["banking.transfer_money.account_type"])
    assert res['intent'] == 'transfer_money'
    assert 'account_type' not in {entity['type'] for entity in res['entities']}

    res = kwik_e_mart_nlp.process("please can you tell me if springfield is "
                                  "possibly open at this time on friday",
                                  allow_nlp=["store_info"])
    assert res['intent'] == 'get_store_hours'
    assert 'sys_time' in {entity['type'] for entity in res['entities']}

    res = kwik_e_mart_nlp.process("please can you tell me if springfield is "
                                  "possibly open at this time on friday",
                                  allow_nlp=["store_info"],
                                  deny_nlp=["store_info.get_store_hours.sys_time"])
    assert res['intent'] == 'get_store_hours'
    assert 'sys_time' not in {entity['type'] for entity in res['entities']}

    # If all entities in an intent are denied, the intent is NOT denied but
    # the entities are denied
    res = kwik_e_mart_nlp.process("please can you tell me if springfield is "
                                  "possibly open at this time on friday",
                                  allow_nlp=["store_info"],
                                  deny_nlp=["store_info.get_store_hours.sys_time",
                                            "store_info.get_store_hours.store_name"])
    assert res['intent'] == 'get_store_hours'
    assert 'sys_time' not in {entity['type'] for entity in res['entities']}
    assert 'store_name' not in {entity['type'] for entity in res['entities']}

    res = kwik_e_mart_nlp.process("please can you tell me if springfield is "
                                  "possibly open at this time on friday",
                                  allow_nlp=["store_info", "banking.transfer_money"],
                                  deny_nlp=["store_info.get_store_hours"])
    assert res['domain'] == 'store_info'
    assert res['intent'] != 'get_store_hours'

    # We fail open here since allow_nlp and deny_nlp are the same
    res = kwik_e_mart_nlp.process("please can you tell me if springfield is "
                                  "possibly open at this time on friday",
                                  allow_nlp=["store_info", "banking.transfer_money"],
                                  deny_nlp=["store_info", "banking.transfer_money"])
    assert res['domain'] == 'store_info'
    assert 'sys_time' in {entity['type'] for entity in res['entities']}

    # We fail open here since allow_nlp is a subset of deny_nlp
    res = kwik_e_mart_nlp.process("please can you tell me if springfield is "
                                  "possibly open at this time on friday",
                                  allow_nlp=["store_info.get_store_hours"],
                                  deny_nlp=["store_info"])
    assert res['domain'] == 'store_info'
    assert 'sys_time' in {entity['type'] for entity in res['entities']}


test_find_entities_in_text_data = [
    ('20', None, {'sys_temperature': {}}),
    ('2:30', None, {'sys_time': {}}),
    # The below test case has overlapping entities
    ('$20 5', None, {'sys_amount-of-money': {}}),
    ('foyer', {"gazetteers": {"location": {"foyer": 10.0}}}, {'location': {}}),
]


@pytest.mark.parametrize(
    "query_text,dyn_gaz,allowed_nlp", test_find_entities_in_text_data
)
def test_find_entities_in_text(home_assistant_nlp, query_factory, query_text, dyn_gaz, allowed_nlp):
    # Duckling tests
    query = (query_factory.create_query(text=query_text),)
    res = home_assistant_nlp.domains['smart_home'].intents['set_thermostat']._find_entities_in_text(
        query, dyn_gaz, allowed_nlp, 3)
    assert res[0][0].text == query_text
    assert res[0][0].entity.type == list(allowed_nlp.keys())[0]


test_data_3 = [
    "what mythical scottish town appears for one day every 100 years",
    "lets run 818m",
    "Get me the product id ws-c2950t-24-24 tomorrow",
    "",
]


@pytest.mark.parametrize("query", test_data_3)
def test_nlp_hierarchy_for_queries_duckling_fails_on(kwik_e_mart_nlp, query):
    """Tests user specified allowable domains and intents"""
    response = kwik_e_mart_nlp.process(query)
    assert response["text"] == query


def test_validate_and_extract_allowed_intents(kwik_e_mart_nlp):
    """Tests user specified allowable domains and intents"""
    with pytest.raises(ValidationError):
        kwik_e_mart_nlp.extract_nlp_masked_components_list(["unrelated_domain.*"])
    with pytest.raises(ValidationError):
        kwik_e_mart_nlp.extract_nlp_masked_components_list(["store_info.unrelated_intent"])


def test_process_verbose_no_entity(kwik_e_mart_nlp):
    """Test basic processing without metadata"""
    response = kwik_e_mart_nlp.process("Hello", verbose=True)

    assert response["domain"] == "store_info"
    assert response["intent"] == "greet"
    assert response["entities"] == []
    assert isinstance(response["confidences"]["domains"]["store_info"], float)
    assert isinstance(response["confidences"]["intents"]["greet"], float)


def test_process_verbose(kwik_e_mart_nlp):
    """Test basic processing with metadata"""
    response = kwik_e_mart_nlp.process("is the elm street store open", verbose=True)

    assert response["domain"] == "store_info"
    assert response["intent"] == "get_store_hours"
    assert response["entities"][0]["text"] == "elm street"
    assert isinstance(
        response["confidences"]["entities"][0][response["entities"][0]["type"]], float
    )
    assert isinstance(response["confidences"]["domains"]["store_info"], float)
    assert isinstance(response["confidences"]["intents"]["get_store_hours"], float)


def test_process_verbose_long_tokens(kwik_e_mart_nlp):
    """Test confidence for entities that have lower raw tokens indices than normalized tokens"""
    text = "Is the Kwik-E-Mart open tomorrow?"
    response = kwik_e_mart_nlp.process(text, verbose=True)

    text_preparation_pipeline = (
        kwik_e_mart_nlp.resource_loader.query_factory.text_preparation_pipeline
    )

    raw_tokens_text = [t["text"] for t in text_preparation_pipeline.tokenize(text)]
    assert raw_tokens_text == ["Is", "the", "Kwik-E-Mart", "open", "tomorrow?"]

    normalized_tokens = text_preparation_pipeline.tokenize_and_normalize(text)
    normalized_tokens_text = [t["entity"] for t in normalized_tokens]
    assert normalized_tokens_text == ["is", "the", "kwik", "e", "mart", "open", "tomorrow"]

    assert response["domain"] == "store_info"
    assert response["intent"] == "get_store_hours"
    assert response["entities"][0]["text"] == "tomorrow"
    assert isinstance(
        response["confidences"]["entities"][0][response["entities"][0]["type"]], float
    )


def test_process_verbose_short_tokens(kwik_e_mart_nlp):
    """Test confidence for entities that have higher raw tokens indices than normalized tokens"""
    text = "when ** open -- tomorrow?"
    response = kwik_e_mart_nlp.process(text, verbose=True)

    text_preparation_pipeline = (
        kwik_e_mart_nlp.resource_loader.query_factory.text_preparation_pipeline
    )

    raw_tokens_text = [t["text"] for t in text_preparation_pipeline.tokenize(text)]
    assert raw_tokens_text == ["when", "**", "open", "--", "tomorrow?"]

    normalized_tokens = text_preparation_pipeline.tokenize_and_normalize(text)
    normalized_tokens_text = [t["entity"] for t in normalized_tokens]
    assert normalized_tokens_text == ["when", "open", "tomorrow"]

    assert response["domain"] == "store_info"
    assert response["intent"] == "get_store_hours"
    assert response["entities"][0]["text"] == "tomorrow"
    assert isinstance(
        response["confidences"]["entities"][0][response["entities"][0]["type"]], float
    )


test_nbest = [
    (
        [
            "when is the 23rd elm street quickie mart open?",
            "when is the 23rd elm st kwik-e-mart open?",
            "when is the 23 elm street quicky mart open?",
        ],
        "store_info",
        "get_store_hours",
    )
]


@pytest.mark.parametrize("queries,expected_domain,expected_intent", test_nbest)
def test_nbest_process_verbose(
    kwik_e_mart_nlp, queries, expected_domain, expected_intent
):
    response = kwik_e_mart_nlp.process(queries, verbose=True)
    response["entities_text"] = [e["text"] for e in response["entities"]]
    for i, e in enumerate(response["entities"]):
        assert isinstance(response["confidences"]["entities"][i][e["type"]], float)
    assert isinstance(response["confidences"]["domains"][expected_domain], float)
    assert isinstance(response["confidences"]["intents"][expected_intent], float)


test_data_4 = [
    (
        [
            "when is the 23rd elm street quickie mart open?",
            "when is the 23rd elm st kwik-e-mart open?",
            "when is the 23 elm street quicky mart open?",
        ],
        "store_info",
        "get_store_hours",
        [["23rd elm street"], ["23rd elm st"], ["23 elm street"]],
        [["23rd elm street", "23rd elm st", "23 elm street"]],
    ),
    (
        [
            "is the 104 first street store open this sunday",
            "is the first street store open this sunday",
            "is the 10 4 street store open this sunday",
        ],
        "store_info",
        "get_store_hours",
        [
            ["104 first street", "sunday"],
            ["first street", "sunday"],
            ["10 4 street", "sunday"],
        ],
        [["104 first street", "first street", "10 4 street"], ["sunday", "sunday"]],
    ),
]


@pytest.mark.parametrize(
    "queries,expected_domain,expected_intent,expected_nbest_entities,"
    "expected_aligned_entities",
    test_data_4,
)
def test_process_nbest(
    kwik_e_mart_nlp,
    queries,
    expected_domain,
    expected_intent,
    expected_nbest_entities,
    expected_aligned_entities,
):
    """Tests a call to process with n-best transcripts passed in."""
    response = kwik_e_mart_nlp.process(queries)
    response["entities_text"] = [e["text"] for e in response["entities"]]
    response.pop("entities")
    response["nbest_transcripts_entities_text"] = [
        [e["text"] for e in n_entities]
        for n_entities in response["nbest_transcripts_entities"]
    ]
    response.pop("nbest_transcripts_entities")
    response["nbest_aligned_entities_text"] = [
        [e["text"] for e in n_entities]
        for n_entities in response["nbest_aligned_entities"]
    ]
    response.pop("nbest_aligned_entities")

    assert response == {
        "text": queries[0],
        "domain": expected_domain,
        "intent": expected_intent,
        "entities_text": expected_nbest_entities[0],
        "nbest_transcripts_text": queries,
        "nbest_transcripts_entities_text": expected_nbest_entities,
        "nbest_aligned_entities_text": expected_aligned_entities,
    }


test_data_5 = [(["hi there", "hi bear", "high chair"], "store_info", "greet")]


@pytest.mark.parametrize("queries,expected_domain,expected_intent", test_data_5)
def test_process_nbest_unspecified_intent(
    kwik_e_mart_nlp, queries, expected_domain, expected_intent
):
    """Tests a basic call to process with n-best transcripts passed in
    for an intent where n-best processing is unavailable.
    """
    response = kwik_e_mart_nlp.process(queries)

    assert response == {
        "text": queries[0],
        "domain": expected_domain,
        "intent": expected_intent,
        "entities": [],
    }


test_data_6 = [([])]


@pytest.mark.parametrize("queries", test_data_6)
def test_process_empty_nbest_unspecified_intent(kwik_e_mart_nlp, queries):
    """Tests a basic call to process with n-best transcripts passed in
    for an intent where n-best is an empty list.
    """
    response = kwik_e_mart_nlp.process(queries)
    assert response["text"] == ""


def test_parallel_processing(kwik_e_mart_nlp):
    nlp = kwik_e_mart_nlp
    import mindmeld.components.nlp as nlp_module
    import os
    import time

    if nlp_module.executor:
        input_list = ["A", "B", "C"]
        parent = os.getpid()

        # test adding a new function to an instance
        def test_function(self, item):
            item = item.lower()
            if os.getpid() == parent:
                item = item + "-parent"
            else:
                item = item + "-child"
            return item

        prev_executor = nlp_module.executor
        nlp._test_function = test_function.__get__(nlp)
        processed = nlp._process_list(input_list, "_test_function")
        # verify the process pool was restarted
        # (the function doesn't exist yet in the subprocesses)
        assert nlp_module.executor != prev_executor
        # verify the list was processed by main process
        assert processed == ("a-parent", "b-parent", "c-parent")

        prev_executor = nlp_module.executor
        processed = nlp._process_list(input_list, "_test_function")
        # verify the process pool was not restarted
        assert prev_executor == nlp_module.executor
        # verify the list was processed by subprocesses
        assert processed == ("a-child", "b-child", "c-child")

        # test that the timeout works properly
        def slow_function(self, item):
            item = item.lower()
            if os.getpid() == parent:
                item = item + "-parent"
            else:
                # sleep enough to trigger a timeout in the child process
                time.sleep(nlp_module.SUBPROCESS_WAIT_TIME + 0.1)
                item = item + "-child"
            return item

        nlp._test_function = slow_function.__get__(nlp)
        nlp_module.restart_subprocesses()
        prev_executor = nlp_module.executor
        processed = nlp._process_list(input_list, "_test_function")
        # verify the process pool was restarted due to timeout
        assert prev_executor != nlp_module.executor
        # verify the list was processed by main process
        assert processed == ("a-parent", "b-parent", "c-parent")


def test_custom_data(kwik_e_mart_nlp):
    store_info_processor = kwik_e_mart_nlp.domains.store_info

    store_hours_ent_rec = store_info_processor.intents.get_store_hours.entity_recognizer
    assert store_hours_ent_rec._model.config.train_label_set == "testtrain.*\\.txt"
    assert store_hours_ent_rec._model.config.test_label_set == "testtrain.*\\.txt"

    # make sure another intent having an entity recognizer doesn't have the same custom data specs
    exit_ent_rec = store_info_processor.intents.get_store_number.entity_recognizer
    assert exit_ent_rec._model.config.train_label_set != "testtrain.*\\.txt"
    assert exit_ent_rec._model.config.test_label_set != "testtrain.*\\.txt"


def test_dynamic_gazetteer_case_sensitiveness(kwik_e_mart_nlp):
    response = kwik_e_mart_nlp.process(
        "find me ala bazaar",
        dynamic_resource={"gazetteers": {"store_name": {"aLA bazAAr": 1000000.0}}},
    )
    # Todo: fix this test for span equality
    assert response["entities"][0]["text"] in "ala bazaar"


def test_word_shape_feature(kwik_e_mart_nlp):
    ic = kwik_e_mart_nlp.domains["store_info"].intent_classifier
    features = ic.view_extracted_features("is the 104 first street store open")

    word_shape_features = {}
    for key in features:
        if "word_shape" in key:
            word_shape_features[key] = features[key]

    shape_1_value = math.log(2, 2) / 7
    expected_features = {
        "bag_of_words|length:1|word_shape:xx": shape_1_value,
        "bag_of_words|length:1|word_shape:xxx": shape_1_value,
        "bag_of_words|length:1|word_shape:xxxx": shape_1_value,
        "bag_of_words|length:1|word_shape:xxxxx": math.log(3, 2) / 7,
        "bag_of_words|length:1|word_shape:xxxxx+": shape_1_value,
        "bag_of_words|length:1|word_shape:ddd": shape_1_value,
    }

    assert expected_features == word_shape_features


def test_sys_entity_feature(kwik_e_mart_nlp):
    ic = kwik_e_mart_nlp.domains["store_info"].intent_classifier
    features = ic.view_extracted_features("is the 104 1st street store open")

    sys_candidate_features = {}
    for key in features:
        if "sys_candidate" in key:
            sys_candidate_features[key] = features[key]

    expected_features = {
        "sys_candidate|type:sys_number": 2,
        "sys_candidate|type:sys_number|granularity:None": 2,
        "sys_candidate|type:sys_ordinal": 1,
        "sys_candidate|type:sys_ordinal|granularity:None": 1,
    }

    assert expected_features == sys_candidate_features


test_data_dyn = [
    ("kadubeesanahalli", None, "store_info", "get_store_hours", ""),
    ("45 Fifth", None, "store_info", "get_store_hours", "45 Fifth"),
    (
        "kadubeesanahalli",
        {"gazetteers": {"store_name": {"kadubeesanahalli": 1}}},
        "store_info",
        "get_store_hours",
        "kadubeesanahalli",
    ),
]


@pytest.mark.parametrize(
    "query,dyn_gaz,expected_domain,expected_intent,expected_entity", test_data_dyn
)
def test_nlp_hierarchy_using_dynamic_gazetteer_and_allowed_intents(
    kwik_e_mart_nlp, query, dyn_gaz, expected_domain, expected_intent, expected_entity
):
    """Tests user specified allowed_nlp_classes and dynamic_resource"""
    response = kwik_e_mart_nlp.process(
        query,
        dynamic_resource=dyn_gaz,
        allowed_nlp_classes={"store_info": {"get_store_hours": {"store_name": {}, "sys_time": {}}}},
    )
    if dyn_gaz:
        assert (
            query
            not in kwik_e_mart_nlp.resource_loader.get_gazetteer("store_name")[
                "entities"
            ]
        )
        in_gaz_tokens = "45 Fifth".lower()
        assert (
            in_gaz_tokens
            in kwik_e_mart_nlp.resource_loader.get_gazetteer("store_name")["entities"]
        )

    assert response["domain"] == expected_domain

    assert response["intent"] == expected_intent

    if expected_entity == "":
        assert response["entities"] == []
    else:
        assert expected_entity in [entity["text"] for entity in response["entities"]]


def test_extract_entity_resolvers(kwik_e_mart_app_path):
    """Tests extracting entity resolvers
    """
    nlp = NaturalLanguageProcessor(kwik_e_mart_app_path)
    entity_processors = nlp.domains['banking'].intents['transfer_money'].get_entity_processors()
    assert len(entity_processors.keys()) == 2
    assert "account_type" in entity_processors.keys()
    assert "sys_amount-of-money" in entity_processors.keys()
    entity_processors = nlp.domains['store_info'].intents['get_store_hours'].get_entity_processors()
    assert len(entity_processors.keys()) == 2
    assert "store_name" in entity_processors.keys()
    assert "sys_time" in entity_processors.keys()
    er = entity_processors["store_name"].entity_resolver
    er.fit()
    expected = {"id": "2", "cname": "Pine and Market"}
    predicted = er.predict("Pine and Market")[0]
    assert predicted["id"] == expected["id"]
    assert predicted["cname"] == expected["cname"]
