"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

import pytest

from mindmeld import NaturalLanguageProcessor


@pytest.fixture
def rasa_nlp(rasa_converter, mindmeld_rasa_converter_app_path):
    nlp = NaturalLanguageProcessor(mindmeld_rasa_converter_app_path)
    nlp.build()
    return nlp


intent_test_data = [
    ("hi there", "greet"),
    ("My name is Josh", "name"),
    ("thank you", "thanks"),
    ("See you later", "goodbye"),
    ("I am Josh Williams", "lastname"),
    ("No, I don't want that", "deny"),
]


@pytest.mark.parametrize("query1,expected_intent", intent_test_data)
def test_nlp_correct_intent(rasa_nlp, query1, expected_intent):
    assert rasa_nlp.process(query1)["intent"] == "{}".format(expected_intent)


def test_nlp_name(rasa_nlp):
    query1 = "My name is Josh"
    assert rasa_nlp.process(query1)["intent"] == "name"
    entity = rasa_nlp.process(query1)["entities"]
    assert entity[0]["text"] == "Josh"


def test_nlp_lastname(rasa_nlp):
    query1 = "I am Josh Williams"
    assert rasa_nlp.process(query1)["intent"] == "lastname"
    entity = rasa_nlp.process(query1)["entities"]
    assert entity[0]["text"] == "Josh"
    assert entity[1]["text"] == "Williams"


convert_test_data = [
    (
        "XyZ date for [21.1(5)L&](product_version0) ?",
        "XyZ date for {21.1(5)L&|product_version0} ?",
        {'{21.1(5)L&|product_version0}'}
    ),
    (
        "what are your abilities?",
        "what are your abilities?",
        {}
    ),
    (
        "What is the End of Life Product Restart Time for [XYZ2990-ESPN19-N](PID) ?",
        "What is the End of Life Product Restart Time for {XYZ2990-ESPN19-N|pid} ?",
        {'{XYZ2990-ESPN19-N|pid}'}
    ),
    (
        "XyZ date for [21.1(5)L&](proDucT_version1) - XyZ date for [21.1(5)L&](product_version2)",
        "XyZ date for {21.1(5)L&|product_version1} - XyZ date for {21.1(5)L&|product_version2}",
        {'{21.1(5)L&|product_version1}', '{21.1(5)L&|product_version2}'}
    ),
]


@pytest.mark.parametrize("rasa_entry,expected_mindmeld_entry,expected_entities", convert_test_data)
def test_translate_rasa_entry_to_mindmeld_entry(
        rasa_converter, rasa_entry, expected_mindmeld_entry, expected_entities):
    mindmeld_entry = rasa_converter._translate_rasa_entry_to_mindmeld_entry(rasa_entry)
    assert mindmeld_entry == expected_mindmeld_entry
    assert len(
        rasa_converter.all_entities.intersection(expected_entities)
    ) == len(expected_entities)
