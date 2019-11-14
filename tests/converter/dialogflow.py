"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

import pytest

from mindmeld import NaturalLanguageProcessor
from mindmeld.converter.dialogflow import DialogflowConverter


def test_dialogflow_converter(dialogflow_project_path, mindmeld_project_path2):
    converter = DialogflowConverter(dialogflow_project_path, mindmeld_project_path2)
    converter.convert_project()


@pytest.fixture
def diaglogflow_nlp(mindmeld_project_path2):
    nlp = NaturalLanguageProcessor(mindmeld_project_path2)
    nlp.build()
    return nlp


def test_nlp_greet(diaglogflow_nlp):
    query1 = "hi there"
    assert (
        diaglogflow_nlp.process(query1)["intent"]
        == "default_welcome_intent_usersays_en"
    )


def test_nlp_i_can_code_in_en(diaglogflow_nlp):
    query1 = "I know Python"
    assert (
        diaglogflow_nlp.process(query1)["intent"]
        == "i_can_code_in_a_language_usersays_en"
    )
    entity = diaglogflow_nlp.process(query1)["entities"]
    assert entity[0]["text"] == "Python"


def test_nlp_i_can_code_in_es(diaglogflow_nlp):
    query1 = "Puedo codificar en Java"
    assert (
        diaglogflow_nlp.process(query1)["intent"]
        == "i_can_code_in_a_language_usersays_es"
    )
    entity = diaglogflow_nlp.process(query1)["entities"]
    assert entity[0]["text"] == "Java"
