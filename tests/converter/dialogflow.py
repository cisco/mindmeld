"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

from mindmeld import NaturalLanguageProcessor
from mindmeld.converter.dialogflowconverter import DialogFlowConverter


def test_dialogflow_converter(dialogflow_project_path, mindmeld_project_path2):
    converter = DialogFlowConverter(dialogflow_project_path, mindmeld_project_path2)
    converter.convert_project()
