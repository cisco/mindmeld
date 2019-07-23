"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

from mindmeld.converter.rasa_converter import RasaConverter


def test_rasa_converter(rasa_project_path, mindmeld_project_path):
    converter = RasaConverter(rasa_project_path, mindmeld_project_path)
    converter.convert_project()
