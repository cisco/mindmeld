"""This module tests the convertion tool that is used to convert a rasa or diaglogflow
project into a mindmeld project"""

import os

from mindmeld.converter.rasaconverter import RasaConverter


current_file_loc = os.path.dirname(os.path.realpath(__file__))
# Location of the rasa project folder
RASA_PROJECT = current_file_loc + "/rasa_sample_project"
# Location of where you want to create mindmeld project folder
MINDMELD_PROJECT = current_file_loc + "/mindmeld_project"
converter = RasaConverter(RASA_PROJECT, MINDMELD_PROJECT)
converter.convert_project()
