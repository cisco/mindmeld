# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains the DialogFlowconverter class used to convert DialogFlow projects
into Mindmeld projects"""

from keyword import iskeyword
import re, os, yaml, copy, json
from converter import Converter

class DialogFlowconverter(Converter):
    def __init__(self, dialogflow_project_directory, mindmeld_project_directory):
        self.dialogflow_project_directory = dialogflow_project_directory
        self.mindmeld_project_directory = mindmeld_project_directory

    @staticmethod
    def create_mindmeld_directory(mindmeld_project_path):
        Converter.create_directory(mindmeld_project_path)
        Converter.create_directory(mindmeld_project_path + "/data") #dialog flow doesn't have a knowledge base
        Converter.create_directory(mindmeld_project_path + "/domains")
        Converter.create_directory(mindmeld_project_path + "/entities")

    @staticmethod
    def __create_entities_directories(dialogflow_project_directory, mindmeld_project_directory, entities):
        """ Creates directories + files for english entities. """
        for entity in entities:
            dialogflow_entity_file = dialogflow_project_directory + "/entities/" + entity + "_entries_en.json" #TODO: make this work for non-english, multiple entries

            if os.path.exists(dialogflow_entity_file):
                mindmeld_entity_directory = mindmeld_project_directory + "/entities/" + entity
                Converter.create_directory(mindmeld_entity_directory)
                DialogFlowconverter.__create_entity_file(dialogflow_entity_file, mindmeld_entity_directory, entity)
            else:
                print("cannot find en entity file.")

    @staticmethod
    def __create_entity_file(dialogflow_entity_file, mindmeld_entity_directory, entity):
        source_en        = open(dialogflow_entity_file, 'r')
        target_gazetteer = open(mindmeld_entity_directory + "/gazetteer.txt"  , 'w')
        target_mapping   = open(mindmeld_entity_directory + "/mapping.json"   , 'w')

        datastore = json.load(source_en)
        mapping_dict = {"entities" : []}

        for item in datastore:
            newDict = {}
            item['synonyms'].remove(item['value'])
            newDict['whitelist'] = item['synonyms']
            newDict['cname'] = item['value']
            #newDict['id'] = "n/a" #TODO: when do you need an ID?
            mapping_dict["entities"].append(newDict)

            target_gazetteer.write(item['value'] + "\n")

        json.dump(mapping_dict, target_mapping, ensure_ascii=False, indent=2)

        source_en.close()
        target_gazetteer.close()
        target_mapping.close()

    @staticmethod
    def __read_entities(self):
        """ Gets the names of the entities from DialogFlow as a list"""
        dialogflow_entities_directory = os.path.join(self.dialogflow_project_directory, "entities")
        dialogflow_entities_files = os.listdir(dialogflow_entities_directory)

        exp = '^.*(?=(_entries_.+\.json))' # matches everything before '_entries_??.json'
        #exp = '.*(_entries).*' #matches everything containing 'entries'

        entities = set()
        for name in dialogflow_entities_files:
            match = re.match(exp, name)
            if match:
                entities.add(match.group(0))

        return list(entities)

    def create_training_data(self, dialogflow_project_directory, mindmeld_project_directory):
        entities = self.__read_entities(self)
        DialogFlowconverter.__create_entities_directories(dialogflow_project_directory, mindmeld_project_directory, entities)

    def create_main(self):
        pass

    def create_init(self):
        pass

    def create_config(self):
        pass

    def convert_project(self):
        # Create project directory with sub folders
        DialogFlowconverter.create_mindmeld_directory(self.mindmeld_project_directory)
        # Transfer over test data from Rasa project and reformat to Mindmeld project

        self.create_training_data(self.dialogflow_project_directory, self.mindmeld_project_directory)
        # self.create_main(self.mindmeld_project_directory)
        # self.create_init(self.mindmeld_project_directory)
        # self.create_config(self.mindmeld_project_directory)
