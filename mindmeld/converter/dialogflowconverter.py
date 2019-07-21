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

"""This module contains the DialogFlowConverter class used to convert DialogFlow projects
into Mindmeld projects"""

import re
import os
import json
import logging

from converter import Converter  # from mindmeld.converter.converter import Converter

logger = logging.getLogger(__name__)


class DialogFlowConverter(Converter):
    def __init__(self, dialogflow_project_directory, mindmeld_project_directory):
        self.dialogflow_project_directory = dialogflow_project_directory
        self.mindmeld_project_directory = mindmeld_project_directory

    def create_mindmeld_directory(self):
        Converter.create_directory(self.mindmeld_project_directory)
        Converter.create_directory(os.path.join(self.mindmeld_project_directory, "data"))
        Converter.create_directory(os.path.join(self.mindmeld_project_directory, "domains"))
        Converter.create_directory(os.path.join(self.mindmeld_project_directory, "entities"))

    def __create_entities_directories(self, entities):
        """ Creates directories + files. Must have an english file for every entity.
            TODO: make this work for files not en, maybe different category in mm?
        """

        for entity in entities:
            dialogflow_entity_file = os.path.join(self.dialogflow_project_directory,
                                                 "entities", entity + "_entries_en.json")

            if os.path.exists(dialogflow_entity_file):
                mindmeld_entity_directory = os.path.join(self.mindmeld_project_directory,
                                                         "entities", entity)
                Converter.create_directory(mindmeld_entity_directory)
                DialogFlowConverter.__create_entity_file(dialogflow_entity_file,
                                                         mindmeld_entity_directory)
            else:
                logger.error("cannot find en entity file.")

    @staticmethod
    def __create_entity_file(dialogflow_entity_file, mindmeld_entity_directory):
        source_en = open(dialogflow_entity_file, 'r')
        target_gazetteer = open(os.path.join(mindmeld_entity_directory, "gazetteer.txt"), 'w')
        target_mapping = open(os.path.join(mindmeld_entity_directory, "mapping.json"), 'w')

        datastore = json.load(source_en)
        mapping_dict = {"entities": []}

        for item in datastore:
            newDict = {}
            item['synonyms'].remove(item['value'])
            newDict['whitelist'] = item['synonyms']
            newDict['cname'] = item['value']
            # newDict['id'] = "n/a" # TODO: when do you need an ID?
            mapping_dict["entities"].append(newDict)

            target_gazetteer.write(item['value'] + "\n")

        json.dump(mapping_dict, target_mapping, ensure_ascii=False, indent=2)

        source_en.close()
        target_gazetteer.close()
        target_mapping.close()

    def __read_entities(self):
        """ Gets the names of the entities from DialogFlow as a list"""
        dialogflow_entities_directory = os.path.join(self.dialogflow_project_directory,
                                                     "entities")
        dialogflow_entities_files = os.listdir(dialogflow_entities_directory)

        exp = r"^.*(?=(_entries_.+\.json))"  # matches everything before '_entries_??.json'
        # exp = '.*(_entries).*' # matches everything containing 'entries'

        entities = set()
        for name in dialogflow_entities_files:
            match = re.match(exp, name)
            if match:
                entities.add(match.group(0))

        return list(entities)

    def create_training_data(self):
        entities = self.__read_entities()
        self.__create_entities_directories(entities)

    def create_main(self):
        pass

    def create_init(self):
        pass

    def create_config(self):
        pass

    def convert_project(self):
        # Create project directory with sub folders
        self.create_mindmeld_directory()
        # Transfer over test data from Rasa project and reformat to Mindmeld project
        self.create_training_data()
        # self.create_main(self.mindmeld_project_directory)
        # self.create_init(self.mindmeld_project_directory)
        # self.create_config(self.mindmeld_project_directory)
