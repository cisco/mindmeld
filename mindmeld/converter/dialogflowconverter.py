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
from sklearn.model_selection import train_test_split

from converter import Converter  # from mindmeld.converter.converter import Converter

logger = logging.getLogger(__name__)


class DialogFlowConverter(Converter):
    sys_entity_map = {'@sys.date-time': 'sys_interval',
                      '@sys.date': 'sys_time',
                      '@sys.date-period': 'sys_interval',
                      '@sys.time': 'sys_time',
                      '@sys.time-period': 'sys_duration',
                      '@sys.duration': 'sys_duration',
                      '@sys.number': 'sys_number',
                      '@sys.cardinal': 'sys_number',
                      '@sys.ordinal': 'sys_ordinal',
                      '@sys.unit-currency': 'sys_amount-of-money',
                      '@sys.unit-volume': 'sys_volume',
                      '@sys.email': 'sys_email',
                      '@sys.phone-number': 'sys_phone-number',
                      '@sys.url': 'sys_url'}

    def __init__(self, dialogflow_project_directory, mindmeld_project_directory):
        self.dialogflow_project_directory = dialogflow_project_directory
        self.mindmeld_project_directory = mindmeld_project_directory
        self.directory = os.path.dirname(os.path.realpath(__file__))

    def create_mindmeld_directory(self):
        Converter.create_directory(self.mindmeld_project_directory)
        Converter.create_directory(os.path.join(self.mindmeld_project_directory, "data"))
        Converter.create_directory(os.path.join(self.mindmeld_project_directory, "domains"))
        Converter.create_directory(os.path.join(self.mindmeld_project_directory, "domains", "all"))
        Converter.create_directory(os.path.join(self.mindmeld_project_directory, "entities"))

    def _create_entities_directories(self, entities):
        """ Creates directories + files for all languages/files. All file paths should be valid.
        TODO: consider main files"""

        for main, languages in entities.items():
            for language, sub in languages.items():
                dialogflow_entity_file = os.path.join(self.dialogflow_project_directory,
                                                      "entities", sub + ".json")

                mindmeld_entity_directory = os.path.join(self.mindmeld_project_directory,
                                                         "entities", main)

                Converter.create_directory(mindmeld_entity_directory)

                DialogFlowConverter._create_entity_file(dialogflow_entity_file,
                                                        mindmeld_entity_directory)

    @staticmethod
    def _create_entity_file(dialogflow_entity_file, mindmeld_entity_directory):
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

    def _create_intents_directories(self, intents):
        """ Creates directories + files for all languages/files. All file paths should be valid.
        TODO: consider main files"""

        for main, languages in intents.items():
            for language, sub in languages.items():
                dialogflow_intent_file = os.path.join(self.dialogflow_project_directory,
                                                      "intents", sub + ".json")

                mindmeld_intent_directory = os.path.join(self.mindmeld_project_directory,
                                                         "domains", "all", sub)

                Converter.create_directory(mindmeld_intent_directory)

                DialogFlowConverter._create_intent_file(dialogflow_intent_file,
                                                        mindmeld_intent_directory)

    @staticmethod
    def _create_intent_file(dialogflow_intent_file, mindmeld_intent_directory):
        source_en = open(dialogflow_intent_file, 'r')
        target_test = open(os.path.join(mindmeld_intent_directory, "test.txt"), 'w')
        target_train = open(os.path.join(mindmeld_intent_directory, "train.txt"), 'w')

        datastore = json.load(source_en)
        allText = []

        for usersay in datastore:
            sentence = ""
            for texts in usersay["data"]:
                df_text = texts["text"]
                if "meta" in texts and texts["meta"] != "@sys.ignore":
                    df_meta = texts["meta"]

                    if re.match("(@sys.).+", df_meta):  # if text is a dialogflow sys entity
                        if df_meta in DialogFlowConverter.sys_entity_map:
                            mm_meta = DialogFlowConverter.sys_entity_map[df_meta]
                        else:
                            mm_meta = "[DNE: " + df_meta[1:] + "]"
                            logger.info("Unfortunately mindmeld does not currently support"
                                        + df_meta[1:] + "as a sys entity."
                                        + "Please create an entity for this.")

                        part = "{" + df_text + "|" + mm_meta + "}"
                    else:
                        part = "{" + df_text + "|" + df_meta[1:] + "}"
                else:
                    part = df_text

                sentence += part
            allText.append(sentence)

        train, test = train_test_split(allText, test_size=0.2)

        target_test.write("\n".join(test))
        target_train.write("\n".join(train))

        source_en.close()
        target_test.close()
        target_train.close()

    def _get_file_names(self, type):
        """ Gets the names of the entities from DialogFlow as a dictionary.
        ex. if we had the following files in our entities directory:
            ["test.json", "test_entries_en.json", "test_entries_de.json"]
        return:
            {'test': {'en': 'test_entries_en', 'de': 'test_entries_de'}} """

        dir = os.path.join(self.dialogflow_project_directory, type)
        files = os.listdir(dir)

        w = {"entities": "entries", "intents": "usersays"}
        p = r".+(?<=(_" + w[type] + "_))(.*)(?=(.json))"

        info = {}
        for name in files:
            match = re.match(p, name)

            if match:
                isbase = False
                base = name[:match.start(1)]
                language = match.group(2)
            else:
                isbase = True
                base = name[:-5]

            if base not in info:
                info[base] = {}

            if not isbase:
                info[base][language] = name[:-5]

        return info

    def create_training_data(self):
        entities = self._get_file_names("entities")
        self._create_entities_directories(entities)

        intents = self._get_file_names("intents")
        self._create_intents_directories(intents)

    # ^ create training data

    def create_handle(params):
        return "@app.handle(" + params + ")"

    def create_header(function_name):
        return "def " + function_name + "(request, responder):"

    def create_function(handles, function_name, replies):
        assert type(handles) == list

        result = ""
        for handle in handles:
            result += DialogFlowConverter.create_handle(handle) + "\n"
        result += DialogFlowConverter.create_header(function_name) + "\n"
        result += "\t" + "replies = {}".format(replies) + "\n"
        result += "\t" "responder.reply(replies)"
        return result

    def create_init(self):
        with open(os.path.join(self.mindmeld_project_directory, "__init__.py"), 'w') as target:
            begin_info = ["\"\"\"This module contains the MindMeld application\"\"\"",
                          "from mindmeld import Application",
                          "app = Application(__name__)",
                          "__all__ = ['app']"]

            for info in begin_info:
                target.write(info + "\n\n")

            intents = self._get_file_names("intents")

            # iterate over all the intents
            for i, (main, languages) in enumerate(intents.items()):
                df_main = os.path.join(self.dialogflow_project_directory,
                                       "intents", main + ".json")

                with open(df_main) as source:
                    datastore = json.load(source)

                    replies = []
                    for response in datastore["responses"]:
                        for message in response["messages"]:
                            data = message["speech"]
                            replies = data if type(data) == list else [data]

                            if datastore["fallbackIntent"]:
                                function_name = "default"
                                handles = ["default=True", "intent='unsupported'"]
                            else:
                                function_name = "renameMe" + str(i)
                                handles = ["intent=" + "'" + datastore["name"] + "''"]

                            target.write(DialogFlowConverter.create_function(function_name,
                                                                             handles,
                                                                             replies) + "\n\n")

    def create_main(self):
        pass

    def create_config(self):
        pass

    def convert_project(self):
        # Create project directory with sub folders
        self.create_mindmeld_directory()
        # Transfer over test data from DialogFlow project and reformat to Mindmeld project
        self.create_training_data()

        # self.create_main()
        # self.create_config()

        self.create_init()
