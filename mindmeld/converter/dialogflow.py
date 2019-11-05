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

"""This module contains the DialogflowConverter class used to convert Dialogflow projects
into Mindmeld projects"""

import json
import logging
import os
import re

from sklearn.model_selection import train_test_split

from mindmeld.converter.converter import Converter

logger = logging.getLogger(__name__)


class DialogflowConverter(Converter):
    """The class is a sub class of the abstract Converter class. This class
    contains the methods required to convert a Dialogflow project into a MindMeld project
    """

    sys_entity_map = {
        "@sys.date-time": "sys_interval",
        "@sys.date": "sys_time",
        "@sys.date-period": "sys_interval",
        "@sys.time": "sys_time",
        "@sys.time-period": "sys_duration",
        "@sys.duration": "sys_duration",
        "@sys.number": "sys_number",
        "@sys.cardinal": "sys_number",
        "@sys.ordinal": "sys_ordinal",
        "@sys.unit-currency": "sys_amount-of-money",
        "@sys.unit-volume": "sys_volume",
        "@sys.email": "sys_email",
        "@sys.phone-number": "sys_phone-number",
        "@sys.url": "sys_url",
    }

    # TODO: provide support for entities listed in sys_entity_map_todo
    sys_entity_map_todo = [
        "@sys.number-integer",
        "@sys.number-sequence",
        "@sys.flight-number",
        "@sys.unit-area",
        "@sys.unit-length",
        "@sys.unit-speed",
        "@sys.unit-information",
        "@sys.percentage",
        "@sys.temperature",
        "@sys.duration",
        "@sys.age",
        "@sys.currency-name",
        "@sys.unit-area-name",
        "@sys.unit-length-name",
        "@sys.unit-speed-name",
        "@sys.unit-volume-name",
        "@sys.unit-weight-name",
        "@sys.unit-information-name",
        "@sys.address",
        "@sys.zip-code",
        "@sys.geo-capital",
        "@sys.geo-country",
        "@sys.geo-country-code",
        "@sys.geo-city",
        "@sys.geo-state",
        "@sys.geo-city",
        "@sys.geo-state",
        "@sys.place-attraction",
        "@sys.airport",
        "@sys.location",
        "@sys.given-name",
        "@sys.last-name",
        "@sys.person",
        "@sys.music-artist",
        "@sys.music-genre",
        "@sys.color",
        "@sys.language",
        "@sys.any",
    ]

    def __init__(self, dialogflow_project_directory, mindmeld_project_directory):
        if os.path.exists(os.path.dirname(dialogflow_project_directory)):
            self.dialogflow_project_directory = dialogflow_project_directory
            self.mindmeld_project_directory = mindmeld_project_directory
            self.directory = os.path.dirname(os.path.realpath(__file__))
            self.entities_list = set()
            self.intents_list = set()
        else:
            msg = "`{dialogflow_project_directory}` does not exist. Please verify."
            msg = msg.format(dialogflow_project_directory=dialogflow_project_directory)
            raise FileNotFoundError(msg)

    def create_mindmeld_directory(self):
        self.create_directory(self.mindmeld_project_directory)
        self.create_directory(os.path.join(self.mindmeld_project_directory, "data"))
        self.create_directory(os.path.join(self.mindmeld_project_directory, "domains"))
        self.create_directory(
            os.path.join(self.mindmeld_project_directory, "domains", "general")
        )
        self.create_directory(os.path.join(self.mindmeld_project_directory, "entities"))

    # =========================
    # create training data (entities, intents)
    # =========================

    def _create_entities_directories(self, entities):
        """ Creates directories + files for all languages/files.
        Currently does not use meta data in entityName.json files (the keys in var entities).
        """
        for languages in entities.values():
            for sub in languages.values():
                dialogflow_entity_file = os.path.join(
                    self.dialogflow_project_directory, "entities", sub + ".json"
                )

                mindmeld_entity_directory_name = self.clean_check(
                    sub, self.entities_list
                )

                mindmeld_entity_directory = os.path.join(
                    self.mindmeld_project_directory,
                    "entities",
                    mindmeld_entity_directory_name,
                )

                self.create_directory(mindmeld_entity_directory)

                self._create_entity_file(
                    dialogflow_entity_file, mindmeld_entity_directory
                )

    @staticmethod
    def _create_entity_file(dialogflow_entity_file, mindmeld_entity_directory):
        source_en = open(dialogflow_entity_file, "r")
        target_gazetteer = open(
            os.path.join(mindmeld_entity_directory, "gazetteer.txt"), "w"
        )
        target_mapping = open(
            os.path.join(mindmeld_entity_directory, "mapping.json"), "w"
        )

        datastore = json.load(source_en)
        mapping_dict = {"entities": []}

        for item in datastore:
            new_dict = {}
            while ("value" in item) and (item["value"] in item["synonyms"]):
                item["synonyms"].remove(item["value"])
            new_dict["whitelist"] = item["synonyms"]
            new_dict["cname"] = item["value"]
            mapping_dict["entities"].append(new_dict)

            target_gazetteer.write(item["value"] + "\n")

        json.dump(mapping_dict, target_mapping, ensure_ascii=False, indent=2)

        source_en.close()
        target_gazetteer.close()
        target_mapping.close()

    def _create_intents_directories(self, intents):
        """ Creates directories + files for all languages/files."""

        for languages in intents.values():
            for language, sub in languages.items():
                dialogflow_intent_file = os.path.join(
                    self.dialogflow_project_directory, "intents", sub + ".json"
                )

                mindmeld_intent_directory_name = self.clean_check(
                    sub, self.intents_list
                )
                mindmeld_intent_directory = os.path.join(
                    self.mindmeld_project_directory,
                    "domains",
                    "general",
                    mindmeld_intent_directory_name,
                )

                self.create_directory(mindmeld_intent_directory)

                self._create_intent_file(
                    dialogflow_intent_file, mindmeld_intent_directory, language
                )

    def _create_intent_file(
        self, dialogflow_intent_file, mindmeld_intent_directory, language
    ):
        source_en = open(dialogflow_intent_file, "r")
        target_test = open(os.path.join(mindmeld_intent_directory, "test.txt"), "w")
        target_train = open(os.path.join(mindmeld_intent_directory, "train.txt"), "w")

        datastore = json.load(source_en)
        all_text = []

        for usersay in datastore:
            sentence = ""
            for texts in usersay["data"]:
                df_text = texts["text"]
                if "meta" in texts and texts["meta"] != "@sys.ignore":
                    df_meta = texts["meta"]

                    if re.match(
                        "(@sys.).+", df_meta
                    ):  # if text is a dialogflow sys entity
                        if df_meta in DialogflowConverter.sys_entity_map:
                            mm_meta = DialogflowConverter.sys_entity_map[df_meta]
                        else:
                            mm_meta = "[DNE: {sysEntity}]".format(sysEntity=df_meta[1:])
                            logger.info(
                                "Unfortunately mindmeld does not currently support"
                                "%s as a sys entity."
                                "Please create an entity for this.",
                                df_meta[1:],
                            )

                        entity_type = self.clean_name(mm_meta) + "_entries_" + language
                        part = "{" + df_text + "|" + entity_type + "}"
                    else:
                        entity_type = (
                            self.clean_name(df_meta[1:]) + "_entries_" + language
                        )
                        part = "{" + df_text + "|" + entity_type + "}"
                else:
                    part = df_text

                sentence += part
            all_text.append(sentence)

        train, test = train_test_split(all_text, test_size=0.2)

        target_test.write("\n".join(test))
        target_train.write("\n".join(train))

        source_en.close()
        target_test.close()
        target_train.close()

    def _get_file_names(self, level):
        """ Gets the names of the entities from Dialogflow as a dictionary.
        levels (str): either "entities" or "intents"

        ex. if we had the following files in our entities directory:
            ["test.json", "test_entries_en.json", "test_entries_de.json"]
        it returns:
            {'test': {'en': 'test_entries_en', 'de': 'test_entries_de'}} """

        directory = os.path.join(self.dialogflow_project_directory, level)
        files = os.listdir(directory)

        w = {"entities": "entries", "intents": "usersays"}
        p = r".+(?<=(_" + w[level] + "_))(.*)(?=(.json))"

        info = {}
        for name in files:
            match = re.match(p, name)

            if match:
                isbase = False
                base = name[: match.start(1)]
                language = match.group(2)
            else:
                isbase = True
                base = name[:-5]

            if base not in info:
                info[base] = {}

            if not isbase:
                info[base][language] = name[:-5]

        return info

    def create_mindmeld_training_data(self):
        entities = self._get_file_names("entities")
        self._create_entities_directories(entities)

        intents = self._get_file_names("intents")
        self._create_intents_directories(intents)

    # =========================
    # create init
    # =========================

    @staticmethod
    def create_handle(params):
        return "@app.handle(" + params + ")"

    @staticmethod
    def create_header(function_name):
        return "def " + function_name + "(request, responder):"

    @staticmethod
    def create_function(handles, function_name, replies):
        assert isinstance(handles, list)

        result = ""
        for handle in handles:
            result += DialogflowConverter.create_handle(handle) + "\n"
        result += DialogflowConverter.create_header(function_name) + "\n"
        result += "    " + "replies = {}".format(replies) + "\n"
        result += "    " + "responder.reply(replies)"
        return result

    @staticmethod
    def clean_name(name):
        """ Takes in a string and returns a valid folder name (no spaces, all lowercase)."""
        name = re.sub(r"[^\w\s-]", "", name).strip().lower()
        name = re.sub(r"[-\s]+", "_", name)
        return name

    def clean_check(self, name, lst):
        """ Takes in a list of strings and a name.
        Returns name cleaned if the cleaned name is not found in lst."""
        cleaned = self.clean_name(name)

        if cleaned not in lst:
            lst.add(cleaned)
            return cleaned
        else:
            logger.error(
                "%s name has been created twice. Please ensure there "
                "are no duplicate names in the dialogflow files and "
                "filenames are valid (no spaces or special characters)",
                cleaned,
            )

    def create_mindmeld_init(self):
        with open(
            os.path.join(self.mindmeld_project_directory, "__init__.py"), "w"
        ) as target:
            begin_info = [
                "# -*- coding: utf-8 -*-",
                '"""This module contains the MindMeld application"""',
                "from mindmeld import Application",
                "app = Application(__name__)",
                "__all__ = ['app']",
            ]

            for info, spacing in zip(begin_info, [1, 2, 1, 1, 0]):
                target.write(info + "\n" * spacing)

            intents = self._get_file_names("intents")

            for i, main in enumerate(intents.keys()):

                df_main = os.path.join(
                    self.dialogflow_project_directory, "intents", main + ".json"
                )

                with open(df_main) as source:
                    if "usersays" in df_main:
                        logger.error(
                            "Please check if your intent file"
                            "names are correctly labeled."
                        )

                    datastore = json.load(source)
                    replies = []

                    for response in datastore["responses"]:
                        for message in response["messages"]:
                            language = message["lang"]

                            if "speech" in message:
                                data = message["speech"]

                                replies = data if isinstance(data, list) else [data]

                                if datastore["fallbackIntent"]:
                                    function_name = "default" + "_" + language
                                    if language == "en":
                                        # TODO: support multiple defaults for languages
                                        handles = [
                                            "default=True",
                                            "intent='unsupported'",
                                        ]
                                    else:
                                        handles = ["intent='unsupported'"]
                                else:
                                    function_name = "renameMe" + str(i) + "_" + language
                                    handles = [
                                        "intent="
                                        + "'"
                                        + self.clean_name(datastore["name"])
                                        + "_usersays_"
                                        + language
                                        + "'"
                                    ]

                                target.write(
                                    "\n\n\n"
                                    + self.create_function(
                                        handles=handles,
                                        function_name=function_name,
                                        replies=replies,
                                    )
                                )
            target.write("\n")

    # =========================
    # convert project
    # =========================

    def convert_project(self):
        """ Converts a Dialogflow project into a MindMeld project.

        Dialogflow projects consist of entities and intents.
            note on languages:
                Dialogflow supports multiple languages and locales. They store their training
                data for different languages in different files. So, the name of each training
                file ends with a meta tag, two letters long for language, and an additional
                two letters for dialect (if applicable). For example, a file ending in "_en-au"
                indicates it's in English (Australia). Below we use "la" to represent this
                meta tag.

            entities folder contains:
                entityName.json - Meta data about entityName for all languages.
                entityName_entries_la.json - One for each language, contains entitiy mappings.

            intents folder contain:
                intentName.json - Contains rules, information about conversation flow, meta data.
                    Contains previously mentioned information and responses for all languages.
                intentName_usersays_la.json - one for each language,
                    contains training data to recognize intentName

        Limitations:
        - The converter is unable to create an entity when it encounters an
        unrecognized entity (an entity not defined under entities folder
         or system entities), and labels such entities as DNE in training data.
        - The converter currently does not automatically convert features like
        slot filling, contexts, and follow-up intents. Users can still implement such
        features and more.
        - Information in agent.json are not copied over.
        - There is no official support for different languages. Users can still
        implement this. The converter is able to successfully convert dialogflow
        bots that support multiple languages.

        Mindmeld:
        - Users can store data locally
        - Users can build a knowledge base (currently beta in Dialogflow).
        - Users can configure the machine learning models to best suit their needs.
        - Users have more flexibility in defining their own features, including
         ones like slot filling, contexts, and follow-up intents.
        """

        logger.info("Converting project.")

        # Create project directory with sub folders
        self.create_mindmeld_directory()

        # Transfer over test data from Dialogflow project and reformat to Mindmeld project
        self.create_mindmeld_training_data()
        file_loc = os.path.dirname(os.path.realpath(__file__))

        self.create_config(self.mindmeld_project_directory, file_loc)
        self.create_main(self.mindmeld_project_directory, file_loc)
        self.create_mindmeld_init()

        logger.info("Project converted.")
