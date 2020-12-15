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
into MindMeld projects"""

import json
import logging
import os
import re
import importlib.util

from shutil import copyfile
from mindmeld.converter.converter import Converter
from mindmeld.converter.code_generator import MindmeldCodeGenerator
from mindmeld.components._config import DEFAULT_INTENT_CLASSIFIER_CONFIG

logger = logging.getLogger(__name__)
package_dir = os.path.dirname(os.path.abspath(__file__))


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
        "@sys.temperature": "sys_temperature",
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

    def __init__(
        self,
        dialogflow_project_directory,
        mindmeld_project_directory,
        custom_config_file_path=None,
        language="en",
    ):
        if os.path.exists(os.path.dirname(dialogflow_project_directory)):
            self.dialogflow_project_directory = dialogflow_project_directory
            self.mindmeld_project_directory = mindmeld_project_directory
            self.directory = os.path.dirname(os.path.realpath(__file__))
            self.entities_list = set()
            self.intents_list = set()
            self.code_gen = MindmeldCodeGenerator()
            self.custom_config_file_path = custom_config_file_path
            self.language = language
        else:
            msg = "`{dialogflow_project_directory}` does not exist. Please verify."
            msg = msg.format(dialogflow_project_directory=dialogflow_project_directory)
            raise FileNotFoundError(msg)

    def create_mindmeld_directory(self):
        self.create_directory(self.mindmeld_project_directory)
        self.create_directory(os.path.join(self.mindmeld_project_directory, "data"))
        self.create_directory(os.path.join(self.mindmeld_project_directory, "domains"))
        self.create_directory(
            os.path.join(self.mindmeld_project_directory, "domains", "app_specific")
        )
        self.create_directory(
            os.path.join(self.mindmeld_project_directory, "domains", "unrelated")
        )
        self.create_directory(os.path.join(self.mindmeld_project_directory, "entities"))

    # =========================
    # create training data (entities, intents)
    # =========================

    def _create_entities_directories(self, entities):
        """Creates directories + files for all languages/files.
        Currently does not use meta data in entityName.json files (the keys in var entities).
        """
        for languages in entities.values():
            for sub in languages.values():

                if sub != self.language:
                    # Each MindMeld app works on one language
                    continue

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

                # remove DF entity reference "entries"
                mindmeld_entity_directory = mindmeld_entity_directory.replace(
                    "entries_", ""
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

                if language != self.language:
                    # Each MindMeld app works on one language
                    continue

                dialogflow_intent_file = os.path.join(
                    self.dialogflow_project_directory, "intents", sub + ".json"
                )

                mindmeld_intent_directory_name = self.clean_check(
                    sub, self.intents_list
                )

                # DF has "default" intents like "default_fallback" and "default_greeting"
                # which are in-built intents. We map these intents to the "unrelated" domain
                # compared to the other app specific intents being mapped to the "app_specific"
                # domain.
                if "default" in mindmeld_intent_directory_name:
                    domain = "unrelated"
                else:
                    domain = "app_specific"

                mindmeld_intent_directory = os.path.join(
                    self.mindmeld_project_directory,
                    "domains",
                    domain,
                    mindmeld_intent_directory_name,
                )

                # remove DF intent reference "usersays_"
                mindmeld_intent_directory = mindmeld_intent_directory.replace(
                    "usersays_", ""
                )
                self.create_directory(mindmeld_intent_directory)
                self._create_intent_file(
                    dialogflow_intent_file, mindmeld_intent_directory, language
                )

    def _create_intent_file(
        self, dialogflow_intent_file, mindmeld_intent_directory, language
    ):
        source_en = open(dialogflow_intent_file, "r")
        target_train = open(os.path.join(mindmeld_intent_directory, "train.txt"), "w")
        datastore = json.load(source_en)
        all_text = []
        default_intent_to_training_file = {
            "default_fallback_intent": "unrelated.txt",
            "default_welcome_intent": "greetings.txt",
        }

        for usersay in datastore:
            sentence = ""
            for texts in usersay["data"]:
                df_text = texts["text"]
                if "meta" in texts and texts["meta"] != "@sys.ignore":
                    df_meta = texts["meta"]
                    role_type = texts["alias"].replace("-", "_")

                    if re.match(
                        "(@sys.).+", df_meta
                    ):  # if text is a dialogflow sys entity
                        if df_meta in DialogflowConverter.sys_entity_map:
                            mm_meta = DialogflowConverter.sys_entity_map[df_meta]
                            entity_type = mm_meta
                        else:
                            mm_meta = "[DNE: {sysEntity}]".format(sysEntity=df_meta[1:])
                            logger.info(
                                "Unfortunately mindmeld does not currently support"
                                "%s as a sys entity."
                                "Please create an entity for this.",
                                df_meta[1:],
                            )
                            entity_type = self.clean_name(mm_meta) + "_" + language

                        part = "{" + df_text + "|" + entity_type + "|" + role_type + "}"
                    else:
                        entity_type = self.clean_name(df_meta[1:]) + "_" + language
                        part = "{" + df_text + "|" + entity_type + "|" + role_type + "}"
                else:
                    part = df_text

                sentence += part
            all_text.append(sentence)

        for key in default_intent_to_training_file:
            if key in mindmeld_intent_directory:
                with open(
                    os.path.join(package_dir, default_intent_to_training_file[key])
                ) as fp:
                    for line in fp:
                        all_text.append(line.strip())

        # Double the size of the training set if there are less than the number of
        # folds for cross-val in the config.py file
        intent_config = DEFAULT_INTENT_CLASSIFIER_CONFIG
        if self.custom_config_file_path:
            config_path = os.path.join(self.mindmeld_project_directory, "config.py")
            spec = importlib.util.spec_from_file_location("mindmeld_app", config_path)
            config = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(config)
            intent_config = getattr(config, "INTENT_RECOGNIZER_CONFIG", intent_config)

        while len(all_text) < intent_config["param_selection"]["k"]:
            all_text = all_text * 2

        target_train.write("\n".join(all_text))
        source_en.close()
        target_train.close()

    def _get_file_names(self, level):
        """Gets the names of the entities from Dialogflow as a dictionary.
        levels (str): either "entities" or "intents"

        ex. if we had the following files in our entities directory:
            ["test.json", "test_entries_en.json", "test_entries_de.json"]
        it returns:
            {'test': {'en': 'test_entries_en', 'de': 'test_entries_de'}}"""

        directory = os.path.join(self.dialogflow_project_directory, level)
        files = os.listdir(directory)

        w = {"entities": "entries", "intents": "usersays"}
        p = r".+(?<=(_" + w[level] + "_))(.*)(?=(.json))"
        language = "en"

        info = {}
        for name in files:
            match = re.match(p, name)

            if match:
                isbase = False
                base = name[: match.start(1)]
                language = str(match.group(2))
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
    def clean_name(name):
        """ Takes in a string and returns a valid folder name (no spaces, all lowercase)."""
        name = re.sub(r"[^\w\s-]", "", name).strip().lower()
        name = re.sub(r"[-\s]+", "_", name)
        return name

    def clean_check(self, name, lst):
        """Takes in a list of strings and a name.
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

            self.code_gen.begin(tab="    ")
            self.code_gen.generate_top_block()

            intents = self._get_file_names("intents")

            for main in intents:

                df_main = os.path.join(
                    self.dialogflow_project_directory, "intents", main + ".json"
                )

                with open(df_main) as source:
                    if "usersays" in df_main:
                        logger.error(
                            "Please check if your intent file"
                            "names are correctly labeled."
                        )
                        return

                    datastore = json.load(source)
                    intent = self.clean_name(datastore["name"])
                    for response in datastore["responses"]:
                        self.generate_handlers(intent, response)

            target.write(self.code_gen.end())
            target.write("\n")

    def generate_handlers(self, intent, response):
        message = response["messages"][0]
        language = message["lang"]
        intent_lang = "%s_%s" % (intent, language)
        intent_entity_role_replies = {intent_lang: {}}

        for param in response["parameters"]:
            if param["required"]:
                entity = param["dataType"]
                if entity in DialogflowConverter.sys_entity_map:
                    entity = DialogflowConverter.sys_entity_map[entity]
                else:
                    entity = param["dataType"].replace("@", "").replace("-", "_")
                    entity = "%s_%s" % (entity, language)
                role = param["name"].replace("@", "").replace("-", "_")

                prompts = []
                if "prompts" in param:
                    prompts = [x["value"] for x in param["prompts"]]
                else:
                    prompts = ["What is the " + param["name"]]

                if entity in intent_entity_role_replies[intent_lang]:
                    intent_entity_role_replies[intent_lang][entity][role] = prompts
                else:
                    intent_entity_role_replies[intent_lang][entity] = {role: prompts}

        if "speech" in message:
            data = message["speech"]
            replies = data if isinstance(data, list) else [data]
            slot_templated_replies = []

            is_slot_template = False
            for resp in replies:
                template = resp
                slots = re.findall("\$([\w\-\_]+)", resp)
                for slot in slots:
                    template = template.replace(
                        "$" + slot, "{" + slot.replace("-", "_") + "}"
                    )
                if template != resp:
                    is_slot_template = True
                slot_templated_replies.append(template)

            handle = "intent='%s_%s'" % (intent, language)
            function_name = intent + "_" + language + "_handler"

            if is_slot_template:
                self.code_gen.generate_followup_function_code_block(
                    handle,
                    function_name,
                    intent_entity_role_replies,
                    slot_templated_replies,
                )
            else:
                self.code_gen.generate_function(
                    handle=handle,
                    function_name=function_name,
                    replies=replies,
                )

    # =========================
    # convert project
    # =========================

    def convert_project(self):
        """Converts a Dialogflow project into a MindMeld project.

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
                entityName_la.json - One for each language, contains entitiy mappings.

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

        MindMeld:
        - Users can store data locally
        - Users can build a knowledge base (currently beta in Dialogflow).
        - Users can configure the machine learning models to best suit their needs.
        - Users have more flexibility in defining their own features, including
         ones like slot filling, contexts, and follow-up intents.
        """

        logger.info("Converting project.")

        # Create project directory with sub folders
        self.create_mindmeld_directory()

        # copy config file to the MindMeld dir
        if self.custom_config_file_path:
            copyfile(
                self.custom_config_file_path,
                os.path.join(self.mindmeld_project_directory, "config.py"),
            )

        file_loc = os.path.dirname(os.path.realpath(__file__))
        self.create_main(self.mindmeld_project_directory, file_loc)
        self.create_mindmeld_init()

        # Transfer over test data from Dialogflow project and reformat to MindMeld project
        self.create_mindmeld_training_data()
        logger.info("Project converted.")
