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

"""This module contains the Rasacoverter class used to convert Rasa projects
into Mindmeld projects"""

import copy
import logging
import os
import re
from keyword import iskeyword

import yaml

from mindmeld.converter.converter import Converter

logger = logging.getLogger(__name__)


class RasaConverter(Converter):
    """The class is a sub class of the abstract Converter class. This class
    contains the methods required to convert a Rasa project into a MindMeld project
    """

    def __init__(self, rasa_project_directory, mindmeld_project_directory):
        if os.path.exists(os.path.dirname(rasa_project_directory)):
            self.rasa_project_directory = rasa_project_directory
            self.mindmeld_project_directory = mindmeld_project_directory
        else:
            msg = "`{rasa_project_directory}` does not exist. Please verify.".format(
                rasa_project_directory=rasa_project_directory
            )
            raise FileNotFoundError(msg)

    def _create_intents_directories(self, mindmeld_project_directory, intents):
        """Note: Because Rasa does not support multiple domains at this time. All intents
        are placed under a domain named 'general'."""
        GENERAL_DOMAIN_LOCATION = "/domains/general/"
        for intent in intents:
            self.create_directory(
                mindmeld_project_directory + GENERAL_DOMAIN_LOCATION + intent
            )

    def _create_entities_directories(self, mindmeld_project_directory, entities):
        for entity in entities:
            entity_path = mindmeld_project_directory + "/entities/" + entity
            self.create_directory(entity_path)
            with open(entity_path + "/gazetteer.txt", "w") as f:
                f.close()
            with open(entity_path + "/mapping.json", "w") as f:
                # skeleton mapping file that a user must fill in
                f.write('{\n  "entities":[]\n}')
                f.close()

    @staticmethod
    def _is_line_intent_definiton(line):
        return line[0:10] == "## intent:"

    @staticmethod
    def _get_intent_from_line(line):
        return line.split(" ")[1].split(":")[1].rstrip()

    def _create_intent_training_file(self, intent_directory):
        self.create_directory(intent_directory)
        with open(intent_directory + "/train.txt", "w") as f:
            f.close()

    @staticmethod
    def _does_intent_ex_contain_entity(intent_example):
        return len(re.findall(r"\[.*\]\(.*\)", intent_example)) > 0

    def _write_intent_with_extinty(self, intent_f, intent_example):
        mindmend_intent_example = intent_example
        pattern = re.compile(r"\[\w*\]\(\w*\)")
        for match in pattern.findall(intent_example):
            mindmeld_entity = (
                match.replace("[", "{")
                .replace("]", "|")
                .replace("(", "")
                .replace(")", "}")
            )
            mindmend_intent_example = mindmend_intent_example.replace(
                match, mindmeld_entity
            )
            # add this to the respective entity gazetteer file as well
            self.create_entity_files(mindmeld_entity)
        intent_f.write(mindmend_intent_example)

    @staticmethod
    def _remove_comments_from_line(line):
        start_of_comment = line.find("<!---")
        end_of_comment = line.find("-->")
        line_without_comment = line.replace(
            line[start_of_comment : end_of_comment + 3], ""
        )
        line_without_comment = line_without_comment.rstrip()
        return line_without_comment

    def _add_example_to_training_file(self, current_intent_path, line):
        with open(current_intent_path + "/train.txt", "a") as intent_f:
            intent_example = line[2:]
            intent_example = (
                RasaConverter._remove_comments_from_line(intent_example) + "\n"
            )
            if RasaConverter._does_intent_ex_contain_entity(intent_example):
                self._write_intent_with_extinty(intent_f, intent_example)
            else:
                intent_f.write(intent_example)

    def _read_domain_file(self):
        for file_ending in ["yaml", "yml"]:
            file_name = self.rasa_project_directory + "/domain." + file_ending
            if os.path.isfile(file_name):
                try:
                    with open(file_name, "r") as stream:
                        domain_data_loaded = yaml.safe_load(stream)
                        return domain_data_loaded
                except IOError as e:
                    logger.error("Can not open domain.yml file at %s", file_name)
                    logger.error(e)
        logger.error("Could not find domain.yml file in project directory")
        raise FileNotFoundError

    def _read_entities(self):
        domain_file = self._read_domain_file()
        if "entities" in domain_file:
            return domain_file["entities"]
        else:
            return []

    def _read_slots(self):
        domain_file = self._read_domain_file()
        if "slots" in domain_file:
            return domain_file["slots"]
        else:
            return []

    def _read_intents(self):
        domain_file = self._read_domain_file()
        return domain_file["intents"]

    def _read_actions(self):
        domain_file = self._read_domain_file()
        return domain_file["actions"]

    def _read_templates(self):
        domain_file = self._read_domain_file()
        if "templates" in domain_file:
            return domain_file["templates"]
        else:
            return []

    def create_entity_files(self, mm_entry):
        entity = mm_entry.strip("{}").split("|")
        gazetteer_location = (
            self.mindmeld_project_directory
            + "/entities/"
            + entity[1]
            + "/gazetteer.txt"
        )
        try:
            with open(gazetteer_location, "a") as f:
                f.write(entity[0] + "\n")
                f.close()
        except FileNotFoundError as e:
            self._create_entities_directories(
                self.mindmeld_project_directory, [entity[1]]
            )
            with open(gazetteer_location, "a") as f:
                f.write(entity[0] + "\n")
                f.close()
            logger.error("Domain file may not contain entity %s", entity[1])
            logger.error(e)

    @staticmethod
    def _is_valid_function_name(name):
        return name.isidentifier() and not iskeyword(name)

    @staticmethod
    def _is_story_name(stories_line):
        return stories_line[0:3] == "## "

    def _get_story_name(self, stories_line):
        if "<!--" in stories_line:
            return self._remove_comments_from_line(
                stories_line.replace("## ", "")
            ).rstrip()
        else:
            return stories_line.replace("## ", "").rstrip()

    @staticmethod
    def _is_intent(stories_line):
        return stories_line[0:2] == "* "

    @staticmethod
    def _is_action(stories_line):
        return stories_line[0:3] == " - "

    @staticmethod
    def _does_intent_have_entity(stories_line):
        return len(re.findall(r"\{.*\}", stories_line)) > 0

    @staticmethod
    def _clean_up_entities_list(entities_with_values):
        # trim off { }
        entities_with_values = entities_with_values[1:-1]
        # split data entities if there are multiples and clean white space
        entities_list = entities_with_values.split(",")
        for i, entity in enumerate(entities_list):
            entities_list[i] = entity.replace('"', "")
            entities_list[i] = entities_list[i].lstrip()
        return entities_list

    def _get_intent_with_entity(self, stories_line):
        if RasaConverter._does_intent_have_entity(stories_line):
            entities_with_values = re.search(r"\{.*\}", stories_line)
            entities_with_values = entities_with_values.group(0)
            entities_list = self._clean_up_entities_list(entities_with_values)
            start_of_entity = stories_line.find(entities_with_values)
            intent = self._remove_comments_from_line(
                stories_line[2:start_of_entity]
            ).rstrip()
            return intent, entities_list
        else:
            intent = self._remove_comments_from_line(stories_line[2:]).rstrip()
            entities_list = []
            return intent, entities_list

    def _get_stories(self):
        if os.path.isfile(self.rasa_project_directory + "/data/stories.md"):
            try:
                with open(self.rasa_project_directory + "/data/stories.md", "r") as f:
                    stories_dictionary = {}
                    current_story_name = ""
                    steps = []
                    current_step = {}
                    current_intent = ""
                    current_actions = []
                    stories_lines = f.readlines()
                    max_lines = len(stories_lines)
                    for line_num, line in enumerate(stories_lines):
                        if self._is_story_name(line):
                            current_story_name = self._get_story_name(line)
                            continue

                        if self._is_intent(line):
                            (
                                current_intent,
                                current_entities,
                            ) = self._get_intent_with_entity(line)
                            current_step["intent"] = copy.deepcopy(current_intent)
                            current_step["entities"] = copy.deepcopy(current_entities)
                            continue

                        if self._is_action(line):
                            current_actions.append(
                                RasaConverter._remove_comments_from_line(
                                    line[3:]
                                ).rstrip()
                            )

                            if (
                                (line_num + 1) < max_lines
                            ) and RasaConverter._is_action(stories_lines[line_num + 1]):
                                continue

                            current_step["actions"] = copy.deepcopy(current_actions)
                            current_actions.clear()
                            steps.append(copy.deepcopy(current_step))
                            current_step.clear()
                        elif len(line.strip()) == 0:
                            if current_story_name != "":
                                stories_dictionary[current_story_name] = copy.deepcopy(
                                    steps
                                )
                                steps.clear()
                                current_story_name = ""
                        if line_num == (max_lines - 1):
                            stories_dictionary[current_story_name] = copy.deepcopy(
                                steps
                            )
                            steps.clear()
                            current_story_name = ""
                    f.close()
                    return stories_dictionary
            except IOError as e:
                logger.error(
                    "Can not open stories.md file at %s",
                    self.rasa_project_directory + "/data/stories.md",
                )
                logger.error(e)
        else:
            logger.error(
                "Could not find stories.md file in %s",
                self.rasa_project_directory + "/data/stories.md",
            )
            raise FileNotFoundError

    def create_mindmeld_directory(self, mindmeld_project_path):
        self.create_directory(mindmeld_project_path)
        self.create_directory(mindmeld_project_path + "/data")
        self.create_directory(mindmeld_project_path + "/domains")
        self.create_directory(mindmeld_project_path + "/domains/general")
        self.create_directory(mindmeld_project_path + "/entities")

    def create_mindmeld_training_data(self):
        """Method to transfer and reformat the training data in a Rasa Project
        """
        # read intents listed in domain.yml
        intents = self._read_intents()

        # create intents subdirectories
        self._create_intents_directories(self.mindmeld_project_directory, intents)

        # read entities in domain.yml
        entities = self._read_entities()

        # create entities subdirectories if entities is not empty
        if entities:
            self._create_entities_directories(self.mindmeld_project_directory, entities)

        # try and open data files from rasa project
        nlu_data_loc = self.rasa_project_directory + "/data/nlu_data.md"
        try:
            with open(nlu_data_loc, "r") as nlu_data_md_file:
                nlu_data_lines = nlu_data_md_file.readlines()
        except FileNotFoundError:
            logger.error("Cannot open nlu_data.md file at %s", nlu_data_loc)
        # iterate through each line
        current_intent = ""
        current_intent_path = ""
        for line in nlu_data_lines:
            if self._is_line_intent_definiton(line):
                current_intent = RasaConverter._get_intent_from_line(line)
                current_intent_path = (
                    self.mindmeld_project_directory
                    + "/domains/general/"
                    + current_intent
                )
                # create data text file for intent examples`
                self._create_intent_training_file(current_intent_path)
            else:
                if line[0] == "-":
                    self._add_example_to_training_file(current_intent_path, line)

    def _write_init_header(self):
        string = """from mindmeld import Application
from . import custom_features  # noqa: F401

app = Application(__name__)

__all__ = ['app']


"""
        f = open(self.mindmeld_project_directory + "/__init__.py", "w+")
        f.write(string)
        return f

    @staticmethod
    def _get_app_handle(intent, entities):
        has_entity_string = ", has_entity="
        has_entities_string = ", has_entities=["
        entities_string = ""
        if len(entities) > 1:
            entities_string = has_entities_string
            for entity_value in entities:
                entity_string = entity_value.split(":")[0]
                if entity_value == entities[-1]:
                    entities_string += "'" + entity_string + "']"
                else:
                    entities_string += "'" + entity_string + "', "
        elif len(entities) == 1:
            for entity_value in entities:
                entity_string = entity_value.split(":")[0]
                entities_string += has_entity_string + "'" + entity_string + "'"
        handle_string = "@app.handle(intent='" + intent + "'" + entities_string + ")\n"
        return handle_string

    def _write_function_declaration(self, action, f):
        if self._is_valid_function_name(action):
            function_declartion_string = "def {}(request, responder):\n".format(action)
            f.write(function_declartion_string)
        else:
            logger.error("Action {action} is not a valid name for a python function")
            raise SyntaxError

    @staticmethod
    def _write_function_body_prompt(prompts, f):
        entities_list = []
        prompts_list = []
        # check if prompts contain any entities
        for prompt in prompts:
            entities = re.findall(r"\{.*\}", prompt)
            entities_list = []
            newprompt = prompt
            for i, entity in enumerate(entities, start=0):
                newprompt = prompt.replace(entity, "{" + str(i) + "}")
                entities_list.append(entity.replace("{", "").replace("}", ""))
            entities_args = ", ".join(map(str, entities_list))
            prompts_list.append('"' + newprompt + '".format({})'.format(entities_args))
            for entity in entities_list:
                newentity = entity.replace("{", "").replace("}", "")
                entities_string = "    {}_s = [e['text'] for e in ".format(
                    newentity
                ) + "request.entities if e['type'] == '{}']\n".format(newentity)
                entity_string = "    {0} = {0}_s[0]\n".format(newentity)
                f.write(entities_string)
                f.write(entity_string)
        prompts_string = "    prompts = [{}]\n".format(", ".join(prompts_list))
        f.write(prompts_string)

    @staticmethod
    def _write_default_function():
        pass

    @staticmethod
    def _get_text_prompts_list(action_templates):
        prompts = []
        for template in action_templates:
            if "text" in template:
                prompts.append(template["text"])
        return prompts

    @staticmethod
    def _write_responder_lines(f):
        responder_string = "    responder.reply(prompts)\n    responder.listen()\n"
        f.write(responder_string)

    def _read_file_lines(self):
        with open(self.mindmeld_project_directory + "/__init__.py", "r+") as f:
            return f.readlines()

    def _write_functions(self, actions, templates, f):
        for action in actions:
            self._write_function_declaration(action, f)
            if action in templates:
                # Get list of templates per action
                action_templates = templates[action]
                prompts_list = RasaConverter._get_text_prompts_list(action_templates)
                self._write_function_body_prompt(prompts_list, f)
                self._write_responder_lines(f)
            else:
                if action[0:6] == "action":
                    f.write("    # This is a custom action from rasa\n")
                    f.write("    pass\n")
                else:
                    # If no templates, write a blank function
                    f.write("    # No templates were provided for action\n")
                    f.write("    pass\n")
            if action != actions[-1]:
                f.write("\n")
                f.write("\n")

    @staticmethod
    def _attach_handle_to_function(handle, action, file_lines):
        for i, line in enumerate(file_lines):
            if len(re.findall("def {}".format(action), line)) > 0:
                insert_line = i
                while file_lines[i - 1].strip() != "":
                    if file_lines[i - 1] == handle:
                        return
                    i = i - 1
                file_lines.insert(insert_line, handle)

    @staticmethod
    def _attach_actions_to_function(actions, file_lines):
        current_line = 0
        for i, line in enumerate(file_lines):
            if len(re.findall("def {action}", line) > 0):
                current_line = i
                break
        while file_lines[current_line] != "":
            current_line += 1
        assert file_lines[current_line] == ""
        file_lines[current_line:current_line] = actions

    def create_mindmeld_init(self):
        f = self._write_init_header()
        actions = self._read_actions()
        templates = self._read_templates()
        # Write all functions for each action
        self._write_functions(actions, templates, f)
        f.close()
        # Read contents of current file
        file_lines = self._read_file_lines()
        stories_dictionary = self._get_stories()
        # Loop through all stories and create intent-action relationship
        for item in stories_dictionary.items():
            # Loop through steps for each story
            for step in item[1]:
                # Get intent, any entities, and actions
                intent = step["intent"]
                entities = step["entities"]
                actions = step["actions"]
                # attach handle to correct function
                app_handle_string = RasaConverter._get_app_handle(intent, entities)
                self._attach_handle_to_function(
                    app_handle_string, actions[0], file_lines
                )
                # check if more than 1 action per intent
                if len(actions) > 1:
                    self._attach_actions_to_function(actions[1:], file_lines)
        # write all lines back to file
        with open(self.mindmeld_project_directory + "/__init__.py", "w") as f:
            f.writelines(file_lines)

    @staticmethod
    def create_custom_features(mindmeld_project_directory, main_file_loc):
        with open(main_file_loc + "/rasa_custom_features.txt", "r") as f:
            string = f.read()
        with open(mindmeld_project_directory + "/custom_features.py", "w") as f:
            f.write(string)

    def convert_project(self):
        """Main function that will convert a Rasa project into a Mindmeld project.

        The Rasa project consists of three major files that contain much of data
            that is converted into the Mindmeld project:
        /domain.yml - Contains all of the intents, entities, actions, and templates
            used in the rasa project
        /data/stories.md - Contains the stories which are used to match intents and
            actions together
        /data/nlu_data.md - Contains the training data for each intent. Some of the
            training data may contain entities

        limitations:
        - Rasa has the ability to have custom actions, which is not supported by
        the converter.
        - Rasa has the ability to handle multiple intents per query, while Mindmeld
        does not.
        - Rasa training data may be json format, which is not currently supported.
        - Rasa has a feature called Rasa forms which is not currently supported.
        - Rasa's configuration files are not transfered, instead generic Mindmeld
        configuration files are copied over.
        """
        # Create project directory with sub folders
        self.create_mindmeld_directory(self.mindmeld_project_directory)
        # Transfer over test data from Rasa project and reformat to Mindmeld project
        self.create_mindmeld_training_data()
        file_loc = os.path.dirname(os.path.realpath(__file__))
        self.create_main(self.mindmeld_project_directory, file_loc)
        self.create_mindmeld_init()
        self.create_custom_features(self.mindmeld_project_directory, file_loc)
