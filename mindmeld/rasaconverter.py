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

from keyword import iskeyword
import re, os, yaml, copy
from converter import Converter


class Rasaconverter(Converter):

    def __init__(self, rasa_project_directory, mindmeld_project_directory):
        self.rasa_project_directory = rasa_project_directory
        self.mindmeld_project_directory = mindmeld_project_directory

    @staticmethod
    def __create_intents_directories(mindmeld_project_directory, intents):
        for intent in intents:
            Converter.create_directory(mindmeld_project_directory + "/domains/default/" + intent) 

    @staticmethod
    def __create_entities_directories(mindmeld_project_directory, entities):
        for entity in entities:
            Converter.create_directory(mindmeld_project_directory + "/entities/" + entity)
            print(mindmeld_project_directory + "/entities/" + entity)
            with open(mindmeld_project_directory + "/entities/" + entity + "/gazetteer.txt", "w") as f:
                f.close()
            with open(mindmeld_project_directory + "/entities/" + entity + "/mapping.json", "w") as f:
                #HAVE TO DOCUMENT WHAT USER HAS TO DO
                f.write("{\n  \"entities\":[]\n}")
                f.close()

    @staticmethod
    def __is_line_intent_definiton(line):
        return (line[0:10] == "## intent:")

    @staticmethod
    def __get_intent_from_line(line):
            return line.split(' ')[1].split(':')[1].rstrip()

    @staticmethod
    def __create_intent_training_file(intent_directory):
        Converter.create_directory(intent_directory)
        with open(intent_directory + "/train.txt", "w") as f:
                f.close()

    @staticmethod
    def __does_intent_ex_contain_entity(intent_example):
            return len(re.findall("\[.*\]\(.*\)", intent_example)) > 0

    @staticmethod
    def __write_intent_with_extinty(intent_f, intent_example):
        mindmend_intent_example = intent_example
        for match in re.findall("\[.*\]\(.*\)", intent_example):
            mindmeld_entity = match.replace("[","{").replace("]","|").replace("(","").replace(")","}")
            mindmend_intent_example = mindmend_intent_example.replace(match, mindmeld_entity)
            intent_f.write(mindmend_intent_example)
            # add this to the respective entity gazetteer file as well
            Rasaconverter.create_entity_files(mindmeld_entity)

    @staticmethod
    def __remove_comments_from_line(line):
        start_of_comment = line.find("<!---")
        end_of_comment = line.find("-->")
        line_without_comment = line.replace(line[start_of_comment:end_of_comment+3], '')
        line_without_comment = line_without_comment.rstrip()
        return line_without_comment

    @staticmethod
    def __add_example_to_training_file(current_intent_path, line):
        with open(current_intent_path + "/train.txt", "a") as intent_f:
            intent_example = line[2:]
            intent_example = Rasaconverter.__remove_comments_from_line(intent_example) + "\n"
            if Rasaconverter.__does_intent_ex_contain_entity(intent_example):
                Rasaconverter.__write_intent_with_extinty(intent_f, intent_example)
            else:
                intent_f.write(intent_example)

    def __read_domain_file(self):
        if os.path.isfile(self.rasa_project_directory + "/domain.yml"):
            try:
                with open(self.rasa_project_directory + "/domain.yml", "r") as stream:
                    domain_data_loaded = yaml.safe_load(stream)
                    return domain_data_loaded
            except IOError as e:
                print("Can not open domain.yml file at %s" % self.rasa_project_directory + "/domain.yml")
                print(e)
        elif os.path.isfile(self.rasa_project_directory + "/domain.yaml"):
            try:
                with open(self.rasa_project_directory + "/domain.yml", "r") as stream:
                    domain_data_loaded = yaml.safe_load(stream)
                    return domain_data_loaded
            except IOError as e:
                print("Can not open domain.yml file at %s" % self.rasa_project_directory + "/domain.yaml")
                print(e)
        else:
            print("Could not find domain.yml file in project directory")
            raise FileNotFoundError

    def __read_entities(self):
        domain_file = self.__read_domain_file()
        return domain_file['entities']

    def __read_slots(self):
        domain_file = self.__read_domain_file()
        return domain_file['slots']

    def __read_intents(self):
        domain_file = self.__read_domain_file()
        return domain_file['intents']

    def __read_actions(self):
        domain_file = self.__read_domain_file()
        return domain_file['actions']

    def __read_templates(self):
        domain_file = self.__read_domain_file()
        return domain_file['templates']
    
    @staticmethod
    def create_entity_files(mm_entry):
        entity = mm_entry.strip('{}').split("|")
        with open("mindmeld_project/entities/" + entity[1] + "/gazetteer.txt", "a") as f:
            f.write(entity[0] + "\n")
            f.close()

    @staticmethod
    def __is_valid_function_name(name):
        return name.isidentifier() and not iskeyword(name)

    @staticmethod
    def __is_story_name(stories_line):
        return stories_line[0:3] == '## '

    @staticmethod
    def __get_story_name(stories_line):
        if "<!--" in stories_line:
            return Rasaconverter.__remove_comments_from_line(stories_line.replace("## ", "")).rstrip()
        else:
            return stories_line.replace("## ", "").rstrip()

    @staticmethod
    def __is_intent(stories_line):
        return stories_line[0:2] == '* '

    @staticmethod
    def __is_action(stories_line):
        return stories_line[0:3] == ' - '

    @staticmethod
    def __does_intent_have_entity(stories_line):
        return len(re.findall(r"\{.*\}", stories_line)) > 0

    @staticmethod
    def __clean_up_entities_list(entities_with_values):
        # trim off { }
        entities_with_values = entities_with_values[1:-1]
        # split data entities if there are multiples and clean white space
        entities_list = entities_with_values.split(",")
        for i, entity in enumerate(entities_list):
            entities_list[i] = entity.replace("\"",'')
            entities_list[i] = entities_list[i].lstrip()
        return entities_list

    @staticmethod
    def __get_intent_with_entity(stories_line):
        if Rasaconverter.__does_intent_have_entity(stories_line):
            entities_with_values = re.search(r"\{.*\}", stories_line)
            entities_with_values = entities_with_values.group(0)
            entities_list = Rasaconverter.__clean_up_entities_list(entities_with_values)
            start_of_entity = stories_line.find(entities_with_values)
            intent = Rasaconverter.__remove_comments_from_line(stories_line[2:start_of_entity]).rstrip()
            return intent, entities_list
        else:
            intent = Rasaconverter.__remove_comments_from_line(stories_line[2:]).rstrip()
            entities_list = []
            return intent, entities_list

    def __get_stories(self):
        if os.path.isfile(self.rasa_project_directory + "/data/stories.md"):
            try:
                with open(self.rasa_project_directory + "/data/stories.md", "r") as f:
                    stories_dictionary = {}
                    current_story_name = ''
                    steps = []
                    current_step = {}
                    current_intent = ''
                    current_actions = []
                    stories_lines = f.readlines()
                    max_lines = len(stories_lines)
                    for line_num, line in enumerate(stories_lines):
                        if Rasaconverter.__is_story_name(line):
                            current_story_name = Rasaconverter.__get_story_name(line)
                            continue 
                        elif Rasaconverter.__is_intent(line):
                            current_intent, current_entities = Rasaconverter.__get_intent_with_entity(line)
                            current_step["intent"] = copy.deepcopy(current_intent)
                            current_step["entities"] = copy.deepcopy(current_entities)
                            continue
                        elif Rasaconverter.__is_action(line):
                            current_actions.append(Rasaconverter.__remove_comments_from_line(line[3:]).rstrip())                
                            if ((line_num + 1) < max_lines) and Rasaconverter.__is_action(stories_lines[line_num + 1]):
                                continue
                            else:
                                current_step["actions"] = copy.deepcopy(current_actions)
                                current_actions.clear()
                                steps.append(copy.deepcopy(current_step))
                                current_step.clear()
                        elif len(line.strip()) == 0:
                            if current_story_name != '':
                                stories_dictionary[current_story_name] = copy.deepcopy(steps)
                                steps.clear()
                                current_story_name = ''
                        if line_num == (max_lines - 1):
                            stories_dictionary[current_story_name] = copy.deepcopy(steps)
                            steps.clear()
                            current_story_name = ''
                    f.close()
                    return stories_dictionary
            except Exception as e: 
                print("Can not open stories.md file at %s" % self.rasa_project_directory + "/data/stories.md")
                print(e)
        else:
            print("Could not find stories.md file in %s" % self.rasa_project_directory + "/data/stories.md")
            raise FileNotFoundError

    @staticmethod
    def create_mindmeld_directory(mindmeld_project_path):
        Converter.create_directory(mindmeld_project_path)
        Converter.create_directory(mindmeld_project_path + "/data")
        Converter.create_directory(mindmeld_project_path + "/domains")
        Converter.create_directory(mindmeld_project_path + "/domains/default")
        Converter.create_directory(mindmeld_project_path + "/entities")

    def create_training_data(self, rasa_project_directory, mindmeld_project_directory):
        """Method to transfer and reformat the training data in a Rasa Project
        """
        # read intents listed in domain.yml
        intents = self.__read_intents()

        # create intents subdirectories
        Rasaconverter.__create_intents_directories(mindmeld_project_directory, intents)

         # read entities in domain.yml
        entities = self.__read_entities()

        # create entities subdirectories
        Rasaconverter.__create_entities_directories(mindmeld_project_directory, entities)
        
        # try and open data files from rasa project
        nlu_data_loc = rasa_project_directory + "/data/nlu_data.md"
        try:
            nlu_data_md_file = open(nlu_data_loc, "r")
        except:
            print("Can not open nlu_data.md file at %s" % nlu_data_loc)
            raise FileNotFoundError
        nlu_data_lines = nlu_data_md_file.readlines()
        # iterate through each line
        current_intent = ''
        current_intent_path = ''
        for line in nlu_data_lines:
            if (Rasaconverter.__is_line_intent_definiton(line)):
                current_intent = Rasaconverter.__get_intent_from_line(line)
                current_intent_path = mindmeld_project_directory + "/domains/default/" + current_intent
                # create data text file for intent examples`
                Rasaconverter.__create_intent_training_file(current_intent_path)
            else:
                if (line[0] == '-'):
                    Rasaconverter.__add_example_to_training_file(current_intent_path, line)
    
    def __write_init_header(self):
        string = '''from mindmeld import Application
from . import custom_features  # noqa: F401

app = Application(__name__)

__all__ = ['app']



'''
        f = open(self.mindmeld_project_directory + "/__init__.py", "w+")
        f.write(string)
        return f
    
    @staticmethod
    def __get_app_handle(intent, entities):
        has_entity_string = ', has_entity='
        entities_string = ""
        if len(entities) > 0:
            for entity_value in entities:
                entity_string = entity_value.split(":")[0]     
                entities_string += has_entity_string + "'" + entity_string + "'"
        handle_string = "@app.handle(intent='" + intent + "'" + entities_string + ")\n"
        return handle_string    
    
    @staticmethod
    def __write_function_declaration(action, f):
        if Rasaconverter.__is_valid_function_name(action):
            function_declartion_string = "def {}(request, responder):\n".format(action)
            f.write(function_declartion_string)
        else:
            print("Action {action} is not a valid name for a python function")
            raise SyntaxError

    @staticmethod
    def __write_function_body_prompt(prompts, f):
        entities_list = []
        # check if prompts contain any entities
        for prompt in prompts:
            entities = re.findall(r"\{.*\}", prompt)
            entities_list += entities
        for entity in entities_list:
            newentity = entity.replace("{","").replace("}","")  
            entity_string = "    {} = request.context['{}']\n".format(newentity, newentity)
            f.write(entity_string)
        prompts_string = "    prompts = {}\n".format(prompts)
        f.write(prompts_string)

    @staticmethod
    def __get_text_prompts_list(action_templates):
        prompts = []
        for template in action_templates:
            if 'text' in template:
                prompts.append(template['text'])
        return prompts    

    @staticmethod
    def __write_responder_lines(f):
        responder_string = "    responder.reply(prompts)\n    responder.listen()\n"
        f.write(responder_string)

    def __read_file_lines(self):
        with open(self.mindmeld_project_directory + "/__init__.py", "r+") as f:
            return f.readlines()

    @staticmethod
    def __write_functions(actions, templates, f):
        for action in actions:
            Rasaconverter.__write_function_declaration(action, f)
            if action in templates:
                # Get list of templates per action
                action_templates = templates[action]
                prompts_list = Rasaconverter.__get_text_prompts_list(action_templates)
                Rasaconverter.__write_function_body_prompt(prompts_list, f)
                Rasaconverter.__write_responder_lines(f)
            f.write('\n')

    @staticmethod
    def __attach_handle_to_function(handle, action, file_lines):
        for i, line in enumerate(file_lines):
            if (len(re.findall("def {}".format(action),line)) > 0):
                insert_line = i
                while (file_lines[i - 1].strip() != ''): 
                    if (file_lines[i - 1] == handle):
                        return
                    i = i - 1
                file_lines.insert(insert_line, handle)
    
    @staticmethod
    def __attach_actions_to_function(function_name, actions, file_lines):
        current_line = 0
        for i, line in enumerate(file_lines):
            if (len(re.findall("def {action}",line) > 0)):
                current_line = i
                break
        while (file_lines[current_line] != ""):
            current_line += 1
    
        assert file_lines[current_line] == ""
        file_lines[current_line:current_line] = actions


    def create_init(self, mindmeld_project_directory):
        f = self.__write_init_header()
        intents = self.__read_intents()
        actions = self.__read_actions()
        templates = self.__read_templates()
        # Write all functions for each action
        Rasaconverter.__write_functions(actions, templates, f)
        f.close()
        # Read contents of current file
        file_lines = self.__read_file_lines()

        stories_dictionary = self.__get_stories()
        # Loop through all stories and create intent-action relationship
        for story_name, steps in stories_dictionary.items():
            # Loop through steps for each story
            for step in steps:
                # Get intent, any entities, and actions
                intent = step['intent']
                entities = step['entities']
                actions = step['actions']
                # attach handle to correct function
                app_handle_string = Rasaconverter.__get_app_handle(intent, entities)
                Rasaconverter.__attach_handle_to_function(app_handle_string, actions[0], file_lines)
                # check if more than 1 action per intent
                if len(actions) > 1:
                    Rasaconverter.__attach_actions_to_function(actions[0], actions[1:], file_lines)
        # write all lines back to file
        with open(mindmeld_project_directory + "/__init__.py", "w") as f:
            f.writelines(file_lines)
    
    @staticmethod
    def create_main(mindmeld_project_directory):
        string = '''# -*- coding: utf-8 -*-
"""This module defines the command line interface for this app. Use
`python -m <app_name>` to see available commands.
"""

if __name__ == '__main__':
    from . import app
    app.cli()'''
        with open(mindmeld_project_directory + "/__main__.py", "w") as f:
            f.write(string)
            f.close()
    
    @staticmethod
    def create_config(mindmeld_project_directory):
        string = '''# -*- coding: utf-8 -*-
"""This module contains a template MindMeld app configuration"""

# The namespace of the application. Used to prevent collisions in supporting services across
# applications. If not set here, the app's enclosing directory name is used.
# APP_NAMESPACE = 'app-name'

# Dictionaries for the various NLP classifier configurations

# An example decision tree model for intent classification
INTENT_CLASSIFIER_CONFIG = {
    'model_type': 'text',
    'model_settings': {
        'classifier_type': 'dtree'
    },
    'param_selection': {
        'type': 'k-fold',
        'k': 10,
        'grid': {
            'max_features': ['log2', 'sqrt', 0.01, 0.1]
        },
    },
    "features": {
        "exact": {},
    }
}

"""
# Fill in the other model configurations if necessary
# DOMAIN_CLASSIFIER_CONFIG = {}
# ENTITY_RECOGNIZER_CONFIG = {}
# ROLE_CLASSIFIER_CONFIG = {}
"""

# A example configuration for the parser
"""
# *** Note: these are place holder entity types ***
PARSER_CONFIG = {
    'grandparent': {
        'parent': {},
        'child': {'max_instances': 1}
    },
    'parent': {
        'child': {'max_instances': 1}
    }
}
"""'''
        with open(mindmeld_project_directory + "/config.py", "w") as f:
            f.write(string)

    @staticmethod
    def create_custom_features(mindmeld_project_directory):
        string = '''from mindmeld.models.helpers import register_query_feature, register_entity_feature


@register_query_feature(feature_name='average-token-length')
def extract_average_token_length(**args):
    """
    Example query feature that gets the average length of normalized tokens in the query

    Returns:
        (function) A feature extraction function that takes a query and
            returns the average normalized token length
    """
    def _extractor(query, resources):
        tokens = query.normalized_tokens
        average_token_length = sum([len(t) for t in tokens]) / len(tokens)
        return {'average_token_length': average_token_length}

    return _extractor


@register_entity_feature(feature_name='entity-span-start')
def extract_entity_span_start(**args):
    """
    Example entity feature that gets the start span for each entity

    Returns:
        (function) A feature extraction function that returns the start span of the entity
    """
    def _extractor(example, resources):
        query, entities, entity_index = example
        features = {}

        current_entity = entities[entity_index]
        current_entity_token_start = current_entity.token_span.start

        features['entity_span_start'] = current_entity_token_start
        return features

    return _extractor'''
        with open(mindmeld_project_directory + "/custom_features.py", "w") as f:
            f.write(string)

    def convert_project(self):
        # Create project directory with sub folders
        Rasaconverter.create_mindmeld_directory(self.mindmeld_project_directory)
        # Transfer over test data from Rasa project and reformat to Mindmeld project
        self.create_training_data(self.rasa_project_directory, self.mindmeld_project_directory)
        Rasaconverter.create_main(self.mindmeld_project_directory)
        self.create_init(self.mindmeld_project_directory)
        Rasaconverter.create_config(self.mindmeld_project_directory)
        Rasaconverter.create_custom_features(self.mindmeld_project_directory)
        