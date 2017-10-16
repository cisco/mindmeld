# -*- coding: utf-8 -*-

"""
This module is responsible for locating various Workbench app files.
"""
from __future__ import absolute_import, unicode_literals

import glob
import os
import logging

WORKBENCH_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.join(WORKBENCH_ROOT, 'mmworkbench')

APP_PATH = '{app_path}'

# Generated folder structure for models
GEN_FOLDER = os.path.join(APP_PATH, '.generated')
DOMAIN_MODEL_PATH = os.path.join(GEN_FOLDER, 'domain.pkl')
GEN_DOMAINS_FOLDER = os.path.join(GEN_FOLDER, 'domains')
GEN_DOMAIN_FOLDER = os.path.join(GEN_DOMAINS_FOLDER, '{domain}')
INTENT_MODEL_PATH = os.path.join(GEN_DOMAIN_FOLDER, 'intent.pkl')
GEN_INTENT_FOLDER = os.path.join(GEN_DOMAIN_FOLDER, '{intent}')
ENTITY_MODEL_PATH = os.path.join(GEN_INTENT_FOLDER, 'entity.pkl')
ROLE_MODEL_PATH = os.path.join(GEN_INTENT_FOLDER, '{entity}-role.pkl')
GAZETTEER_PATH = os.path.join(GEN_FOLDER, 'gaz-{entity}.pkl')
GEN_INDEXES_FOLDER = os.path.join(GEN_FOLDER, 'indexes')
GEN_INDEX_FOLDER = os.path.join(GEN_INDEXES_FOLDER, '{index}')
RANKING_MODEL_PATH = os.path.join(GEN_INDEX_FOLDER, 'ranking.pkl')

# Domains sub tree for labeled queries
DOMAINS_FOLDER = os.path.join(APP_PATH, 'domains')
DOMAIN_FOLDER = os.path.join(DOMAINS_FOLDER, '{domain}')
INTENT_FOLDER = os.path.join(DOMAIN_FOLDER, '{intent}')
LABELED_QUERY_FILE = os.path.join(INTENT_FOLDER, '{filename}')

# Entities sub tree
ENTITIES_FOLDER = os.path.join(APP_PATH, 'entities')
ENTITY_FOLDER = os.path.join(ENTITIES_FOLDER, '{entity}')
GAZETTEER_TXT_PATH = os.path.join(ENTITY_FOLDER, 'gazetteer.txt')
ENTITY_MAP_PATH = os.path.join(ENTITY_FOLDER, 'mapping.json')

# Indexes sub tree
INDEXES_FOLDER = os.path.join(APP_PATH, 'indexes')
INDEX_FOLDER = os.path.join(INDEXES_FOLDER, '{index}')
RANKING_FILE_PATH = os.path.join(INDEX_FOLDER, 'ranking.json')

# App level files
APP_MODULE_PATH = os.path.join(APP_PATH, 'app.py')
CONFIG_MODULE_PATH = os.path.join(APP_PATH, 'config.py')

# Default config files
RESOURCES_FOLDER = os.path.join(PACKAGE_ROOT, 'resources')
DEFAULT_PROCESSOR_CONFIG_PATH = os.path.join(RESOURCES_FOLDER, 'default_processor_config.json')
DEFAULT_TOKENIZER_CONFIG_PATH = os.path.join(RESOURCES_FOLDER, 'default_tokenizer_config.json')
ASCII_FOLDING_DICT_PATH = os.path.join(RESOURCES_FOLDER, 'ascii_folding_dict.txt')
MALLARD_JAR_PATH = os.path.join(RESOURCES_FOLDER, 'mindmeld-mallard.jar')
EMBEDDINGS_FOLDER_PATH = os.path.join(WORKBENCH_ROOT, 'data')
EMBEDDINGS_FILE_PATH = os.path.join(EMBEDDINGS_FOLDER_PATH, 'glove.6B.zip')
PREVIOUSLY_USED_CHAR_EMBEDDINGS_FILE_PATH = \
    os.path.join(EMBEDDINGS_FOLDER_PATH, 'previously_used_char_embeddings.pkl')
PREVIOUSLY_USED_WORD_EMBEDDINGS_FILE_PATH = \
    os.path.join(EMBEDDINGS_FOLDER_PATH, 'previously_used_word_embeddings.pkl')

# User specific directories
USER_CONFIG_DIR = os.path.join(os.path.expanduser('~'), '.mmworkbench')
USER_CONFIG_PATH = os.path.join(USER_CONFIG_DIR, 'config')
BLUEPRINTS_PATH = os.path.join(USER_CONFIG_DIR, 'blueprints')
BLUEPRINT_PATH = os.path.join(BLUEPRINTS_PATH, '{name}')

logger = logging.getLogger(__name__)


# Helpers
def safe_path(func):
    """A decorator to make the path safe by replacing unsafe characters"""
    def _wrapper(*args, **kwargs):
        return func(*args, **kwargs).replace(':', '_')

    return _wrapper


def _resolve_model_name(path, model_name=None):
    if model_name:
        path, ext = os.path.splitext(path)
        path = "{}_{}{}".format(path, model_name, ext)
    return path


# Collections

def get_domains(app_path):
    """Gets all domains for a given application.

    Args:
        app_path (str): The path to the app data.

    Returns:
        (set of str) A list of domain names.
    """
    if not os.path.exists(app_path):
        raise OSError('No app found at "{}"'.format(app_path))
    domains_dir = DOMAINS_FOLDER.format(app_path=app_path)
    return set(next(os.walk(domains_dir))[1])


def get_intents(app_path, domain):
    """Gets all intents for a given domain and application.

    Args:
        app_path (str): The path to the app data.
        domain (str): A domain under the application.

    Returns:
        (set of str) A list of intent names.
    """
    if not os.path.exists(app_path):
        raise OSError('Domain "{}" not found in app at "{}"'.format(domain, app_path))
    domain_dir = DOMAIN_FOLDER.format(app_path=app_path, domain=domain)
    return set(next(os.walk(domain_dir))[1])


def get_labeled_query_tree(app_path, patterns=None):
    """Gets labeled query files for a given domain and application.

    Args:
        app_path (str): The path to the app data.
        domain (str): A domain under the application.
        intents (list(str)): A list of intents to be considered. If none is passed, all intents will
            be used.
        patterns (list(str)): A list of file patterns to match

    Returns:
        (dict) A set of labeled query files.
    """
    domains_dir = DOMAINS_FOLDER.format(app_path=app_path)
    walker = os.walk(domains_dir)
    tree = {}
    found_pattern = False

    for parent, _, files in walker:
        components = []
        while parent != domains_dir:
            parent, component = os.path.split(parent)
            components.append(component)
        if len(components) == 1:
            domain = components[0]
            tree[domain] = {}
        if len(components) == 2:
            domain = components[1]
            intent = components[0]
            tree[domain][intent] = {}
            if patterns:
                for pattern in patterns:
                    for file_path in glob.iglob(os.path.join(parent, domain, intent, pattern)):
                        _, filename = os.path.split(file_path)
                        mod_time = os.path.getmtime(file_path)
                        tree[domain][intent][filename] = mod_time
                        found_pattern = True
            else:
                for filename in files:
                    mod_time = os.path.getmtime(os.path.join(parent, domain, intent, filename))
                    tree[domain][intent][filename] = mod_time

    if patterns and not found_pattern:
        logger.error("Couldn't find {} pattern files in {} directory".format(patterns, domains_dir))

    return tree


def get_entity_types(app_path):
    """Gets all entities for an application.

    Args:
        app_path (str): The path to the app data.

    Returns:
        (str) The path for this app's domain classifier model.

    """
    if not os.path.exists(app_path):
        raise OSError('No app found at "{}"'.format(app_path))
    entities_dir = ENTITIES_FOLDER.format(app_path=app_path)

    return next(os.walk(entities_dir))[1]


def get_indexes(app_path):
    """Gets all indexes for an application.

    Args:
        app_path (str): The path to the app data.

    Returns:
        (str) The path for this app's domain classifier model.

    """
    if not os.path.exists(app_path):
        raise OSError('No app found at "{}"'.format(app_path))
    indexes_dir = INDEXES_FOLDER.format(app_path=app_path)
    return next(os.walk(indexes_dir))[1]


def load_app_package(app_path, package_name=None):
    """
    Args:
        path (str): The path to the app data (for example: /user/joe/home_assistant).
        package_name (str): The name of the imported package. If none, use folder name.
    """
    import imp
    if not package_name:
        package_name = os.path.basename(app_path)
    package_path = os.path.dirname(app_path)

    fp, pathname, description = imp.find_module(package_name, [package_path])
    imp.load_module(package_name, fp, pathname, description)


# Files and folders

@safe_path
def get_current_app_path(path):
    """
    Args:
        path (str): The directory path inside a particular application

    Returns:
        str: The path for the root directory of this application
    """
    app_path = path
    while 'app.py' not in os.listdir(app_path):
        app_path = os.path.dirname(app_path)
    return app_path


@safe_path
def get_generated_data_folder(app_path):
    """
    Args:
        app_path (str): The path to the app data.

    Returns:
        str: The path for this app's generated files

    """
    path = GEN_FOLDER.format(app_path=app_path)
    return path


@safe_path
def get_domain_model_path(app_path, model_name=None):
    """
    Args:
        app_path (str): The path to the app data.
        model_name (str): The name of the model. Allows multiple models to be stored.

    Returns:
        (str) The path for this app's domain classifier model.

    """
    path = DOMAIN_MODEL_PATH.format(app_path=app_path)
    return _resolve_model_name(path, model_name)


@safe_path
def get_intent_model_path(app_path, domain, model_name=None):
    """
    Args:
        app_path (str): The path to the app data.
        domain (str): A domain under the application.
        model_name (str): The name of the model. Allows multiple models to be stored.

    Returns:
        (str) The path for this app's domain classifier model.

    """
    path = INTENT_MODEL_PATH.format(app_path=app_path, domain=domain)
    return _resolve_model_name(path, model_name)


@safe_path
def get_entity_model_path(app_path, domain, intent, model_name=None):
    """
    Args:
        app_path (str): The path to the app data.
        domain (str): A domain under the application.
        intent (str): A intent under the domain.
        model_name (str): The name of the model. Allows multiple models to be stored.

    Returns:
        (str) The path for this intent's named entity recognizer.

    """
    path = ENTITY_MODEL_PATH.format(app_path=app_path, domain=domain, intent=intent)
    return _resolve_model_name(path, model_name)


@safe_path
def get_role_model_path(app_path, domain, intent, entity, model_name=None):
    """
    Args:
        app_path (str): The path to the app data.
        domain (str): A domain under the application.
        intent (str): A intent under the domain.
        model_name (str): The name of the model. Allows multiple models to be stored.

    Returns:
        (str) The path for the intent's role classifier pickle.

    """
    path = ROLE_MODEL_PATH.format(app_path=app_path, domain=domain, intent=intent, entity=entity)
    return _resolve_model_name(path, model_name)


@safe_path
def get_gazetteer_data_path(app_path, gaz_name, model_name=None):
    """
    Args:
        app_path (str): The path to the app data.
        gaz_name (str): The name of the gazetteer

    Returns:
        (str) The path for the gazetteer pickle.
    """
    path = GAZETTEER_PATH.format(app_path=app_path, entity=gaz_name)
    return _resolve_model_name(path, model_name)


@safe_path
def get_labeled_query_file_path(app_path, domain, intent, filename):
    """
    Args:
        app_path (str): The path to the app data.
        domain (str): A domain under the application.
        intent (str): A intent under the domain.
        filename (str): The name of the queries file.

    Returns:
        (str) The full path of the specified file.
    """
    return LABELED_QUERY_FILE.format(app_path=app_path, domain=domain, intent=intent,
                                     filename=filename)


@safe_path
def get_entity_gaz_path(app_path, entity):
    """
    Args:
        app_path (str): The path to the app data.
        entity (str): An entity under the application.

    Returns:
        (str) The path for an mapping of the entity

    """
    return GAZETTEER_TXT_PATH.format(app_path=app_path, entity=entity)


@safe_path
def get_entity_folder(app_path, entity):
    """
    Args:
        app_path (str): The path to the app data.
        entity (str): An entity under the application.

    Returns:
        (str) The path for an entity folder

    """
    return ENTITY_FOLDER.format(app_path=app_path, entity=entity)


@safe_path
def get_entity_map_path(app_path, entity):
    """
    Args:
        app_path (str): The path to the app data.
        entity (str): An entity under the application.

    Returns:
        (str) The path for an mapping of the entity

    """
    return ENTITY_MAP_PATH.format(app_path=app_path, entity=entity)


@safe_path
def get_ranking_file_path(app_path, index):
    """
    Args:
        app_path (str): The path to the app data.
        index (str): A knowledge base index under the application.

    Returns:
        (str) The path for an mapping of the entity

    """
    return RANKING_FILE_PATH.format(app_path=app_path, index=index)


@safe_path
def get_app_module_path(app_path):
    """
    Args:
        app_path (str): The path to the app data.

    Returns:
        str: The path of the app module file.
    """
    return APP_MODULE_PATH.format(app_path=app_path)


@safe_path
def get_config_module_path(app_path):
    """
    Args:
        app_path (str): The path to the app data.

    Returns:
        str: The path of the config module file.
    """
    return CONFIG_MODULE_PATH.format(app_path=app_path)


def get_cached_blueprint_path(name):
    """
    Args:
        name (str): The name of the blueprint

    Returns:
        str: The path to the blueprint archives.
    """
    return BLUEPRINT_PATH.format(name=name)


def get_user_config_path():
    """

    Returns:
        str: The path to the current user's workbench configuration file.
    """
    return USER_CONFIG_PATH
