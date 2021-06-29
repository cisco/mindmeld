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


"""
This module is responsible for locating various MindMeld app files.
"""
import logging
import os
import re
import sys
from functools import wraps
from importlib.machinery import SourceFileLoader

from .exceptions import MindMeldImportError

MINDMELD_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PACKAGE_ROOT = os.path.join(MINDMELD_ROOT, "mindmeld")

APP_PATH = "{app_path}"

# Generated folder structure for models
GEN_FOLDER = os.path.join(APP_PATH, ".generated")
MODEL_CACHE_PATH = os.path.join(GEN_FOLDER, "cached_models")
QUERY_CACHE_DB_PATH = os.path.join(GEN_FOLDER, "query_cache.db")
DOMAIN_MODEL_PATH = os.path.join(GEN_FOLDER, "domain.pkl")
GEN_DOMAINS_FOLDER = os.path.join(GEN_FOLDER, "domains")
GEN_TIMESTAMP_FOLDER = os.path.join(MODEL_CACHE_PATH, "{timestamp}")
DOMAIN_MODEL_TIMESTAMP_PATH = os.path.join(GEN_TIMESTAMP_FOLDER, "domain.pkl")
GEN_DOMAINS_CHECKPOINT_FOLDER = os.path.join(GEN_TIMESTAMP_FOLDER, "domains")
GEN_DOMAIN_FOLDER = os.path.join(GEN_DOMAINS_FOLDER, "{domain}")
GEN_DOMAIN_CHECKPOINT_FOLDER = os.path.join(GEN_DOMAINS_CHECKPOINT_FOLDER, "{domain}")
INTENT_MODEL_PATH = os.path.join(GEN_DOMAIN_FOLDER, "intent.pkl")
INTENT_MODEL_CHECKPOINT_PATH = os.path.join(GEN_DOMAIN_CHECKPOINT_FOLDER, "intent.pkl")
GEN_INTENT_FOLDER = os.path.join(GEN_DOMAIN_FOLDER, "{intent}")
GEN_INTENT_CHECKPOINT_FOLDER = os.path.join(GEN_DOMAIN_CHECKPOINT_FOLDER, "{intent}")
ENTITY_MODEL_PATH = os.path.join(GEN_INTENT_FOLDER, "entity.pkl")
ENTITY_MODEL_CHECKPOINT_PATH = os.path.join(GEN_INTENT_CHECKPOINT_FOLDER, "entity.pkl")
ROLE_MODEL_PATH = os.path.join(GEN_INTENT_FOLDER, "{entity}-role.pkl")
ROLE_MODEL_CHECKPOINT_PATH = os.path.join(
    GEN_INTENT_CHECKPOINT_FOLDER, "{entity}-role.pkl"
)
GAZETTEER_PATH = os.path.join(GEN_FOLDER, "gaz-{entity}.pkl")
GEN_INDEXES_FOLDER = os.path.join(GEN_FOLDER, "indexes")
GEN_INDEX_FOLDER = os.path.join(GEN_INDEXES_FOLDER, "{index}")
RANKING_MODEL_PATH = os.path.join(GEN_INDEX_FOLDER, "ranking.pkl")
DEPRECATED_GEN_EMBEDDER_MODEL_PATH = os.path.join(
    GEN_INDEXES_FOLDER, "{embedder_type}_{model_name}_cache.pkl"
)
GEN_EMBEDDER_MODEL_PATH = os.path.join(
    GEN_INDEXES_FOLDER, "{model_id}_cache.pkl"
)
GEN_ENTITY_RESOLVERS_FOLDER = os.path.join(GEN_FOLDER, "entity_resolvers")
GEN_ENTITY_RESOLVER_CACHE = os.path.join(GEN_ENTITY_RESOLVERS_FOLDER, "{uid}.pkl")
GEN_QUESTION_ANSWERERS_FOLDER = os.path.join(GEN_FOLDER, "question_answerers")
GEN_QUESTION_ANSWERER_INDICES_CACHE = os.path.join(GEN_QUESTION_ANSWERERS_FOLDER, "{uid}.pkl")
NATIVE_QUESTION_ANSWERER_INDICES_CACHE_PATH = os.path.join(
    os.path.expanduser("~"), ".cache/mindmeld"
)

# Domains sub tree for labeled queries
DOMAINS_FOLDER = os.path.join(APP_PATH, "domains")
DOMAIN_FOLDER = os.path.join(DOMAINS_FOLDER, "{domain}")
INTENT_FOLDER = os.path.join(DOMAIN_FOLDER, "{intent}")
LABELED_QUERY_FILE = os.path.join(INTENT_FOLDER, "{filename}")

# Entities sub tree
ENTITIES_FOLDER = os.path.join(APP_PATH, "entities")
ENTITY_FOLDER = os.path.join(ENTITIES_FOLDER, "{entity}")
GAZETTEER_TXT_PATH = os.path.join(ENTITY_FOLDER, "gazetteer.txt")
ENTITY_MAP_PATH = os.path.join(ENTITY_FOLDER, "mapping.json")

# Indexes sub tree
INDEXES_FOLDER = os.path.join(APP_PATH, "indexes")
INDEX_FOLDER = os.path.join(INDEXES_FOLDER, "{index}")
RANKING_FILE_PATH = os.path.join(INDEX_FOLDER, "ranking.json")

# App level files
APP_MODULE_PATH = os.path.join(APP_PATH, "app.py")
CONFIG_MODULE_PATH = os.path.join(APP_PATH, "config.py")

# DVC local remote folder
DVC_LOCAL_REMOTE_PATH = os.path.join(APP_PATH, "dvc_local_remote")

# Default config files
RESOURCES_FOLDER = os.path.join(PACKAGE_ROOT, "resources")
DEFAULT_PROCESSOR_CONFIG_PATH = os.path.join(
    RESOURCES_FOLDER, "default_processor_config.json"
)
DEFAULT_TOKENIZER_CONFIG_PATH = os.path.join(
    RESOURCES_FOLDER, "default_tokenizer_config.json"
)
ASCII_FOLDING_DICT_PATH = os.path.join(RESOURCES_FOLDER, "ascii_folding_dict.txt")

DUCKLING_UBUNTU16_PATH = os.path.join(
    RESOURCES_FOLDER, "duckling-x86_64-linux-ubuntu-16"
)
DUCKLING_UBUNTU18_PATH = os.path.join(
    RESOURCES_FOLDER, "duckling-x86_64-linux-ubuntu-18"
)
DUCKLING_CENTOS8_PATH = os.path.join(RESOURCES_FOLDER, "duckling-x86_64-centos-8-core")
DUCKLING_OSX_PATH = os.path.join(RESOURCES_FOLDER, "duckling-x86_64-osx")
DUCKLING_UBUNTU16_MD5 = "1e58b4e91d580d98c8ac4f5a69b0ebd1"
DUCKLING_UBUNTU18_MD5 = "b9c4891b27731df97d7f01c33441a91c"
DUCKLING_OSX_MD5 = "d01753261e6e7940b533f09dc17d0c19"
DUCKLING_CENTOS8_MD5 = "f840176c2b96c0037edd9524faac5e93"
DUCKLING_OS_MAPPINGS = {
    "ubuntu-16.04": DUCKLING_UBUNTU16_PATH,
    "ubuntu-18.04": DUCKLING_UBUNTU18_PATH,
    "darwin": DUCKLING_OSX_PATH,
    "centos-8-core": DUCKLING_CENTOS8_PATH,
}
DUCKLING_PATH_TO_MD5_MAPPINGS = {
    DUCKLING_UBUNTU16_PATH: DUCKLING_UBUNTU16_MD5,
    DUCKLING_UBUNTU18_PATH: DUCKLING_UBUNTU18_MD5,
    DUCKLING_OSX_PATH: DUCKLING_OSX_MD5,
    DUCKLING_CENTOS8_PATH: DUCKLING_CENTOS8_MD5,
}

EMBEDDINGS_FOLDER_PATH = os.path.join(MINDMELD_ROOT, "data")
EMBEDDINGS_FILE_PATH = os.path.join(EMBEDDINGS_FOLDER_PATH, "glove.6B.zip")
PREVIOUSLY_USED_CHAR_EMBEDDINGS_FILE_PATH = os.path.join(
    EMBEDDINGS_FOLDER_PATH, "previously_used_char_embeddings.pkl"
)
PREVIOUSLY_USED_WORD_EMBEDDINGS_FILE_PATH = os.path.join(
    EMBEDDINGS_FOLDER_PATH, "previously_used_word_embeddings.pkl"
)

# User specific directories
USER_CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".mindmeld")
USER_CONFIG_PATH = os.path.join(USER_CONFIG_DIR, "config")
BLUEPRINTS_PATH = os.path.join(USER_CONFIG_DIR, "blueprints")
BLUEPRINT_PATH = os.path.join(BLUEPRINTS_PATH, "{name}")

# Active Learning (AL)
AL_EXPERIMENT_FOLDER = "{experiment_folder}"
AL_PARAMS_PATH = os.path.join(AL_EXPERIMENT_FOLDER, "params.json")
AL_RESULTS_FOLDER = os.path.join(AL_EXPERIMENT_FOLDER, "results")
AL_ACCURACIES_PATH = os.path.join(AL_RESULTS_FOLDER, "accuracies.json")
AL_SELECTED_QUERIES_PATH = os.path.join(AL_RESULTS_FOLDER, "selected_queries.json")
AL_PLOTS_FOLDER = os.path.join(AL_EXPERIMENT_FOLDER, "plots")

logger = logging.getLogger(__name__)


# Helpers
def safe_path(func):
    """A decorator to make the path safe by replacing unsafe characters"""

    @wraps(func)
    def _wrapper(*args, **kwargs):
        res = func(*args, **kwargs)
        if isinstance(res, tuple):
            return tuple(map(lambda x: x.replace(":", "_") if x else x, res))
        elif isinstance(res, str):
            return res.replace(":", "_")
        else:
            return res

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


def _search_pattern(patterns, parent, domain, intent, tree, found_pattern):
    domain_intent_dir = os.path.join(parent, domain, intent)
    for app_file in os.listdir(domain_intent_dir):
        for pattern in patterns:
            if re.match(pattern, app_file):
                abs_filepath = os.path.join(domain_intent_dir, app_file)
                mod_time = os.path.getmtime(abs_filepath)
                tree[domain][intent][abs_filepath] = mod_time
                found_pattern[pattern] = True


def get_labeled_query_tree(app_path, patterns=None):
    """Gets labeled query files for a given domain and application.

    Args:
        app_path (str): The path to the app data.
        patterns (list(str)): A list of file patterns to match

    Returns:
        (dict) A set of labeled query files.
    """
    domains_dir = DOMAINS_FOLDER.format(app_path=app_path)
    walker = os.walk(domains_dir)
    tree = {}
    found_pattern = {pattern: False for pattern in patterns} if patterns else {}

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
                _search_pattern(patterns, parent, domain, intent, tree, found_pattern)
            else:
                for filename in files:
                    abs_filepath = os.path.join(parent, domain, intent, filename)
                    mod_time = os.path.getmtime(abs_filepath)
                    tree[domain][intent][abs_filepath] = mod_time

    for pattern in found_pattern:
        if not found_pattern[pattern]:
            logger.error(
                "Couldn't find %s pattern files in %s directory", patterns, domains_dir
            )

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


# Files and folders


@safe_path
def get_generated_data_folder(app_path):
    """Gets the path to the folder containing files the app generates.

    Args:
        app_path (str): The path to the app data.

    Returns:
        str: The path for this app's generated files

    """
    path = GEN_FOLDER.format(app_path=app_path)
    return path


@safe_path
def get_domain_model_paths(app_path, model_name=None, timestamp=None):
    """Gets the path to the domain classifier model as well as the path to a
    timestamp-cached domain classifier model.

    Args:
        app_path (str): The path to the app data.
        model_name (str): The name of the model. Allows multiple models to be stored.
        timestamp (str): The timestamp string to store cached models in

    Returns:
        (str) A tuple with the main model path and the cached model path

    """
    main_path = DOMAIN_MODEL_PATH.format(app_path=app_path)
    main_path = _resolve_model_name(main_path)

    ts_path = None
    if timestamp:
        ts_path = DOMAIN_MODEL_TIMESTAMP_PATH.format(
            app_path=app_path, timestamp=timestamp
        )
        ts_path = _resolve_model_name(ts_path, model_name)

    return main_path, ts_path


@safe_path
def get_intent_model_paths(app_path, domain, model_name=None, timestamp=None):
    """Gets the path to the intent classifier model as well as the path to a
    timestamp-cached intent classifier model.

    Args:
        app_path (str): The path to the app data.
        domain (str): A domain under the application.
        model_name (str): The name of the model. Allows multiple models to be stored.
        timestamp (str): The timestamp string to store cached models in

    Returns:
        (tuple) A tuple with the main model path and the cached model path

    """
    main_path = INTENT_MODEL_PATH.format(app_path=app_path, domain=domain)
    main_path = _resolve_model_name(main_path)

    ts_path = None
    if timestamp:
        ts_path = INTENT_MODEL_CHECKPOINT_PATH.format(
            app_path=app_path, domain=domain, timestamp=timestamp
        )
        ts_path = _resolve_model_name(ts_path, model_name)

    return main_path, ts_path


@safe_path
def get_entity_model_paths(app_path, domain, intent, model_name=None, timestamp=None):
    """Gets the path to the entity recognizer model as well as the path to a
    timestamp-cached entity recognizer model.

    Args:
        app_path (str): The path to the app data.
        domain (str): A domain under the application.
        intent (str): A intent under the domain.
        model_name (str): The name of the model. Allows multiple models to be stored.
        timestamp (str): The timestamp string to store cached models in

    Returns:
        (tuple) A tuple with the main model path and the cached model path

    """
    main_path = ENTITY_MODEL_PATH.format(
        app_path=app_path, domain=domain, intent=intent
    )
    main_path = _resolve_model_name(main_path)

    ts_path = None
    if timestamp:
        ts_path = ENTITY_MODEL_CHECKPOINT_PATH.format(
            app_path=app_path, domain=domain, intent=intent, timestamp=timestamp
        )
        ts_path = _resolve_model_name(ts_path, model_name)

    return main_path, ts_path


@safe_path
def get_role_model_paths(
    app_path, domain, intent, entity, model_name=None, timestamp=None
):
    """Gets the path to the role classifier model as well as the path to a
    timestamp-cached role classifier model.

    Args:
        app_path (str): The path to the app data.
        domain (str): A domain under the application.
        intent (str): A intent under the domain.
        entity (str): An entity under the intent
        model_name (str): The name of the model. Allows multiple models to be stored.
        timestamp (str): The timestamp string to store cached models in

    Returns:
        (tuple) A tuple with the main model path and the cached model path

    """
    main_path = ROLE_MODEL_PATH.format(
        app_path=app_path, domain=domain, intent=intent, entity=entity
    )
    main_path = _resolve_model_name(main_path)

    ts_path = None
    if timestamp:
        ts_path = ROLE_MODEL_CHECKPOINT_PATH.format(
            app_path=app_path,
            domain=domain,
            intent=intent,
            entity=entity,
            timestamp=timestamp,
        )
        ts_path = _resolve_model_name(ts_path, model_name)

    return main_path, ts_path


@safe_path
def get_gazetteer_data_path(app_path, gaz_name, model_name=None):
    """Gets path to the saved gazetteer pickle.

    Args:
        app_path (str): The path to the app data.
        gaz_name (str): The name of the gazetteer.
        model_name (str): The name of the model.

    Returns:
        (str) The path for the gazetteer pickle.
    """
    path = GAZETTEER_PATH.format(app_path=app_path, entity=gaz_name)
    return _resolve_model_name(path, model_name)


@safe_path
def get_labeled_query_file_path(app_path, domain, intent, filename):
    """Gets path to a labeled query file corresponding to a specific domain and intent.

    Args:
        app_path (str): The path to the app data.
        domain (str): A domain under the application.
        intent (str): A intent under the domain.
        filename (str): The name of the queries file.

    Returns:
        (str) The full path of the specified file.
    """
    return LABELED_QUERY_FILE.format(
        app_path=app_path, domain=domain, intent=intent, filename=filename
    )


@safe_path
def get_entity_gaz_path(app_path, entity):
    """Gets the path to the gazetteer text file for a given entity.

    Args:
        app_path (str): The path to the app data.
        entity (str): An entity under the application.

    Returns:
        (str) The path for a mapping of the entity

    """
    return GAZETTEER_TXT_PATH.format(app_path=app_path, entity=entity)


@safe_path
def get_entity_folder(app_path, entity):
    """Gets the path to the folder for a given entity.

    Args:
        app_path (str): The path to the app data.
        entity (str): An entity under the application.

    Returns:
        (str) The path for an entity folder

    """
    return ENTITY_FOLDER.format(app_path=app_path, entity=entity)


@safe_path
def get_entity_map_path(app_path, entity):
    """Gets the path to the entity mapping file (mapping.json) for a given entity.

    Args:
        app_path (str): The path to the app data.
        entity (str): An entity under the application.

    Returns:
        (str) The path for a mapping of the entity

    """
    return ENTITY_MAP_PATH.format(app_path=app_path, entity=entity)


@safe_path
def get_ranking_file_path(app_path, index):
    """Gets the path to the ranking.json file for a given entity.

    Args:
        app_path (str): The path to the app data.
        index (str): A knowledge base index under the application.

    Returns:
        (str) The path for a mapping of the entity

    """
    return RANKING_FILE_PATH.format(app_path=app_path, index=index)


@safe_path
def get_embedder_cache_file_path(app_path, embedder_type, model_name=None):
    """Gets the path to the model_cache.json file for a given embedder model.

    Args:
        app_path (str): The path to the app data.
        embedder_type (str): The name of the embedder type.
        model_name (str, optional): The name of the specific trained model.

    Returns:
        (str) The path for the json cached of the embedded values.
    """
    if model_name:
        return DEPRECATED_GEN_EMBEDDER_MODEL_PATH.format(
            app_path=app_path,
            embedder_type=embedder_type,
            model_name=model_name,
        )
    else:
        return GEN_EMBEDDER_MODEL_PATH.format(
            app_path=app_path,
            model_id=embedder_type
        )


@safe_path
def get_question_answerer_index_cache_file_path(app_path, uid):
    """Gets the path to the question answerer index cache file.

    Args:
        app_path (str): The path to the app data.
        uid (str): A unique filename for the .pkl file

    Returns:
        (str) The path for the .pkl cache.
    """
    return GEN_QUESTION_ANSWERER_INDICES_CACHE.format(app_path=app_path, uid=uid)


@safe_path
def get_entity_resolver_cache_file_path(app_path, uid):
    """Gets the path to the entity resolver cache file for a given entity type and a chosen model type.

    Args:
        app_path (str): The path to the app data.
        uid (str): A unique filename for the .pkl file

    Returns:
        (str) The path for the .pkl cache.
    """
    return GEN_ENTITY_RESOLVER_CACHE.format(app_path=app_path, uid=uid)


@safe_path
def get_app_module_path(app_path):
    """Gets the path to the application file (app.py) for a given application if it exists.

    Args:
        app_path (str): The path to the app data.

    Returns:
        str: The path of the app module file.
    """
    return APP_MODULE_PATH.format(app_path=app_path)


@safe_path
def get_config_module_path(app_path):
    """Gets the path to the configuration file (config.py) for a given application.

    Args:
        app_path (str): The path to the app data.

    Returns:
        str: The path of the config module file.
    """
    return CONFIG_MODULE_PATH.format(app_path=app_path)


@safe_path
def get_dvc_local_remote_path(app_path):
    return DVC_LOCAL_REMOTE_PATH.format(app_path=app_path)


def get_cached_blueprint_path(name):
    """Gets the path to a cached version of the given blueprint.

    Args:
        name (str): The name of the blueprint

    Returns:
        str: The path to the blueprint archives.
    """
    return BLUEPRINT_PATH.format(name=name)


def get_user_config_path():
    """Gets the path to the current configuration file used by MindMeld.

    Returns:
        str: The path to the current user's MindMeld configuration file.
    """
    return USER_CONFIG_PATH


def get_app(app_path):
    """Get the Application instance for given application path.

    Args:
        app_path (str): The path to an application on disk

    Returns:
        mindmeld.app.Application: the MindMeld application

    Raises:
        MindMeldImportError: when the application can not be found
    """
    app_path = os.path.abspath(app_path)
    package_name = os.path.basename(app_path)

    try:
        # check if package is already imported
        if package_name in sys.modules:
            logger.warning(
                "The application package %s is already imported.", package_name
            )
            mod = __import__(package_name)
            return mod.app
        # try to load as package first
        loader = SourceFileLoader(package_name, os.path.join(app_path, "__init__.py"))
        return loader.load_module(package_name).app  # pylint: disable=deprecated-method
    except AttributeError:
        # __init__.py exists but has no app attribute
        # fallback to app.py, but emit warning
        logger.warning(
            "The application package at %r includes no %r attribute. Falling back to %r.",
            app_path,
            "app",
            "app.py",
        )
    except FileNotFoundError:
        # fallback to app.py
        pass

    try:
        # try to load 'app.py'
        loader = SourceFileLoader(package_name, get_app_module_path(app_path))
        return loader.load_module(package_name).app  # pylint: disable=deprecated-method
    except (FileNotFoundError, AttributeError) as e:
        msg = (
            "Could not import application at {!r}. Create a __init__.py or app.py file"
            " containing the application.".format(app_path)
        )
        raise MindMeldImportError(msg) from e
