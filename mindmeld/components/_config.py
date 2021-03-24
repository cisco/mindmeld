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
This module contains the Config class.
"""
import copy
import imp
import logging
import os
import warnings

from .. import path
from .schemas import validate_language_code, validate_locale_code
from ..constants import CURRENCY_SYMBOLS

logger = logging.getLogger(__name__)

DUCKLING_SERVICE_NAME = "duckling"
DEFAULT_DUCKLING_URL = "http://localhost:7151/parse"

CONFIG_DEPRECATION_MAPPING = {
    "DOMAIN_CLASSIFIER_CONFIG": "DOMAIN_MODEL_CONFIG",
    "INTENT_CLASSIFIER_CONFIG": "INTENT_MODEL_CONFIG",
    "ENTITY_RECOGNIZER_CONFIG": "ENTITY_MODEL_CONFIG",
    "ROLE_CLASSIFIER_CONFIG": "ROLE_MODEL_CONFIG",
    "ENTITY_RESOLVER_CONFIG": "ENTITY_RESOLUTION_CONFIG",
    "QUESTION_ANSWERER_CONFIG": "QUESTION_ANSWERING_CONFIG",
    "get_entity_recognizer_config": "get_entity_model_config",
    "get_intent_classifier_config": "get_intent_model_config",
    "get_entity_resolver_config": "get_entity_resolution_model_config",
    "get_role_classifier_config": "get_role_model_config",
}

DEFAULT_DOMAIN_CLASSIFIER_CONFIG = {
    "model_type": "text",
    "model_settings": {
        "classifier_type": "logreg",
    },
    "param_selection": {
        "type": "k-fold",
        "k": 10,
        "grid": {"fit_intercept": [True, False], "C": [10, 100, 1000, 10000, 100000]},
    },
    "features": {"bag-of-words": {"lengths": [1]}, "freq": {"bins": 5}, "in-gaz": {}},
}

DEFAULT_INTENT_CLASSIFIER_CONFIG = {
    "model_type": "text",
    "model_settings": {"classifier_type": "logreg"},
    "param_selection": {
        "type": "k-fold",
        "k": 10,
        "grid": {
            "fit_intercept": [True, False],
            "C": [0.01, 1, 100, 10000, 1000000],
            "class_bias": [1, 0.7, 0.3, 0],
        },
    },
    "features": {
        "bag-of-words": {"lengths": [1]},
        "in-gaz": {},
        "freq": {"bins": 5},
        "length": {},
    },
}

DEFAULT_ENTITY_RECOGNIZER_CONFIG = {
    "model_type": "tagger",
    "label_type": "entities",
    "model_settings": {
        "classifier_type": "memm",
        "tag_scheme": "IOB",
        "feature_scaler": "max-abs",
    },
    "param_selection": {
        "type": "k-fold",
        "k": 5,
        "scoring": "accuracy",
        "grid": {
            "penalty": ["l1", "l2"],
            "C": [0.01, 1, 100, 10000, 1000000, 100000000],
        },
    },
    "features": {
        "bag-of-words-seq": {
            "ngram_lengths_to_start_positions": {
                1: [-2, -1, 0, 1, 2],
                2: [-2, -1, 0, 1],
            }
        },
        "in-gaz-span-seq": {},
        "sys-candidates-seq": {"start_positions": [-1, 0, 1]},
    },
}

DEFAULT_ENTITY_RESOLVER_CONFIG = {"model_type": "text_relevance"}

DEFAULT_QUESTION_ANSWERER_CONFIG = {"model_type": "keyword"}

ENGLISH_LANGUAGE_CODE = "en"
ENGLISH_US_LOCALE = "en_US"
DEFAULT_LANGUAGE_CONFIG = {
    "language": ENGLISH_LANGUAGE_CODE,
    "locale": ENGLISH_US_LOCALE,
}


# ElasticSearch mapping to define text analysis settings for text fields.
# It defines specific index configuration for synonym indices. The common index configuration
# is in default index template.
DEFAULT_ES_SYNONYM_MAPPING = {
    "mappings": {
        "properties": {
            "sort_factor": {"type": "double"},
            "whitelist": {
                "type": "nested",
                "properties": {
                    "name": {
                        "type": "text",
                        "fields": {
                            "raw": {"type": "keyword", "ignore_above": 256},
                            "normalized_keyword": {
                                "type": "text",
                                "analyzer": "keyword_match_analyzer",
                            },
                            "char_ngram": {
                                "type": "text",
                                "analyzer": "char_ngram_analyzer",
                            },
                        },
                        "analyzer": "default_analyzer",
                    }
                },
            },
        }
    }
}

PHONETIC_ES_SYNONYM_MAPPING = {
    "mappings": {
        "properties": {
            "sort_factor": {"type": "double"},
            "whitelist": {
                "type": "nested",
                "properties": {
                    "name": {
                        "type": "text",
                        "fields": {
                            "raw": {"type": "keyword", "ignore_above": 256},
                            "normalized_keyword": {
                                "type": "text",
                                "analyzer": "keyword_match_analyzer",
                            },
                            "char_ngram": {
                                "type": "text",
                                "analyzer": "char_ngram_analyzer",
                            },
                            "double_metaphone": {
                                "type": "text",
                                "analyzer": "phonetic_analyzer",
                            },
                        },
                        "analyzer": "default_analyzer",
                    }
                },
            },
            "cname": {
                "type": "text",
                "analyzer": "default_analyzer",
                "fields": {
                    "raw": {"type": "keyword", "ignore_above": 256},
                    "normalized_keyword": {
                        "type": "text",
                        "analyzer": "keyword_match_analyzer",
                    },
                    "char_ngram": {
                        "type": "text",
                        "analyzer": "char_ngram_analyzer",
                    },
                    "double_metaphone": {
                        "type": "text",
                        "analyzer": "phonetic_analyzer",
                    },
                },
            },
        }
    },
    "settings": {
        "analysis": {
            "filter": {
                "phonetic_filter": {
                    "type": "phonetic",
                    "encoder": "doublemetaphone",
                    "replace": True,
                    "max_code_len": 7,
                }
            },
            "analyzer": {
                "phonetic_analyzer": {
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "token_shingle",
                        "phonetic_filter",
                    ],
                    "char_filter": [
                        "remove_comma",
                        "remove_tm_and_r",
                        "remove_loose_apostrophes",
                        "space_possessive_apostrophes",
                        "remove_special_beginning",
                        "remove_special_end",
                        "remove_special1",
                        "remove_special2",
                        "remove_special3",
                        "remove_dot",
                    ],
                    "type": "custom",
                    "tokenizer": "whitespace",
                }
            },
        }
    },
}

DEFAULT_ROLE_CLASSIFIER_CONFIG = {
    "model_type": "text",
    "model_settings": {"classifier_type": "logreg"},
    "params": {"C": 100, "penalty": "l1"},
    "features": {
        "bag-of-words-before": {
            "ngram_lengths_to_start_positions": {1: [-2, -1], 2: [-2, -1]}
        },
        "bag-of-words-after": {
            "ngram_lengths_to_start_positions": {1: [0, 1], 2: [0, 1]}
        },
        "other-entities": {},
    },
}

DEFAULT_ES_INDEX_TEMPLATE_NAME = "mindmeld_default"

# Default ES index template that contains the base index configuration shared across different
# types of indices. Currently all ES indices will be created using this template.
# - custom text analysis settings such as custom analyzers, token filters and character filters.
# - dynamic field mapping template for text fields
# - common fields, e.g. id.
DEFAULT_ES_INDEX_TEMPLATE = {
    "template": "*",
    "mappings": {
        "dynamic_templates": [
            {
                "default_text": {
                    "match": "*",
                    "match_mapping_type": "string",
                    "mapping": {
                        "type": "text",
                        "analyzer": "default_analyzer",
                        "fields": {
                            "raw": {"type": "keyword", "ignore_above": 256},
                            "normalized_keyword": {
                                "type": "text",
                                "analyzer": "keyword_match_analyzer",
                            },
                            "processed_text": {
                                "type": "text",
                                "analyzer": "english",
                            },
                            "char_ngram": {
                                "type": "text",
                                "analyzer": "char_ngram_analyzer",
                            },
                        },
                    },
                }
            }
        ],
        "properties": {"id": {"type": "keyword"}},
    },
    "settings": {
        "analysis": {
            "char_filter": {
                "remove_loose_apostrophes": {
                    "pattern": " '|' ",
                    "type": "pattern_replace",
                    "replacement": "",
                },
                "space_possessive_apostrophes": {
                    "pattern": "([^\\p{N}\\s]+)'s ",
                    "type": "pattern_replace",
                    "replacement": "$1 's ",
                },
                "remove_special_beginning": {
                    "pattern": "^[^\\p{L}\\p{N}\\p{Sc}&']+",
                    "type": "pattern_replace",
                    "replacement": "",
                },
                "remove_special_end": {
                    "pattern": "[^\\p{L}\\p{N}&']+$",
                    "type": "pattern_replace",
                    "replacement": "",
                },
                "remove_special1": {
                    "pattern": "([\\p{L}]+)[^\\p{L}\\p{N}&']+(?=[\\p{N}\\s]+)",
                    "type": "pattern_replace",
                    "replacement": "$1 ",
                },
                "remove_special2": {
                    "pattern": "([\\p{N}]+)[^\\p{L}\\p{N}&']+(?=[\\p{L}\\s]+)",
                    "type": "pattern_replace",
                    "replacement": "$1 ",
                },
                "remove_special3": {
                    "pattern": "([\\p{L}]+)[^\\p{L}\\p{N}&']+(?=[\\p{L}]+)",
                    "type": "pattern_replace",
                    "replacement": "$1 ",
                },
                "remove_comma": {
                    "pattern": ",",
                    "type": "pattern_replace",
                    "replacement": "",
                },
                "remove_tm_and_r": {
                    "pattern": "™|®",
                    "type": "pattern_replace",
                    "replacement": "",
                },
                "remove_dot": {
                    "pattern": "([\\p{L}]+)[.]+(?=[\\p{L}\\s]+)",
                    "type": "pattern_replace",
                    "replacement": "$1",
                },
            },
            "filter": {
                "token_shingle": {
                    "max_shingle_size": "4",
                    "min_shingle_size": "2",
                    "output_unigrams": "true",
                    "type": "shingle",
                },
                "ngram_filter": {"type": "ngram", "min_gram": "3", "max_gram": "3"},
            },
            "analyzer": {
                "default_analyzer": {
                    "filter": ["lowercase", "asciifolding", "token_shingle"],
                    "char_filter": [
                        "remove_comma",
                        "remove_tm_and_r",
                        "remove_loose_apostrophes",
                        "space_possessive_apostrophes",
                        "remove_special_beginning",
                        "remove_special_end",
                        "remove_special1",
                        "remove_special2",
                        "remove_special3",
                    ],
                    "type": "custom",
                    "tokenizer": "whitespace",
                },
                "keyword_match_analyzer": {
                    "filter": ["lowercase", "asciifolding"],
                    "char_filter": [
                        "remove_comma",
                        "remove_tm_and_r",
                        "remove_loose_apostrophes",
                        "space_possessive_apostrophes",
                        "remove_special_beginning",
                        "remove_special_end",
                        "remove_special1",
                        "remove_special2",
                        "remove_special3",
                    ],
                    "type": "custom",
                    "tokenizer": "keyword",
                },
                "char_ngram_analyzer": {
                    "filter": ["lowercase", "asciifolding", "ngram_filter"],
                    "char_filter": [
                        "remove_comma",
                        "remove_tm_and_r",
                        "remove_loose_apostrophes",
                        "space_possessive_apostrophes",
                        "remove_special_beginning",
                        "remove_special_end",
                        "remove_special1",
                        "remove_special2",
                        "remove_special3",
                    ],
                    "type": "custom",
                    "tokenizer": "whitespace",
                },
            },
        }
    },
}


# Elasticsearch mapping to define knowledge base index specific configuration:
# - dynamic field mapping to index all synonym whitelist in fields with "$whitelist" suffix.
# - location field
#
# The common configuration is defined in default index template
DEFAULT_ES_QA_MAPPING = {
    "mappings": {
        "dynamic_templates": [
            {
                "synonym_whitelist_text": {
                    "match": "*$whitelist",
                    "match_mapping_type": "object",
                    "mapping": {
                        "type": "nested",
                        "properties": {
                            "name": {
                                "type": "text",
                                "fields": {
                                    "raw": {"type": "keyword", "ignore_above": 256},
                                    "normalized_keyword": {
                                        "type": "text",
                                        "analyzer": "keyword_match_analyzer",
                                    },
                                    "char_ngram": {
                                        "type": "text",
                                        "analyzer": "char_ngram_analyzer",
                                    },
                                },
                                "analyzer": "default_analyzer",
                            }
                        },
                    },
                }
            }
        ],
        "properties": {"location": {"type": "geo_point"}},
    }
}

DEFAULT_PARSER_DEPENDENT_CONFIG = {
    "left": True,
    "right": True,
    "min_instances": 0,
    "max_instances": None,
    "precedence": "left",
    "linking_words": frozenset(),
}

DEFAULT_RANKING_CONFIG = {"query_clauses_operator": "or"}

DEFAULT_NLP_CONFIG = {
    "resolve_entities_using_nbest_transcripts": [],
    "system_entity_recognizer": {
        "type": DUCKLING_SERVICE_NAME,
        "url": DEFAULT_DUCKLING_URL,
    },
}

DEFAULT_AUGMENTATION_CONFIG = {
    "augmentor_class": "EnglishParaphraser",
    "batch_size": 8,
    "paths": [
        {
            "domains": ".*",
            "intents": ".*",
            "files": ".*",
        }
    ],
    "path_suffix": "-augment.txt"
}

DEFAULT_AUTO_ANNOTATOR_CONFIG = {
    "annotator_class": "MultiLingualAnnotator",
    "overwrite": False,
    "annotation_rules": [
        {
            "domains": ".*",
            "intents": ".*",
            "files": ".*",
            "entities": ".*",
        }
    ],
    "unannotate_supported_entities_only": True,
    "unannotation_rules": None,
    "translator": "NoOpTranslator",
}

DEFAULT_TOKENIZER_CONFIG = {
    # populated in the `get_tokenizer_config` func
    "allowed_patterns": [],
}


class NlpConfigError(Exception):
    pass


def get_custom_action_config(app_path):
    if not app_path:
        return None
    try:
        custom_action_config = getattr(
            _get_config_module(app_path), "CUSTOM_ACTION_CONFIG", None
        )
        return custom_action_config
    except (OSError, IOError):
        logger.info("No app configuration file found.")
        return None


def get_max_history_len(app_path):
    if not app_path:
        return None
    try:
        custom_action_config = getattr(
            _get_config_module(app_path), "MAX_HISTORY_LEN", None
        )
        return custom_action_config
    except (OSError, IOError):
        logger.info("No app configuration file found.")
        return None


def get_language_config(app_path):
    if not app_path:
        return ENGLISH_LANGUAGE_CODE, ENGLISH_US_LOCALE
    try:
        language_config = getattr(
            _get_config_module(app_path), "LANGUAGE_CONFIG", DEFAULT_LANGUAGE_CONFIG
        )
        locale = language_config.get("locale")
        language = language_config.get("language")
        resolved_language = resolve_language(language, locale)
        return resolved_language, locale
    except (OSError, IOError):
        logger.info(
            "No app configuration file found. Using default language and locale."
        )
        return ENGLISH_LANGUAGE_CODE, ENGLISH_US_LOCALE


def resolve_language(language=None, locale=None):
    """
    Resolves to a language given a locale.
    """
    locale = validate_locale_code(locale)
    language = validate_language_code(language)

    # Locale overrides language
    if locale:
        language = locale.split("_")[0]

    if not language:
        language = ENGLISH_LANGUAGE_CODE

    return language.lower()


def get_app_namespace(app_path):
    """Returns the namespace of the application at app_path"""
    try:
        _app_namespace = _get_config_module(app_path).APP_NAMESPACE
        if "JUPYTER_USER" in os.environ:
            _app_namespace = "{}_{}".format(os.environ["JUPYTER_USER"], _app_namespace)
        return _app_namespace
    except (OSError, IOError):
        logger.debug("No app configuration file found")
    except AttributeError:
        logger.debug("App namespace not set in app configuration")

    # If a relative path is passed in, we resolve to its abspath
    app_path = os.path.abspath(app_path) if not os.path.isabs(app_path) else app_path

    _app_namespace = os.path.split(app_path)[1]
    if "JUPYTER_USER" in os.environ:
        _app_namespace = "{jupyter_user}_{app_namespace}".format(
            jupyter_user=os.environ["JUPYTER_USER"], app_namespace=_app_namespace
        )
    return _app_namespace


def is_duckling_configured(app_path):
    """Returns True if the app config specifies that duckling should be run
    as a system entity recognizer

    Args:
        app_path (str): A application path

    Returns:
        (bool): True if the app config specifies that the numerical parsing
            should be run
    """
    if not app_path:
        raise NlpConfigError("Application path is not valid")

    config = get_nlp_config(app_path).get("system_entity_recognizer")

    if isinstance(config, dict):
        # We get into this conditional when the app has specified the system_entity_recognizer
        # nlp config
        return config.get("type") == DUCKLING_SERVICE_NAME
    else:
        # We get into this conditional when the app has not specified the system_entity_recognizer
        # nlp config, in which case, we default to the duckling API
        return True


def get_system_entity_url_config(app_path):
    """
    Get system entity url from the application's config. If the application does not define the url,
        return the default duckling url.
    """
    if not app_path:
        raise NlpConfigError("Application path is not valid")

    return (
        get_nlp_config(app_path)
        .get("system_entity_recognizer", {})
        .get("url", DEFAULT_DUCKLING_URL)
    )


def get_classifier_config(
    clf_type, app_path=None, domain=None, intent=None, entity=None
):
    """Returns the config for the specified classifier, with the
    following  order of precedence.

    If the application contains a config.py file:
    - Return the response from the get_*_model_config function in
      config.py for the specified classifier type. E.g.
      `get_intent_model_config`.
    - If the function does not exist, or raise an exception, return the
      config specified by *_MODEL_CONFIG in config.py, e.g.
      INTENT_MODEL_CONFIG.

    Otherwise, use the MindMeld default config for the classifier type


    Args:
        clf_type (str): The type of the classifier. One of 'domain',
            'intent', 'entity', 'entity_resolution', or 'role'.
        app_path (str, optional): The location of the app
        domain (str, optional): The domain of the classifier
        intent (str, optional): The intent of the classifier
        entity (str, optional): The entity type of the classifier

    Returns:
        dict: A classifier config
    """
    try:
        module_conf = _get_config_module(app_path)

    except (OSError, IOError):
        logger.info(
            "No app configuration file found. Using default %s model configuration",
            clf_type,
        )
        return _get_default_classifier_config(clf_type)

    func_name = {
        "intent": "get_intent_classifier_config",
        "entity": "get_entity_recognizer_config",
        "entity_resolution": "get_entity_resolver_config",
        "role": "get_role_classifier_config",
    }.get(clf_type)
    func_args = {
        "intent": ("domain",),
        "entity": ("domain", "intent"),
        "entity_resolution": ("domain", "intent", "entity"),
        "role": ("domain", "intent", "entity"),
    }.get(clf_type)

    if func_name:
        func = None
        try:
            func = getattr(module_conf, func_name)
        except AttributeError:
            try:
                func = getattr(module_conf, CONFIG_DEPRECATION_MAPPING[func_name])
                msg = (
                    "%s config key is deprecated. Please use the equivalent %s config "
                    "key" % (CONFIG_DEPRECATION_MAPPING[func_name], func_name)
                )
                warnings.warn(msg, DeprecationWarning)
            except AttributeError:
                pass
        if func:
            try:
                raw_args = {"domain": domain, "intent": intent, "entity": entity}
                args = {k: raw_args[k] for k in func_args}
                return copy.deepcopy(func(**args))
            except Exception as exc:  # pylint: disable=broad-except
                # Note: this is intentionally broad -- provider could raise any exception
                logger.warning(
                    "%r configuration provider raised exception: %s", clf_type, exc
                )

    attr_name = {
        "domain": "DOMAIN_CLASSIFIER_CONFIG",
        "intent": "INTENT_CLASSIFIER_CONFIG",
        "entity": "ENTITY_RECOGNIZER_CONFIG",
        "entity_resolution": "ENTITY_RESOLVER_CONFIG",
        "role": "ROLE_CLASSIFIER_CONFIG",
        "question_answering": "QUESTION_ANSWERER_CONFIG",
    }[clf_type]
    try:
        return copy.deepcopy(getattr(module_conf, attr_name))
    except AttributeError:
        try:
            result = copy.deepcopy(
                getattr(module_conf, CONFIG_DEPRECATION_MAPPING[attr_name])
            )
            msg = (
                "%s config is deprecated. Please use the equivalent %s config "
                "key" % (CONFIG_DEPRECATION_MAPPING[attr_name], attr_name)
            )
            warnings.warn(msg, DeprecationWarning)
            return result
        except AttributeError:
            logger.info("No %s model configuration set. Using default.", clf_type)

    return _get_default_classifier_config(clf_type)


def _get_default_classifier_config(clf_type):
    return copy.deepcopy(
        {
            "domain": DEFAULT_DOMAIN_CLASSIFIER_CONFIG,
            "intent": DEFAULT_INTENT_CLASSIFIER_CONFIG,
            "entity": DEFAULT_ENTITY_RECOGNIZER_CONFIG,
            "entity_resolution": DEFAULT_ENTITY_RESOLVER_CONFIG,
            "role": DEFAULT_ROLE_CLASSIFIER_CONFIG,
            "language_config": DEFAULT_LANGUAGE_CONFIG,
            "question_answering": DEFAULT_QUESTION_ANSWERER_CONFIG,
        }[clf_type]
    )


def get_parser_config(app_path=None, config=None, domain=None, intent=None):
    """Gets the fully specified parser configuration for the app at the
    given path.

    Args:
        app_path (str, optional): The location of the MindMeld app
        config (dict, optional): A config object to use. This will
            override the config specified by the app's config.py file.
            If necessary, this object will be expanded to a fully
            specified config object.
        domain (str, optional): The domain of the parser
        intent (str, optional): The intent of the parser

    Returns:
        dict: A fully parser configuration
    """
    if config:
        return _expand_parser_config(config)

    if not app_path:
        raise NlpConfigError("Application path is not valid")

    try:
        module_conf = _get_config_module(app_path)
    except (OSError, IOError):
        logger.info("No app configuration file found. Not configuring parser.")
        return _get_default_parser_config()

    # Try provider first
    config_provider = None
    try:
        config_provider = module_conf.get_parser_config
    except AttributeError:
        pass
    if config_provider:
        try:
            config = config or config_provider(domain, intent)
            return _expand_parser_config(config)
        except Exception as exc:  # pylint: disable=broad-except
            # Note: this is intentionally broad -- provider could raise any exception
            logger.warning("Parser configuration provider raised exception: %s", exc)

    # Try object second
    try:
        config = config or module_conf.PARSER_CONFIG
        return _expand_parser_config(config)
    except AttributeError:
        pass

    return _get_default_parser_config()


def _get_default_parser_config():
    return None


def _expand_parser_config(config):
    # Replace with -- since | has a special meaning for parser
    return {
        head.replace("|", "--"): _expand_group_config(group)
        for head, group in config.items()
    }


def _expand_group_config(group_config):
    """Expands a parser group configuration.

    A group config can either be a list of dependents or a dictionary with a
    field for each dependent.

    In the list a dependent can be a string containing the name of the
    entity-role type identifier or a dictionary with at least a type field.

    In the dictionary the dependent must be another dictionary.

    Some example parser configs follow below.

    A very simple configuration:

       {
           'head': ['dependent']
       }

    A more realistic simple config:

        {
            'product|beverage': ['size', 'quantity', 'option|beverage'],
            'product|baked-good': ['size', 'quantity', 'option|baked-good'],
            'store': ['location'],
            'option': ['size']
        }

    A fully specified config:

        {
            'product': {
                'quantity': {
                    'left': True,
                    'right': True,
                    'precedence': 'left',
                    'min_instances': 0,
                    'max_instances': 3
                },
                'size': {
                    'left': True,
                    'right': True,
                    'precedence': 'left',
                    'min_instances': 0,
                    'max_instances': 1
                },
                'option': {
                    'left': True,
                    'right': True,
                    'precedence': 'left',
                    'min_instances': 0,
                    'max_instances': 1
                }
            },
            'store': {
                'location': {
                    'left': True,
                    'right': True,
                    'precedence': 'left',
                    'min_instances': 0,
                    'max_instances': 1
                }
            },
            'option': {
                'size': {
                    'left': True,
                    'right': True,
                    'precedence': 'left',
                    'min_instances': 0,
                    'max_instances': 1
                }
            }
        }
    """
    group_config = copy.deepcopy(group_config)
    expanded = {}
    if isinstance(group_config, (tuple, list, set)):
        for dependent in group_config:
            config = copy.copy(DEFAULT_PARSER_DEPENDENT_CONFIG)
            try:
                dep_type = dependent.pop("type")
                config.update(dependent)
            except (AttributeError, ValueError):
                # simple style config -- dependent is a str
                dep_type = dependent
            # Replace with -- since | has a special meaning for parser
            expanded[dep_type.replace("|", "--")] = config
    else:
        for dep_type, dep_config in group_config.items():
            config = copy.copy(DEFAULT_PARSER_DEPENDENT_CONFIG)
            dep_config.pop("type", None)
            config.update(dep_config)
            # Replace with -- since | has a special meaning for parser
            expanded[dep_type.replace("|", "--")] = config
    return expanded


def _get_config_module(app_path):
    module_path = path.get_config_module_path(app_path)

    config_module = imp.load_source(
        "config_module_" + os.path.basename(app_path), module_path
    )
    return config_module


def _get_default_nlp_config():
    return copy.deepcopy(DEFAULT_NLP_CONFIG)


def get_nlp_config(app_path=None, config=None):
    """Gets the fully specified processor configuration for the app at the
    given path.

    Args:
        app_path (str, optional): The location of the MindMeld app
        config (dict, optional): A config object to use. This will
            override the config specified by the app's config.py file.
            If necessary, this object will be expanded to a fully
            specified config object.

    Returns:
        dict: The nbest inference configuration
    """
    if config:
        return config
    try:
        module_conf = _get_config_module(app_path)
    except (OSError, IOError):
        logger.info("No app configuration file found.")
        return _get_default_nlp_config()

    # Try provider first
    try:
        return copy.deepcopy(module_conf.get_nlp_config())
    except AttributeError:
        pass

    # Try object second
    try:
        config = config or module_conf.NLP_CONFIG
        return config
    except AttributeError:
        pass

    return _get_default_nlp_config()


def get_augmentation_config(app_path=None):
    """Gets the augmentation config for the app.

    Args:
        app_path (str, optional): The location of the MindMeld app.

    Returns:
        dict: The augmentation config.
    """
    try:
        augmentation_config = getattr(
            _get_config_module(app_path),
            "AUGMENTATION_CONFIG",
            DEFAULT_AUGMENTATION_CONFIG,
        )
        return augmentation_config
    except (OSError, IOError, AttributeError):
        logger.info(
            "No app configuration file found. Using the default augmentation config."
        )
        return DEFAULT_AUGMENTATION_CONFIG


def get_auto_annotator_config(app_path=None):
    """Gets the automatic annotator config for the app at the
    given path.

    Args:
        app_path (str, optional): The location of the MindMeld app

    Returns:
        dict: The automatic annotator config.
    """
    try:
        auto_annotator_config = getattr(
            _get_config_module(app_path),
            "AUTO_ANNOTATOR_CONFIG",
            DEFAULT_AUTO_ANNOTATOR_CONFIG,
        )
        return auto_annotator_config
    except (OSError, IOError):
        logger.info(
            "No app configuration file found. Using the default automatic annotator config."
        )
        return DEFAULT_AUTO_ANNOTATOR_CONFIG


def _get_default_regex(exclude_from_norm):
    """Gets the default special character regex for the Tokenizer config.

    Args:
        exclude_from_norm (optional) - list of chars to exclude from normalization

    Returns:
        list: default special character regex list
    """
    # List of regex's for matching and tokenizing when keep_special_chars=True
    keep_special_regex_list = []

    exception_chars = "\@\[\]\|\{\}'"  # noqa: W605

    to_exclude = CURRENCY_SYMBOLS + "".join(exclude_from_norm or [])

    letter_pattern_str = "[^\W\d_]+"  # noqa: W605

    # Make keep special regex list
    keep_special_regex_list.append(
        "?P<start>^[^\w\d&" + to_exclude + exception_chars + "]+"  # noqa: W605
    )
    keep_special_regex_list.append(
        "?P<end>[^\w\d&" + to_exclude + exception_chars + "]+$"  # noqa: W605
    )
    keep_special_regex_list.append(
        "?P<pattern1>(?P<pattern1_replace>"  # noqa: W605
        + letter_pattern_str
        + ")"
        + "[^\w\d\s&"  # noqa: W605
        + exception_chars
        + "]+(?=[\d]+)"  # noqa: W605
    )
    keep_special_regex_list.append(
        "?P<pattern2>(?P<pattern2_replace>[\d]+)[^\w\d\s&"  # noqa: W605
        + exception_chars
        + "]+"
        + "u(?="
        + letter_pattern_str
        + ")"
    )
    keep_special_regex_list.append(
        "?P<pattern3>(?P<pattern3_replace>"
        + letter_pattern_str
        + ")"  # noqa: W605
        + "[^\w\d\s&"  # noqa: W605
        + exception_chars
        + "]+"
        + "(?="  # noqa: W605
        + letter_pattern_str
        + ")"
    )
    keep_special_regex_list.append(
        "?P<escape1>(?P<escape1_replace>[\w\d]+)"  # noqa: W605
        + "[^\w\d\s"  # noqa: W605
        + exception_chars
        + "]+"
        + "(?=\|)"  # noqa: W605
    )
    keep_special_regex_list.append(
        "?P<escape2>(?P<escape2_replace>[\]\}]+)"  # noqa: W605
        + "[^\w\d\s"  # noqa: W605
        + exception_chars
        + "]+(?=s)"
    )

    keep_special_regex_list.append("?P<underscore>_")  # noqa: W605
    keep_special_regex_list.append("?P<begspace>^\s+")  # noqa: W605
    keep_special_regex_list.append("?P<trailspace>\s+$")  # noqa: W605
    keep_special_regex_list.append("?P<spaceplus>\s+")  # noqa: W605
    keep_special_regex_list.append("?P<apos_space> '|' ")  # noqa: W605
    keep_special_regex_list.append("?P<apos_s>(?<=[^\\s])'[sS]")  # noqa: W605
    # handle the apostrophes used at the end of a possessive form, e.g. dennis'
    keep_special_regex_list.append("?P<apos_poss>(^'(?=\S)|(?<=\S)'$)")  # noqa: W605

    return keep_special_regex_list


def get_tokenizer_config(app_path=None, exclude_from_norm=None):
    """Gets the tokenizer configuration for the app at the specified path.

    Args:
        app_path (str, optional): The location of the MindMeld app
        exclude_from_norm (list, optional): chars to exclude from normalization

    Returns:
        dict: The tokenizer configuration.
    """
    DEFAULT_TOKENIZER_CONFIG["default_allowed_patterns"] = _get_default_regex(
        exclude_from_norm
    )

    if not app_path:
        return DEFAULT_TOKENIZER_CONFIG
    try:
        tokenizer_config = getattr(
            _get_config_module(app_path), "TOKENIZER_CONFIG", DEFAULT_TOKENIZER_CONFIG
        )
        return tokenizer_config
    except (OSError, IOError, AttributeError):
        logger.info("No app configuration file found.")
        return DEFAULT_TOKENIZER_CONFIG
