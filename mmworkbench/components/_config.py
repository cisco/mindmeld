# -*- coding: utf-8 -*-
"""
This module contains the Config class.
"""
from __future__ import absolute_import, unicode_literals

import copy
import logging
import os

from .. import path


logger = logging.getLogger(__name__)

DEFAULT_DOMAIN_MODEL_CONFIG = {
    'model_type': 'text',
    'model_settings': {
        'classifier_type': 'logreg',
    },
    'param_selection': {
        'type': 'k-fold',
        'k': 10,
        'grid': {
            'fit_intercept': [True, False],
            'C': [10, 100, 1000, 10000, 100000]
        },
    },
    'features': {
        'bag-of-words': {
            'lengths': [1]
        },
        'freq': {'bins': 5},
        'in-gaz': {}
    }
}

DEFAULT_INTENT_MODEL_CONFIG = {
    'model_type': 'text',
    'model_settings': {
        'classifier_type': 'logreg'
    },
    'param_selection': {
        'type': 'k-fold',
        'k': 10,
        'grid': {
            'fit_intercept': [True, False],
            'C': [0.01, 1, 100, 10000, 1000000],
            'class_bias': [1, 0.7, 0.3, 0]
        }
    },
    'features': {
        'bag-of-words': {
            'lengths': [1]
        },
        'in-gaz': {},
        'freq': {'bins': 5},
        'length': {}
    }
}

DEFAULT_ENTITY_MODEL_CONFIG = {
    'model_type': 'tagger',
    'label_type': 'entities',
    'model_settings': {
        'classifier_type': 'lstm',
        'tag_scheme': 'IOB',
        'feature_scaler': 'max-abs'
    },
    'params': {
        'padding_length': 19,
        'batch_size': 128,
        'display_step': 20,
        'maximum_number_of_epochs': 40,
        'optimizer': 'adam',
        'learning_rate': 0.005,
        'dropout_rate': 0.7,
        'token_lstm_hidden_state_dimension': 300,
        'token_embedding_dimension': 300,
        'token_pretrained_embedding_filepath': 'glove.6B.300d.txt'
    },
    'features': {
        'in-gaz-span-seq': {},
    }
}

DEFAULT_ENTITY_RESOLUTION_CONFIG = {
    'model_type': 'text_relevance'
}

DOC_TYPE = 'document'

# ElasticSearch mapping to define text analysis settings for text fields.
# It defines specific index configuration for synonym indices. The common index configuration
# is in default index template.
DEFAULT_ES_SYNONYM_MAPPING = {
    "mappings": {
        DOC_TYPE: {
            "properties": {
                "sort_factor": {
                    "type": "double"
                },
                "whitelist": {
                    "type": "nested",
                    "properties": {
                        "name": {
                            "type": "text",
                            "fields": {
                                "raw": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                },
                                "normalized_keyword": {
                                    "type": "text",
                                    "analyzer": "keyword_match_analyzer"
                                },
                                "char_ngram": {
                                    "type": "text",
                                    "analyzer": "char_ngram_analyzer"
                                }
                            },
                            "analyzer": "default_analyzer"
                        }
                    }
                }
            }
        }
    }
}

DEFAULT_ROLE_MODEL_CONFIG = {
    'model_type': 'text',
    'model_settings': {
        'classifier_type': 'logreg'
    },
    'params': {
        'C': 100,
        'penalty': 'l1'
    },
    'features': {
        'bag-of-words-before': {
            'ngram_lengths_to_start_positions': {
                1: [-2, -1],
                2: [-2, -1]
            }
        },
        'bag-of-words-after': {
            'ngram_lengths_to_start_positions': {
                1: [0, 1],
                2: [0, 1]
            }
        },
        'other-entities': {}
    }
}

DEFAULT_ES_INDEX_TEMPLATE_NAME = "default"

# Default ES index template that contains the base index configuration shared across different
# types of indices. Currently all ES indices will be created using this template.
# - custom text analysis settings such as custom analyzers, token filters and character filters.
# - dynamic field mapping template for text fields
# - common fields, e.g. id.
DEFAULT_ES_INDEX_TEMPLATE = {
    "template": "*",
    "mappings": {
        DOC_TYPE: {
            "dynamic_templates": [
                {
                    "default_text": {
                        "match": "*",
                        "match_mapping_type": "string",
                        "mapping": {
                            "type": "text",
                            "analyzer": "default_analyzer",
                            "fields": {
                                "raw": {
                                    "type": "keyword",
                                    "ignore_above": 256
                                },
                                "normalized_keyword": {
                                    "type": "text",
                                    "analyzer": "keyword_match_analyzer"
                                },
                                "char_ngram": {
                                    "type": "text",
                                    "analyzer": "char_ngram_analyzer"
                                }
                            }
                        }
                    }
                }
            ],
            "properties": {
                "id": {
                    "type": "keyword"
                }
            }
        }
    },
    "settings": {
        "analysis": {
            "char_filter": {
                "remove_loose_apostrophes": {
                    "pattern": " '|' ",
                    "type": "pattern_replace",
                    "replacement": ""
                },
                "space_possessive_apostrophes": {
                    "pattern": "([^\\p{N}\\s]+)'s ",
                    "type": "pattern_replace",
                    "replacement": "$1 's "
                },
                "remove_special_beginning": {
                    "pattern": "^[^\\p{L}\\p{N}\\p{Sc}&']+",
                    "type": "pattern_replace",
                    "replacement": ""
                },
                "remove_special_end": {
                    "pattern": "[^\\p{L}\\p{N}&']+$",
                    "type": "pattern_replace",
                    "replacement": ""
                },
                "remove_special1": {
                    "pattern": "([\\p{L}]+)[^\\p{L}\\p{N}&']+(?=[\\p{N}\\s]+)",
                    "type": "pattern_replace",
                    "replacement": "$1 "
                },
                "remove_special2": {
                    "pattern": "([\\p{N}]+)[^\\p{L}\\p{N}&']+(?=[\\p{L}\\s]+)",
                    "type": "pattern_replace",
                    "replacement": "$1 "
                },
                "remove_special3": {
                    "pattern": "([\\p{L}]+)[^\\p{L}\\p{N}&']+(?=[\\p{L}]+)",
                    "type": "pattern_replace",
                    "replacement": "$1 "
                },
                "remove_comma": {
                    "pattern": ",",
                    "type": "pattern_replace",
                    "replacement": ""
                },
                "remove_tm_and_r": {
                    "pattern": "™|®",
                    "type": "pattern_replace",
                    "replacement": ""
                }
            },
            "filter": {
                "token_shingle": {
                    "max_shingle_size": "4",
                    "min_shingle_size": "2",
                    "output_unigrams": "true",
                    "type": "shingle"
                },
                "ngram_filter": {
                    "type": "ngram",
                    "min_gram": "3",
                    "max_gram": "3"
                }
            },
            "analyzer": {
                "default_analyzer": {
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "token_shingle"
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
                        "remove_special3"
                    ],
                    "type": "custom",
                    "tokenizer": "whitespace"
                },
                "keyword_match_analyzer": {
                    "filter": [
                        "lowercase",
                        "asciifolding"
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
                        "remove_special3"
                    ],
                    "type": "custom",
                    "tokenizer": "keyword"
                },
                "char_ngram_analyzer": {
                    "filter": [
                        "lowercase",
                        "asciifolding",
                        "ngram_filter"
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
                        "remove_special3"
                    ],
                    "type": "custom",
                    "tokenizer": "whitespace"
                }
            }
        }
    }
}


# Elasticsearch mapping to define knowledge base index specific configuration:
# - dynamic field mapping to index all synonym whitelist in fields with "$whitelist" suffix.
# - location field
#
# The common configuration is defined in default index template
DEFAULT_ES_QA_MAPPING = {
    "mappings": {
        DOC_TYPE: {
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
                                        "raw": {
                                            "type": "keyword",
                                            "ignore_above": 256
                                        },
                                        "normalized_keyword": {
                                            "type": "text",
                                            "analyzer": "keyword_match_analyzer"
                                        },
                                        "char_ngram": {
                                            "type": "text",
                                            "analyzer": "char_ngram_analyzer"
                                        }
                                    },
                                    "analyzer": "default_analyzer"
                                }
                            }
                        }
                    }
                }
            ],
            "properties": {
                "location": {
                    "type": "geo_point"
                }
            }
        }
    }
}

DEFAULT_PARSER_DEPENDENT_CONFIG = {
    'left': True,
    'right': True,
    'min_instances': 0,
    'max_instances': None,
    'precedence': 'left',
    'linking_words': frozenset()
}

DEFAULT_RANKING_CONFIG = {
    'query_clauses_operator': 'or'
}


def get_app_namespace(app_path):
    """Returns the namespace of the application at app_path"""
    try:
        _app_namespace = _get_config_module(app_path).APP_NAMESPACE
        if 'JUPYTER_USER' in os.environ:
            _app_namespace = '{}_{}'.format(os.environ['JUPYTER_USER'], _app_namespace)
        return _app_namespace
    except (OSError, IOError):
        logger.debug('No app configuration file found')
    except AttributeError:
        logger.debug('App namespace not set in app configuration')

    _app_namespace = os.path.split(app_path)[1]
    if 'JUPYTER_USER' in os.environ:
        _app_namespace = '{jupyter_user}_{app_namespace}'.format(
            jupyter_user=os.environ['JUPYTER_USER'], app_namespace=_app_namespace)
    return _app_namespace


def get_classifier_config(clf_type, app_path=None, domain=None, intent=None, entity=None):
    """Returns the application config if it exists, otherwise returns the default config."""
    try:
        module_conf = _get_config_module(app_path)

        attribute = {
            'domain': 'DOMAIN_MODEL_CONFIG',
            'intent': 'INTENT_MODEL_CONFIG',
            'entity': 'ENTITY_MODEL_CONFIG',
            'entity_resolution': 'ENTITY_RESOLUTION_CONFIG',
            'role': 'ROLE_MODEL_CONFIG',
        }[clf_type]
        return copy.deepcopy(getattr(module_conf, attribute))
    except (OSError, IOError):
        logger.info('No app configuration file found. Using default %s model configuration',
                    clf_type)
    except AttributeError:
        logger.info('No %s model configuration set. Using default.', clf_type)

    return copy.deepcopy({
        'domain': DEFAULT_DOMAIN_MODEL_CONFIG,
        'intent': DEFAULT_INTENT_MODEL_CONFIG,
        'entity': DEFAULT_ENTITY_MODEL_CONFIG,
        'entity_resolution': DEFAULT_ENTITY_RESOLUTION_CONFIG,
        'role': DEFAULT_ROLE_MODEL_CONFIG
    }[clf_type])


def get_parser_config(app_path=None, config=None):
    try:
        config = config or _get_config_module(app_path).PARSER_CONFIG
    except AttributeError:
        return None
    return _expand_parser_config(config)


def _expand_parser_config(config):
    return {head: _expand_group_config(group) for head, group in config.items()}


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
                dep_type = dependent.pop('type')
                config.update(dependent)
            except (AttributeError, ValueError):
                # simple style config -- dependent is a str
                dep_type = dependent
                pass
            expanded[dep_type] = config
    else:
        for dep_type, dep_config in group_config.items():
            config = copy.copy(DEFAULT_PARSER_DEPENDENT_CONFIG)
            dep_config.pop('type', None)
            config.update(dep_config)
            expanded[dep_type] = config
    return expanded


def _get_config_module(app_path):
    module_path = path.get_config_module_path(app_path)

    import imp
    config_module = imp.load_source('config_module', module_path)
    return config_module
