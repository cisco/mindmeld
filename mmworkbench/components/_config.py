# -*- coding: utf-8 -*-
"""
This module contains the Config class.
"""
from __future__ import unicode_literals

import copy
import logging
import os

from .. import path


logger = logging.getLogger(__name__)

DEFAULT_DOMAIN_CONFIG = {
    'default_model': 'main',
    'models': {
        'main': {
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
    }
}

DEFAULT_INTENT_CONFIG = {
    'default_model': 'main',
    'models': {
        'main': {
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
        },
        'rforest': {
            'model_type': 'text',
            'model_settings': {
                'classifier_type': 'rforest'
            },
            'param_selection': {
                'type': 'k-fold',
                'k': 10,
                'grid': {
                    'n_estimators': [10],
                    'max_features': ['auto'],
                    'n_jobs': [-1]
                },
            },
            'features': {
                'bag-of-words': {
                    'lengths': [1, 2, 3]
                },
                'edge-ngrams': {'lengths': [1, 2, 3]},
                'in-gaz': {},
                'freq': {'bins': 5},
                'length': {}
            }
        }
    }
}

DEFAULT_ENTITY_CONFIG = {
    'default_model': 'memm-cv',
    'models': {
        'main': {
            'model_type': 'memm',
            'model_settings': {
                'tag_scheme': 'IOB',
                'feature_scaler': 'none'
            },
            'params': {
                'penalty': 'l2',
                'C': 100
            },
            'features': {}  # use default
        },
        'sparse': {
            'model_type': 'memm',
            'model_settings': {
                'tag_scheme': 'IOB',
                'feature_scaler': 'max-abs',
                'feature_selector': 'l1'
            },
            'params': {
                'penalty': 'l2',
                'C': 100
            },
            'features': {}  # use default
        },
        'memm-cv': {
            'model_type': 'memm',
            'model_settings': {
                'tag_scheme': 'IOB',
                'feature_scaler': 'max-abs'
            },
            'param_selection': {
                'type': 'k-fold',
                'k': 5,
                'scoring': 'accuracy',
                'grid': {
                    'penalty': ['l1', 'l2'],
                    'C': [0.01, 1, 100, 10000, 1000000, 100000000]
                },
            },
            'features': {}  # use default
        }
    }
}

DEFAULT_ROLE_CONFIG = {
    'default_model': 'main',
    'main': {
        "classifier_type": "memm",
        "params_grid": {
            "C": [100]
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
            'in-gaz': {},
            'other-entities': {},
            'operator-entities': {},
            'age-entities': {}
        }
    },
    "sparse": {
        "classifier_type": "memm",
        "params_grid": {
            "penalty": ["l1"],
            "C": [1]
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
            'in-gaz': {},
            'other-entities': {},
            'operator-entities': {},
            'age-entities': {}
        }
    },
    "memm-cv": {
        "classifier_type": "memm",
        "params_grid": {
            "penalty": ["l1", "l2"],
            "C": [0.01, 1, 100, 10000, 1000000, 100000000]
        },
        "cv": {
            "type": "k-fold",
            "k": 5,
            "metric": "accuracy"
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
            'in-gaz': {},
            'other-entities': {},
            'operator-entities': {},
            'age-entities': {}
        }
    },
    "ngram": {
        "classifier_type": "ngram",
        "params_grid": {
            "C": [100]
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
            'in-gaz': {},
            'other-entities': {},
            'operator-entities': {},
            'age-entities': {}
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


def get_app_name(app_path):
    try:
        app_name = _get_config_module(app_path).APP_NAME
    except (IOError, AttributeError):
        app_name = os.path.split(app_path)[1]
    return app_name


def get_classifier_config(clf_type, domain=None, intent=None, entity=None):
    pass


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
