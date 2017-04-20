# -*- coding: utf-8 -*-
"""
This module contains the Config class.
"""
from __future__ import unicode_literals
from builtins import object

import copy
import logging

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
    'max_instances': 1,
    'precedence': 'left'
}


def __init__(self, app_path):
    self._config = _get_config_module(app_path)


def get_classifier_config(clf_type, domain=None, intent=None, entity=None):
    pass


def get_parser_config(app_path, config=None):
    try:
        config = config or _get_config_module(app_path).PARSER_CONFIG
    except AttributeError:
        return None
    return _expand_parser_config(config)


def _expand_parser_config(config):
    return {head: _expand_group_config(group) for head, group in config.items()}


def _expand_group_config(group_config):
    """Expands the group config.

    A group config can either be a list of dependents or an object with a
    'dependents' field containing that list.

    A dependent can either be a string containing the name of the entity type
    or an object with at least a type field.


    Some example parser configs follow

    A very simple configuration

       {
           'head': ['dependent']
       }

    A more realistic simple config

        {
            'product|beverage': ['size', 'quantity', 'option|beverage'],
            'product|baked-good': ['size', 'quantity', 'option|baked-good'],
            'store': ['location'],
            'option': ['size']
        }

    A fully specified config

        {
            'product': {
                'role': None,
                'dependents': [{
                    'type': 'quantity',
                    'role': None,
                    'left': True,
                    'right': True,
                    'right_distance': 1
                    'precedence': 'left',
                    'min_instances': 0,
                    'max_instances': 3
                }, {
                    'type': 'size',
                    'role': None,
                    'left': True,
                    'right': True,
                    'precedence': 'left',
                    'min_instances': 0,
                    'max_instances': 1
                }, {
                    'type': 'option',
                    'role': None,
                    'left': True,
                    'right': True,
                    'precedence': 'left',
                    'min_instances': 0,
                    'max_instances': 1
                    'dependents': [ {
                        'type': 'size',
                        'role': None,
                        'left': True,
                        'right': True,
                        'precedence': 'left',
                        'min_instances': 0,
                        'max_instances': 1
                    } ]
                }],
            },
            'store': {
                'role': None,
                'dependents': [{
                    'type': 'location',
                    'role': None,
                    'left': True,
                    'right': True,
                    'precedence': 'left',
                    'min_instances': 0,
                    'max_instances': 1
                }],
            }
            'option': {
                'role': None,
                'dependents': [{
                    'type': 'size',
                    'role': None,
                    'left': True,
                    'right': True,
                    'precedence': 'left',
                    'min_instances': 0,
                    'max_instances': 1
                }]
            }
        }
    """

    if isinstance(group_config, (tuple, list)):
        dependents = group_config
        group_config = {
            'dependents': dependents
        }
    else:
        dependents = group_config['dependents']

    exp_dependents = []
    for dependent in dependents:
        config = copy.copy(DEFAULT_PARSER_DEPENDENT_CONFIG)
        config['type'] = dependent
        try:
            config.update(dependent)
        except ValueError:
            # simple style config -- dependent is a str
            pass
        exp_dependents.append(config)

    group_config['dependents'] = exp_dependents
    return copy.deepcopy(group_config)


def _get_config_module(app_path):
    module_path = path.get_config_module_path(app_path)

    import imp
    config_module = imp.load_source('config_module', module_path)
    return config_module
