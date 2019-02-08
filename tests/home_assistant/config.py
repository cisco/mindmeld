# -*- coding: utf-8 -*-
"""This module contains the Home Assistant Blueprint Application"""
NLP_CONFIG = {
    'resolve_entities_using_nbest_transcripts': []
}

INTENT_MODEL_CONFIG = {
    'model_type': 'text',
    'model_settings': {
        'classifier_type': 'logreg'
    },
    'param_selection': {
        'type': 'k-fold',
        'k': 5,
        'grid': {
            'fit_intercept': [True, False],
            'C': [0.01, 1, 10, 100],
            'class_bias': [0.7, 0.3, 0]
        }
    },
    'features': {
        "bag-of-words": {
            "lengths": [1, 2]
        },
        "edge-ngrams": {"lengths": [1, 2]},
        "in-gaz": {},
        "exact": {"scaling": 10},
        "gaz-freq": {},
        "freq": {"bins": 5}
    }
}

DOMAIN_MODEL_CONFIG = {
    'model_type': 'text',
    'model_settings': {
        'classifier_type': 'logreg'
    },
    'params': {
        'C': 10,
    },
    'features': {
        "bag-of-words": {
            "lengths": [1, 2]
        },
        "edge-ngrams": {"lengths": [1, 2]},
        "in-gaz": {},
        "exact": {"scaling": 10},
        "gaz-freq": {},
        "freq": {"bins": 5},
        "average-token-length": {}  # Custom feature
    }
}
