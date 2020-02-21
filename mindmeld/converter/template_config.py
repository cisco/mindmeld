
DOMAIN_CLASSIFIER_CONFIG = {
    "model_type": "text",
    "model_settings": {"classifier_type": "logreg",},
    "param_selection": {
        "type": "k-fold",
        "k": 5,
        "grid": {"fit_intercept": [True, False], "C": [10, 100, 1000, 10000, 100000]},
    },
    "features": {"bag-of-words": {"lengths": [1]}, "freq": {"bins": 5}, "in-gaz": {}},
}

INTENT_CLASSIFIER_CONFIG = {
    "model_type": "text",
    "model_settings": {"classifier_type": "logreg"},
    "param_selection": {
        "type": "k-fold",
        "k": 5,
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

ENTITY_RECOGNIZER_CONFIG = {
    "model_type": "tagger",
    "label_type": "entities",
    "model_settings": {
        "classifier_type": "memm",
        "tag_scheme": "IOB",
        "feature_scaler": "max-abs",
    },
    "param_selection": {
        "type": "k-fold",
        "k": 3,
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
