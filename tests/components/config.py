INTENT_CLASSIFIER_CONFIG = {
    "model_type": "text",
    "model_settings": {"classifier_type": "logreg"},
    "param_selection": {
        "type": "k-fold",
        "k": 5,
        "grid": {
            "fit_intercept": [True, False],
            "C": [1, 20, 300],
            "class_bias": [1, 0],
        },
    },
    "features": {
        "bag-of-words": {"lengths": [1]},
        "in-gaz": {},
        "freq": {"bins": 5},
        "length": {},
    },
}


def get_entity_recognizer_config(domain, intent):
    if intent == "error":
        raise ValueError("Some error because config provider is not robust")

    return {
        "model_type": "tagger",
        "label_type": "entities",
        "model_settings": {
            "classifier_type": "memm",
            "tag_scheme": "IOB",
            "feature_scaler": "max-abs",
        },
        "params": {"penalty": "l2", "C": 100},
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


ENTITY_RECOGNIZER_CONFIG = {
    "model_type": "tagger",
    "label_type": "entities",
    "model_settings": {
        "classifier_type": "memm",
        "tag_scheme": "IOB",
        "feature_scaler": "max-abs",
    },
    "params": {"error": "intent", "penalty": "l2", "C": 100},
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
