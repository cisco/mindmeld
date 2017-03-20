# -*- coding: utf-8 -*-
"""
This module contains the role classifier component.
"""

import os

# from sklearn.externals import joblib


class RoleClassifier(object):
    """A role classifier is used to determine the target role for entities in a given query. It is
    trained using all the labeled queries for a particular intent. The labels are the role
    names associated with each entity within each query.

    Attributes:
        domain (str): The domain of this role classifier
        intent (str): The intent of this role classifier
        entity_type (str): The entity type of this role classifier
        roles (set): A set containing the roles which can be classified

    """

    DEFAULT_CONFIG = {
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

    def __init__(self, resource_loader, domain, intent, entity_type):
        """Initializes a role classifier

        Args:
            resource_loader (ResourceLoader): An object which can load resources for the classifier
            domain (str): The domain of this role classifier
            intent (str): The intent of this role classifier
            entity_type (str): The entity type of this role classifier
        """
        self._resource_loader = resource_loader
        self.domain = domain
        self.intent = intent
        self.entity_type = entity_type
        self.roles = set()
        self._model = None  # will be set when model is fit or loaded

    def fit(self, model_type=None, features=None, params_grid=None, cv=None):
        """Trains the model

        Args:
            model_type (str): The type of model to use
            features (None, optional): Description
            params_grid (None, optional): Description
            cv (None, optional): Description

        """
        # query_tree = self._resource_loader.get_labeled_queries(domain=self.domain,
        #                                                        intent=self.intent)
        # self._model = something
        pass

    def predict(self, query, entities, entity):
        """Predicts a role for the specified query

        Args:
            query (Query): The input query
            entities (list): The entities in the query
            entity (Entity): The entity whose role should be classified

        Returns:
            list: a list containing the corresponding roles for the entities passed in
        """
        pass

    def predict_proba(self, query, entities, entity):
        """Generates multiple hypotheses and returns their associated probabilities

        Args:
            query (Query): The input query
            entities (list): The entities in the query
            entity (Entity): The entity whose role should be classified

        Returns:
            list: a list of tuples of the form (str, float) grouping roles and their probabilities
        """
        pass

    def evaluate(self, use_blind=False):
        """Evaluates the model on the specified data

        Returns:
            TYPE: Description
        """
        pass

    def dump(self, model_path):
        """Persists the model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored

        """
        folder, filename = os.path.split(model_path)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        # joblib.dump(self._model, model_path)
        pass

    def load(self, model_path):
        """Loads the model from disk

        Args:
            model_path (str): The location on disk where the model is stored

        """
        # self._model = joblib.load(model_path)
        pass
