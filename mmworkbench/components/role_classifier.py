# -*- coding: utf-8 -*-
"""
This module contains the role classifier component of the Workbench natural language processor.
"""

import os

# from sklearn.externals import joblib


class RoleClassifier(object):
    """A role classifier is used to determine the target role for entities in a given query. It is
    trained using all the labeled queries for a particular intent. The labels are the role names
    associated with each entity within each query.

    Attributes:
        domain (str): The domain that this role classifier belongs to
        intent (str): The intent that this role classifier belongs to
        entity_type (str): The entity type that this role classifier is for
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
            domain (str): The domain that this role classifier belongs to
            intent (str): The intent that this role classifier belongs to
            entity_type (str): The entity type that this role classifier is for
        """
        self._resource_loader = resource_loader
        self.domain = domain
        self.intent = intent
        self.entity_type = entity_type
        self.roles = set()
        self._model = None  # will be set when model is fit or loaded

    def fit(self, model_type=None, features=None, params_grid=None, cv=None):
        """Trains a statistical model for role classification using the provided training examples

        Args:
            model_type (str): The type of machine learning model to use. If omitted, the default
                model type will be used.
            features (dict): Features to extract from each example instance to form the feature
                vector used for model training. If omitted, the default feature set for the model
                type will be used.
            params_grid (dict): The grid of hyper-parameters to search, for finding the optimal
                hyper-parameter settings for the model. If omitted, the default hyper-parameter
                search grid will be used.
            cv (None, optional): Cross-validation settings
        """
        # query_tree = self._resource_loader.get_labeled_queries(domain=self.domain,
        #                                                        intent=self.intent)
        # self._model = something
        pass

    def predict(self, query, entities, entity):
        """Predicts a role for the given entity using the trained role classification model

        Args:
            query (Query): The input query
            entities (list): The entities in the query
            entity (Entity): The entity whose role should be classified

        Returns:
            str: The predicted role for the provided entity
        """
        pass

    def predict_proba(self, query, entities, entity):
        """Runs prediction on a given entity and generates multiple role hypotheses with their
        associated probabilities using the trained role classification model

        Args:
            query (Query): The input query
            entities (list): The entities in the query
            entity (Entity): The entity whose role should be classified

        Returns:
            list: a list of tuples of the form (str, float) grouping roles and their probabilities
        """
        pass

    def evaluate(self, use_blind=False):
        """Evaluates the trained role classification model on the given test data

        Returns:
            ModelEvaluation: A ModelEvaluation object that contains evaluation results
        """
        pass

    def dump(self, model_path):
        """Persists the trained role classification model to disk.

        Args:
            model_path (str): The location on disk where the model should be stored
        """
        folder, filename = os.path.split(model_path)
        if not os.path.isdir(folder):
            os.makedirs(folder)
        # joblib.dump(self._model, model_path)
        pass

    def load(self, model_path):
        """Loads the trained role classification model from disk

        Args:
            model_path (str): The location on disk where the model is stored
        """
        # self._model = joblib.load(model_path)
        pass
