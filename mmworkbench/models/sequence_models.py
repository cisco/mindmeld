# -*- coding: utf-8 -*-
"""This module contains the Memm entity recognizer."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from builtins import range, super

import logging
import random

from .helpers import register_model
from .model import EvaluatedExample, ModelConfig, EntityModelEvaluation, Model
from .taggers.crf import ConditionalRandomFields
from .taggers.memm import MemmModel

logger = logging.getLogger(__name__)

# classifier types
CRF_TYPE = 'crf'
MEMM_TYPE = 'memm'

DEFAULT_FEATURES = {
    'bag-of-words-seq': {
        'ngram_lengths_to_start_positions': {
            1: [-2, -1, 0, 1, 2],
            2: [-2, -1, 0, 1]
        }
    },
    'in-gaz-span-seq': {},
    'sys-candidates-seq': {
        'start_positions': [-1, 0, 1]
    }
}


class TaggerModel(Model):
    """A machine learning classifier for tags.

    This class manages feature extraction, training, cross-validation, and
    prediction. The design goal is that after providing initial settings like
    hyperparameters, grid-searchable hyperparameters, feature extractors, and
    cross-validation settings, TaggerModel manages all of the details
    involved in training and prediction

    TODO: define the inputs and outputs
    such that the input to training or
    prediction is Query objects, and the output is class names, and no data
    manipulation is needed from the client.

    Attributes:
        classifier_type (str): The name of the classifier type. Currently
            recognized values are "memm", "crf", and "lstm".
        hyperparams (dict): A kwargs dict of parameters that will be used to
            initialize the classifier object.
        grid_search_hyperparams (dict): Like 'hyperparams', but the values are
            lists of parameters. The training process will grid search over the
            Cartesian product of these parameter lists and select the best via
            cross-validation.
        feat_specs (dict): A mapping from feature extractor names, as given in
            FEATURE_NAME_MAP, to a kwargs dict, which will be passed into the
            associated feature extractor function.
        cross_validation_settings (dict): A dict that contains "type", which
            specifies the name of the cross-validation strategy, such as
            "k-folds" or "shuffle". The remaining keys are parameters
            specific to the cross-validation type, such as "k" when the type is
            "k-folds".

    """

    def __init__(self, config):
        if not config.features:
            config_dict = config.to_dict()
            config_dict['features'] = DEFAULT_FEATURES
            config = ModelConfig(**config_dict)

        super().__init__(config)

        self._no_entities = False

    def __getstate__(self):
        """Returns the information needed pickle an instance of this class.

        By default, pickling removes attributes with names starting with
        underscores. This overrides that behavior. For the _resources field,
        we save the resources that are memory intensive
        """
        attributes = self.__dict__.copy()
        resources_to_persist = set(['sys_types'])
        for key in list(attributes['_resources'].keys()):
            if key not in resources_to_persist:
                del attributes['_resources'][key]
        return attributes

    def fit(self, examples, labels, params=None):
        """Trains the model

        Args:
            examples (list of mmworkbench.core.Query): a list of queries to train on
            labels (list of tuples of mmworkbench.core.QueryEntity): a list of expected labels
            params (dict): Parameters of the classifier
        """
        system_types = self._get_system_types()
        self.register_resources(sys_types=system_types)

        skip_param_selection = params is not None or self.config.param_selection is None
        params = params or self.config.params

        # Shuffle to prevent order effects
        indices = list(range(len(labels)))
        random.shuffle(indices)
        examples = [examples[i] for i in indices]
        labels = [labels[i] for i in indices]

        """
        # TODO: add this code back in
        # distinct_labels = set(labels)
        # if len(set(distinct_labels)) <= 1:
        #     return None

        if len(set(y)) == 1:
            self._no_entities = True
            return self
        """
        if skip_param_selection:
            self._clf = self._fit(examples, labels, params)
            self._current_params = params
        else:
            # run cross validation to select params
            best_clf, best_params = self._fit_cv(examples, labels)
            self._clf = best_clf
            self._current_params = best_params

        return self

    def _fit(self, examples, labels, params):
        """Trains a classifier without cross-validation.

        Args:
            examples (list of mmworkbench.core.Query): a list of queries to train on
            labels (list of tuples of mmworkbench.core.QueryEntity): a list of expected labels
            params (dict): Parameters of the classifier
        """
        model_class = self._get_model_constructor()
        return model_class(self.config).fit(examples, labels, resources=self._resources)

    def _fit_cv(self, examples, labels):
        raise NotImplementedError

    def predict(self, examples):
        """
        Args:
            examples (list of mmworkbench.core.Query): a list of queries to train on

        Returns:
            (list of tuples of mmworkbench.core.QueryEntity): a list of predicted labels
        """
        if self._no_entities:
            # TODO
            return
        return self._clf.predict(examples)

    def evaluate(self, examples, labels):
        """Evaluates a model against the given examples and labels

        Args:
            examples: A list of examples to predict
            labels: A list of expected labels

        Returns:
            ModelEvaluation: an object containing information about the
                evaluation
        """
        # TODO: also expose feature weights?
        predictions = self.predict(examples)
        evaluations = [EvaluatedExample(e, labels[i], predictions[i], None, self.config.label_type)
                       for i, e in enumerate(examples)]

        config = self._get_effective_config()
        model_eval = EntityModelEvaluation(config, evaluations)
        return model_eval

    def _get_system_types(self):
        sys_types = set()
        for gaz in self._resources['gazetteers'].values():
            sys_types.update(gaz['sys_types'])
        return sys_types

    def _get_model_constructor(self):
        """Returns the python class of the actual underlying model"""
        classifier_type = self.config.model_settings['classifier_type']
        try:
            return {
                MEMM_TYPE: MemmModel,
                CRF_TYPE: ConditionalRandomFields,
                # LSTM_TYPE: LSTMModel,
            }[classifier_type]
        except KeyError:
            msg = '{}: Classifier type {!r} not recognized'
            raise ValueError(msg.format(self.__class__.__name__, classifier_type))


register_model('tag', TaggerModel)
