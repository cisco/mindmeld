"""This module contains data structures used by the models subpackage."""
from __future__ import absolute_import
from __future__ import unicode_literals
from builtins import object

from collections import namedtuple
import copy
import itertools

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit


class ModelConfig(namedtuple('ModelConfig', ['model_type', 'model_settings', 'params_grid',
                                             'features', 'cv'])):
    """A simple named tuple containing a model configuration

    Attributes:
        model_type (str): The name of the classifier type. Currently
            recognized values are "logreg","dtree", "rforest" and "svm",
            as well as "super-learner:logreg", "super-learner:dtree" etc.
        params_grid (dict): A kwargs dict of parameters that will be used to
            initialize the classifier object. The value for each key is a list
            of candidate parameters. The training process will grid search over
            the Cartesian product of these parameter lists and select the best
            via cross-validation.
        feat_specs (dict): A mapping from feature extractor names, as given in
            FEATURE_NAME_MAP, to a kwargs dict, which will be passed into the
            associated feature extractor function.
        cv (dict): A dict that contains "type", which specifies the name of the
            cross-validation strategy, such as "k-folds" or "shuffle". The
            remaining keys are parameters specific to the cross-validation type,
            such as "k" when the type is "k-folds".

    """
    pass


class Model(object):
    """An abstract class upon which all models are based.

    Attributes:
        config (ModelConfig): The configuration for the model
    """

    def __init__(self, config):
        self.config = config

    def _iter_settings(self, params_grid):
        """Iterates through all model settings.

        Yields:
            (dict): A kwargs dict to be passed to the classifier object. Each
                item yielded is a unique combination of values from with
                a choice of settings from the hyper params grid.
        """
        if params_grid:
            for config in self.settings_for_params_grid({}, params_grid):
                yield config

    @staticmethod
    def settings_for_params_grid(base, params_grid):
        """Iterates through all model settings.

        Args:
            base (dict): A dictionary containing the base settings which all
                permutations contain
            params_grid (dict): A kwargs dict of parameters that will be used to
                initialize the classifier object. The value for each key is a
                list of candidate parameters. The training process will grid
                search over the Cartesian product of these parameter lists and
                select the best via cross-validation.

        Yields:
            (dict): A kwargs dict to be passed to an underlying model object.
                Each item yielded is a unique combination of values from with
                a choice of settings from the hyper params grid.
        """
        base = copy.deepcopy(base)
        keys, values = list(zip(*list(params_grid.items())))
        for settings in itertools.product(*values):
            base.update(dict(list(zip(keys, settings))))
            yield copy.deepcopy(base)

    def _get_feature_extractor(self, name, params):
        raise NotImplementedError

    def _extract_query_features(self, query):
        raise NotImplementedError

    def _k_fold_iterator(self, y):
        k = self.config.cv['k']
        return StratifiedKFold(y, n_folds=k, shuffle=True)

    def _shuffle_iterator(self, y):
        k = self.config.cv['k']
        n = self.config.cv.get('n', k)
        test_size = 1.0 / k
        return StratifiedShuffleSplit(y, n_iter=n, test_size=test_size)



