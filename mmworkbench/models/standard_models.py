# coding=utf-8
"""
This module contains all code required to perform multinomial classification
of text.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from builtins import range, zip, super
from past.utils import old_div

from collections import Counter
import logging

from numpy import bincount

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .model import SkLearnModel

from .helpers import GAZETTEER_RSC, QUERY_FREQ_RSC, WORD_FREQ_RSC, register_model, mask_numerics

_NEG_INF = -1e10

# classifier types
LOG_REG_TYPE = "logreg"
DECISION_TREE_TYPE = "dtree"
RANDOM_FOREST_TYPE = "rforest"
SVM_TYPE = "svm"
SUPER_LEARNER_TYPE = "super-learner"
BASE_MODEL_TYPES = [LOG_REG_TYPE, DECISION_TREE_TYPE, RANDOM_FOREST_TYPE, SVM_TYPE]

# model scoring types
ACCURACY_SCORING = "accuracy"
LIKELIHOOD_SCORING = "log_loss"


logger = logging.getLogger(__name__)


class TextModel(SkLearnModel):
    """A machine learning classifier for text.

    This class manages feature extraction, training, cross-validation, and
    prediction. The design goal is that after providing initial settings like
    hyperparameters, grid-searchable hyperparameters, feature extractors, and
    cross-validation settings, TextModel manages all of the details
    involved in training and prediction such that the input to training or
    prediction is Query objects, and the output is class names, and no data
    manipulation is needed from the client.

    Attributes:
        classifier_type (str): The name of the classifier type. Currently
            recognized values are "logreg","dtree", "rforest" and "svm",
            as well as "super-learner:logreg", "super-learner:dtree" etc.
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
        super().__init__(config)
        self._meta_type = None
        self._meta_feat_vectorizer = DictVectorizer(sparse=False)
        self._base_clfs = {}
        self.cv_loss_ = None
        self.train_acc_ = None

    def __getstate__(self):
        """Returns the information needed pickle an instance of this class.

        By default, pickling removes attributes with names starting with
        underscores. This overrides that behavior.
        """
        attributes = self.__dict__.copy()
        attributes['_resources'] = {WORD_FREQ_RSC: self._resources.get(WORD_FREQ_RSC, {}),
                                    QUERY_FREQ_RSC: self._resources.get(QUERY_FREQ_RSC, {})}
        return attributes

    def register_resources(self, gazetteers=None, word_freqs=None, query_freqs=None):
        """Loads resources that are built outside the classifier, e.g. gazetteers

        Args:
            gazetteers (dict of Gazetteer): domain gazetteer data
            word_freqs (dict of int): unigram frequencies in queries
            query_freqs (dict of int): whole query index with frequencies
        """
        if gazetteers is not None:
            self._resources[GAZETTEER_RSC] = gazetteers
            if self._meta_type == SUPER_LEARNER_TYPE:
                for model in self._base_clfs.values():
                    model.register_resources(gazetteers=gazetteers)
        if word_freqs is not None:
            self._resources[WORD_FREQ_RSC] = word_freqs
            if self._meta_type == SUPER_LEARNER_TYPE:
                for model in self._base_clfs.values():
                    model.register_resources(word_freqs=word_freqs)
        if query_freqs is not None:
            self._resources[QUERY_FREQ_RSC] = query_freqs
            if self._meta_type == SUPER_LEARNER_TYPE:
                for model in self._base_clfs.values():
                    model.register_resources(query_freqs=query_freqs)

    def compile_word_freq_dict(self, queries):
        """Compiles unigram frequency dictionary of normalized query tokens

        Args:
            queries (list of Query): A list of all queries
        """
        # Unigram frequencies
        tokens = [mask_numerics(tok) for q in queries
                  for tok in q.normalized_tokens]
        freq_dict = Counter(tokens)

        self.register_resources(word_freqs=freq_dict)

    def compile_query_freq_dict(self, queries):
        """Compiles frequency dictionary of normalized query strings

        Args:
            queries (list of Query): A list of all queries
        """
        # Whole query frequencies, with singletons removed
        query_dict = Counter([u'<{}>'.format(q.normalized_text) for q in queries])
        for query in query_dict:
            if query_dict[query] < 2:
                query_dict[query] = 0
        query_dict += Counter()

        self.register_resources(query_freqs=query_dict)

    def _get_model_constructor(self):
        """Returns the class of the actual underlying model"""
        classifier_type = self.config.model_settings['classifier_type']
        try:
            return {LOG_REG_TYPE: LogisticRegression,
                    DECISION_TREE_TYPE: DecisionTreeClassifier,
                    RANDOM_FOREST_TYPE: RandomForestClassifier,
                    SVM_TYPE: SVC}[classifier_type]
        except KeyError:
            msg = '{}: Classifier type {!r} not recognized'
            raise ValueError(msg.format(self.__class__.__name__, classifier_type))

    def fit(self, examples, labels, params=None):
        """Trains this TextModel.

        This method inspects instance attributes to determine the classifier
        object and cross-validation strategy, and then fits the model to the
        training examples passed in.

        Args:
            queries (list of Query): A list of queries.
            classes (list of str): A parallel list to queries. The gold labels
                for each query.
            verbose (bool): Whether to show analysis output

        Returns:
            (TextModel): Returns self to match classifier scikit-learn
                interfaces.
        """
        if self._resources.get(WORD_FREQ_RSC) is None and self.requires_resource(WORD_FREQ_RSC):
            self.compile_word_freq_dict(examples)
        if self._resources.get(QUERY_FREQ_RSC) is None and self.requires_resource(QUERY_FREQ_RSC):
            self.compile_query_freq_dict(examples)

        return super().fit(examples, labels, params)

    def _convert_params(self, param_grid, y):
        """
        Convert the params from the style given by the config to the style
        passed in to the actual classifier.

        Args:
            param_grid (dict): lists of classifier parameter values, keyed by parameter name

        Returns:
            (dict): revised param_grid
        """
        class_count = bincount(y)
        classes = self._class_encoder.classes_

        if 'class_weight' in param_grid:
            param_grid['class_weight'] = [{k if isinstance(k, int) else
                                           self._class_encoder.transform(k): v
                                           for k, v in cw_dict.items()}
                                          for cw_dict in param_grid['class_weight']]
        elif 'class_bias' in param_grid:
            # interpolate between class_bias=0 => class_weight=None
            # and class_bias=1 => class_weight='balanced'
            param_grid['class_weight'] = []
            for class_bias in param_grid['class_bias']:
                # these weights are same as sklearn's class_weight='balanced'
                balanced_w = [old_div(len(y), (float(len(classes)) * c))
                              for c in class_count]
                balanced_tuples = list(zip(list(range(len(classes))), balanced_w))

                param_grid['class_weight'].append({c: (1 - class_bias) + class_bias * w
                                                   for c, w in balanced_tuples})
            del param_grid['class_bias']

        return param_grid

    def _get_feature_selector(self):
        return None


register_model('text', TextModel)
