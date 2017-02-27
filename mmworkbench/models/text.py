# coding=utf-8
"""
This module contains all code required to perform multinomial classification
of text.
"""
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from builtins import next, range, str, zip
from past.utils import old_div

from collections import Counter, defaultdict
import copy
import itertools
import logging
import math
import re
import operator

from numpy import random, bincount, mean, std, Infinity
import numpy

from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.grid_search import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .core import Model

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

# resource/requirements names
GAZETTEER_RSC = "gaz"
WORD_FREQ_RSC = "w_freq"
QUERY_FREQ_RSC = "q_freq"

logger = logging.getLogger(__name__)


class TextModel(Model):
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

        self._is_fit = False
        self._feat_vectorizer = DictVectorizer(sparse=True)
        self._meta_feat_vectorizer = DictVectorizer(sparse=False)
        self._class_encoder = LabelEncoder()
        self._clf = None
        self._base_clfs = {}
        self._meta_type = None
        self._resources = {}
        self._queries = {}
        self.cv_loss_ = None
        self.train_acc_ = None

    @classmethod
    def from_config(cls, config, model_configuration):
        """Initializes a TextModel instance from config file data.

        Args:
            config (dict): The Python representation of a parsed config file.
            model_configuration (str): One of the keys in config['models'].

        Returns:
            (TextModel): A TextModel instance initialized with the
                settings from the config entry given by model_configuration.
        """
        model_config_entry = config['models'].get(model_configuration)
        model_type = model_config_entry.get('model-type',
                                            model_config_entry.get('classifier-type', ''))
        meta_type = None
        if ':' in model_type and model_type.split(':')[0] == SUPER_LEARNER_TYPE:
            meta_type = SUPER_LEARNER_TYPE
            model_type = model_type.split(':')[1]
        if not model_config_entry:
            error_msg = "Model config does not contain a model named '{}'"
            raise ValueError(error_msg.format(model_configuration))

        # Using ensemble method.
        classifiers = {}
        if meta_type == SUPER_LEARNER_TYPE:
            base_models = model_config_entry.get('base-models', [])
            if base_models:
                for base_model_conf in base_models:
                    config_entry = config['models'].get(base_model_conf)
                    if not config_entry:
                        error_msg = "Model config does not contain a model named '{}'"
                        raise ValueError(error_msg.format(model_configuration))
                    base_clf = cls(
                        config_entry.get('model-type', config_entry.get('classifier-type')),
                        config_entry.get('model-parameters', {}),
                        config_entry.get('model-parameter-choices', {}),
                        config_entry['features'],
                        # Cross-validation is not handled in the base models
                        None
                    )
                    classifiers[base_model_conf] = base_clf
            else:
                for base_type in BASE_MODEL_TYPES:
                    # We still need to define the base model's feature set.
                    if base_type == SVM_TYPE:
                        model_choices = {"probability": [True], "C": [100]}
                    else:
                        model_choices = {}
                    base_clf = cls(base_type, {}, model_choices,
                                   model_config_entry['base-model-features'],
                                   None)
                    classifiers[base_type] = base_clf

        if 'cross-validation-settings' not in model_config_entry:
            model_parameter_choices = model_config_entry.get('model-parameter-choices', {})
            for key in model_parameter_choices:
                if len(model_parameter_choices[key]) > 1:
                    raise ValueError('Cannot pass more than one model parameter choice if not '
                                     'using cross validation')

        model = cls(
            model_type,
            model_config_entry.get('model-parameters', {}),
            model_config_entry.get('model-parameter-choices', {}),
            model_config_entry.get('features', {}),
            model_config_entry.get('cross-validation-settings', None)
        )
        model._meta_type = meta_type
        if model._meta_type == SUPER_LEARNER_TYPE:
            model._base_clfs = classifiers

        return model

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

    def extract_features(self, query):
        """Gets all features from a query.

        Args:
            query (Query): A query instance.

        Returns:
            (dict of str: number): A dict of feature names to their values.
        """
        feat_set = {}
        query_text = query.normalized_text
        for name, kwargs in self.config.features.items():
            feat_extractor = FEATURE_NAME_MAP[name](**kwargs)
            feat_set.update(feat_extractor(query_text, self._resources))
        return feat_set

    @staticmethod
    def settings_for_params_grid(base, params_grid):
        base = copy.deepcopy(base)
        gsh_keys = list()
        gsh_keys, gsh_values = list(zip(*list(params_grid.items())))
        for settings in itertools.product(*gsh_values):
            base.update(dict(list(zip(gsh_keys, settings))))
            yield copy.deepcopy(base)

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

    def fit(self, queries, classes, verbose=True):
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
            self.compile_word_freq_dict(queries)
        if self._resources.get(QUERY_FREQ_RSC) is None and self.requires_resource(QUERY_FREQ_RSC):
            self.compile_query_freq_dict(queries)

        # Need to shuffle once to prevent order effects
        indices = list(range(len(classes)))
        random.shuffle(indices)
        classes = [classes[i] for i in indices]
        queries = [queries[i] for i in indices]
        self._queries = list(zip(queries, classes))

        classes_set = set(classes)
        if len(set(classes_set)) <= 1:
            return None

        y = self._class_encoder.fit_transform(classes)

        if self.config.model_type == LOG_REG_TYPE:
            clf_cls = LogisticRegression
        elif self.config.model_type == DECISION_TREE_TYPE:
            clf_cls = DecisionTreeClassifier
        elif self.config.model_type == RANDOM_FOREST_TYPE:
            clf_cls = RandomForestClassifier
        elif self.config.model_type == SVM_TYPE:
            clf_cls = SVC
        else:
            raise ValueError('Classifier type "{}" not recognized'
                             .format(self.config.model_type))

        if self.config.cv is None:
            # Fit without cross validation
            cv_iterator = None
        elif self.config.cv['type'] == 'k-fold':
            cv_iterator = self._k_fold_iterator
        elif self.config.cv['type'] == 'shuffle':
            cv_iterator = self._shuffle_iterator
        else:
            raise ValueError('CV iterator type "{}" not recognized'
                             .format(self.config.cv['type']))

        if (self.config.cv is not None and
                self.config.cv.get("scoring") == LIKELIHOOD_SCORING):
            scoring = LIKELIHOOD_SCORING
        else:
            scoring = ACCURACY_SCORING

        if self._meta_type == SUPER_LEARNER_TYPE:
            if cv_iterator and scoring == LIKELIHOOD_SCORING:
                # compute marginal likelihood contribution of each base classifier
                all_base_clfs = self._base_clfs
                losses = {}
                for clf in all_base_clfs:
                    self._base_clfs = {c: all_base_clfs[c] for c in all_base_clfs if c != clf}
                    self._fit_super_learner(clf_cls, cv_iterator,
                                            verbose=verbose, scoring=LIKELIHOOD_SCORING)
                    losses[clf] = self.cv_loss_
                self._base_clfs = all_base_clfs
                self._fit_super_learner(clf_cls, cv_iterator,
                                        verbose=verbose, scoring=LIKELIHOOD_SCORING)
                losses['all'] = self.cv_loss_
                diff_losses = {clf: losses[clf] - losses['all'] for clf in self._base_clfs}
                logger.info("Marginal likelihood contribution for each base model: {}"
                            .format(diff_losses))
            else:
                self._fit_super_learner(clf_cls, cv_iterator,
                                        verbose=verbose, scoring=scoring)
        else:
            X = self.get_feature_matrix(queries, fit=True)
            if cv_iterator is None:
                self._clf = self._fit(clf_cls, X, y)
            elif verbose is False:
                self._clf = self._fit_cv_grid(clf_cls, X, y, cv_iterator, scoring=scoring)
            else:
                self._clf = self._fit_cv_verbose(clf_cls, X, y, cv_iterator, scoring=scoring)

        pred_classes, pred_probs = self.predict_and_log_proba(queries)
        predictions = self._class_encoder.transform(pred_classes)
        self.train_acc_ = accuracy_score(y, predictions)
        logger.info("Final accuracy on training data: {:.1%}".format(self.train_acc_))
        if verbose:
            for idx in range(len(predictions)):
                if predictions[idx] != y[idx]:
                    logger.debug(u"Class {} mistaken for {}: {}".format(
                                  y[idx], predictions[idx], queries[idx].get_raw_query()))

        return self

    def _fit_super_learner(self, meta_clf, cv_iterator, verbose=False, scoring=ACCURACY_SCORING):
        """
        Trains a super-learner (stacked) classifier

        Args:
            meta_clf: the classifier class for the meta classifier
            cv_iterator:
            verbose:

        Returns:
            (TextModel): self
        """
        queries, classes = list(zip(*self._queries))  # unzip
        predictions = []
        m_classes = []
        m_queries = []
        # train and apply the base classifiers for each fold
        for train_idx, test_idx in StratifiedKFold(classes, n_folds=10, random_state=1):
            train_classes = [classes[i] for i in train_idx]
            train_queries = [queries[i] for i in train_idx]
            test_queries = [queries[i] for i in test_idx]
            test_classes = [classes[i] for i in test_idx]
            m_classes += test_classes
            m_queries += test_queries

            # TODO: parallelize
            for base_model in self._base_clfs.values():
                base_model.fit(train_queries, train_classes, verbose=False)

            predictions.extend(self.extract_meta_features(test_queries))

        m_X = self._meta_feat_vectorizer.fit_transform(predictions)
        m_y = self._class_encoder.transform(m_classes)
        self._queries = list(zip(m_queries, m_classes))

        # train the meta classifier
        if verbose is False:
            self._clf = self._fit_cv_grid(meta_clf, m_X, m_y, cv_iterator, scoring=scoring)
        else:
            self._clf = self._fit_cv_verbose(meta_clf, m_X, m_y, cv_iterator, scoring=scoring)

        # TODO: parallelize
        # retrain the base models on all the data.
        for base_model in self._base_clfs.values():
            base_model.fit(queries, classes, verbose=False)

        return self

    def _k_fold_iterator(self, y):
        k = self.config.cv['k']
        return StratifiedKFold(y, n_folds=k, shuffle=True)

    def _shuffle_iterator(self, y):
        k = self.config.cv['k']
        n = self.config.cv.get('n', k)
        return StratifiedShuffleSplit(y, n_iter=n, test_size=old_div(1.0, k))

    def _convert_settings(self, params_grid, y):
        """
        Convert the settings from the style given by the config
        to the style passed in to the actual classifier.

        Args:
            params_grid (dict): lists of classifier parameter values, keyed by parameter name

        Returns:
            (dict): revised params_grid
        """
        class_count = bincount(y)
        classes = self._class_encoder.classes_

        if 'class_weight' in params_grid:
            params_grid['class_weight'] = [{k if type(k) is int else
                                            self._class_encoder.transform(k): v
                                            for k, v in cw_dict.items()}
                                           for cw_dict in params_grid['class_weight']]
        elif 'class_bias' in params_grid:
            # interpolate between class_bias=0 => class_weight=None
            # and class_bias=1 => class_weight='balanced'
            params_grid['class_weight'] = []
            for class_bias in params_grid['class_bias']:
                # these weights are same as sklearn's class_weight='balanced'
                balanced_w = [old_div(len(y), (float(len(classes)) * c))
                              for c in class_count]
                balanced_tuples = list(zip(list(range(len(classes))), balanced_w))

                params_grid['class_weight'].append({c: (1 - class_bias) + class_bias * w
                                                   for c, w in balanced_tuples})
            del params_grid['class_bias']

        return params_grid

    def _fit(self, classifier_type, X, y):
        """Trains a classifier without cross-validation.

        Args:
            classifier_type (type): A multinomial classifier type. Must have
                methods fit() and predict(), like LogisticRegression in
                scikit-learn.
            X (numpy.matrix): The feature matrix for a dataset.
            y (numpy.array): The target output values.

        Returns:
            (object): An instance of classifier_type.
        """

        msg = 'Fitting {} text classifier without cross-validation'
        logger.info(msg.format(classifier_type.__name__))

        params_grid = self._convert_settings(self.config.params_grid, y)
        settings = next(self._iter_settings(params_grid))

        logger.info('Fitting text classifier with settings: {}'.format(settings))

        return classifier_type(**settings).fit(X, y)

    def _fit_cv_grid(self, classifier_type, X, y, cv_iterator, scoring=ACCURACY_SCORING):
        """Efficiently trains a classifier with stratified cross-validation.

        Args:
            classifier_type (type): A multinomial classifier type. Must have
                methods fit() and predict(), like LogisticRegression in
                scikit-learn.
            X (numpy.matrix): The feature matrix for a dataset.
            y (numpy.array): The target output values.
            cv_iterator (callable): A cross-validation split generator over y

        Returns:
            (object): An optimized instance of classifier_type.

        Grid searches over hyperparameter settings and finds one with the
        highest held-out accuracy, then uses those settings to train over
        the full dataset. Summary scores are shown without error analysis.
        """
        msg = 'Fitting {} text classifier by parallel {} cross-validation with settings: {}'
        logger.info(msg.format(classifier_type.__name__, self.config.cv['type'], self.config.cv))

        params_grid = self._convert_settings(self.config.params_grid, y)
        n_jobs = self.config.cv.get('n_jobs', -1)

        logger.info('Doing grid search over {}'.format(params_grid))
        grid_cv = GridSearchCV(estimator=classifier_type(), scoring=scoring, param_grid=params_grid,
                               cv=cv_iterator(y), verbose=1, n_jobs=n_jobs)
        model = grid_cv.fit(X, y)

        for candidate in model.grid_scores_:
            logger.info('Candidate parameters: {}'.format(candidate.parameters))
            std_err = (2 * numpy.std(candidate.cv_validation_scores) /
                       math.sqrt(len(candidate.cv_validation_scores)))
            if scoring == ACCURACY_SCORING:
                logger.info('Candidate average accuracy: {:.2%} ± '
                            '{:.2%}'.format(candidate.mean_validation_score, std_err))
            elif scoring == LIKELIHOOD_SCORING:
                logger.info('Candidate average log likelihood: {:.4} ± '
                            '{:.4}'.format(candidate.mean_validation_score, std_err))
        if scoring == ACCURACY_SCORING:
            logger.info('Best accuracy: {:.2%}, settings: {}'.format(model.best_score_,
                                                                     model.best_params_))
            self.cv_loss_ = 1 - model.best_score_
        elif scoring == LIKELIHOOD_SCORING:
            logger.info('Best log likelihood: {:.4}, settings: {}'.format(model.best_score_,
                                                                          model.best_params_))
            self.cv_loss_ = - model.best_score_
        return model.best_estimator_

    def _fit_cv_verbose(self, classifier_type, X, y, cv_iterator, scoring=ACCURACY_SCORING):
        """Trains a classifier with stratified cross-validation.

        Args:
            classifier_type (type): A multinomial classifier type. Must have
                methods fit() and predict(), like LogisticRegression in
                scikit-learn.
            X (numpy.matrix): The feature matrix for a dataset.
            y (numpy.array): The target output values.
            cv_iterator (callable): A cross-validation split generator over y

        Returns:
            (object): An optimized instance of classifier_type.

        Grid searches over hyperparameter settings and finds one with the
        highest held-out accuracy, then uses those settings to train over
        the full dataset. Error analysis is provided for each candidate model.
        """
        logger.info('Fitting {} text classifier by {} cross-validation with settings: {}'
                    .format(classifier_type.__name__,
                            self.config.cv['type'],
                            self.config.cv))

        best_accuracy = 0
        best_likelihood = _NEG_INF
        best_settings = None
        classes = self._class_encoder.classes_

        params_grid = self._convert_settings(self.config.params_grid, y)

        for settings in self._iter_settings(params_grid):

            logger.info('Fitting text classifier with settings: {}'.format(settings))

            clf = classifier_type(**settings)

            accuracies = []
            likelihoods = []
            errors = []
            padding = max([len(c) for c in classes])
            int_padding = 4

            for train_idx, test_idx in cv_iterator(y):
                clf.fit(X[train_idx], y[train_idx])
                predictions = clf.predict(X[test_idx])
                try:
                    probs = clf.predict_proba(X[test_idx])
                except AttributeError:
                    # in case clf doesn't supply probabilities, assume 1 vs 0
                    probs = numpy.array([[1 if n == predictions[i] else 10E-10
                                         for n in range(len(classes))]
                                        for i in range(len(predictions))])
                accuracies.append(accuracy_score(y[test_idx], predictions))
                likelihoods.append(-log_loss(y[test_idx], probs, normalize=True))

                # Produce categorization error analysis
                for idx, (i, j) in enumerate(zip(y[test_idx], predictions)):
                    if i != j:
                        real_idx = test_idx[idx]
                        errors.append((i, j))
                        logger.debug("Class {} mistaken for {}: {}"
                                     .format(i, j, self._queries[real_idx][0].text))

            accuracy = mean(accuracies)
            likelihood = mean(likelihoods)
            if ((scoring == ACCURACY_SCORING and accuracy > best_accuracy) or
                    scoring == LIKELIHOOD_SCORING and likelihood > best_likelihood):
                best_accuracy = accuracy
                best_likelihood = likelihood
                best_settings = settings

            # format categorization error table
            table = Counter(errors)
            row_totals = {}

            logger.info('Error analysis:'.ljust(padding + 5 + int_padding * 2) +
                        ''.rjust(int((int_padding + 2) * len(classes) / 2 - 5), '_') +
                        ' predicted ' +
                        ''.rjust(int((int_padding + 2) * len(classes) / 2 - 5), '_'))
            logger.info('  '.join(['gold'.rjust(padding), 'idx'.rjust(int_padding),
                                   'sum'.rjust(int_padding)] +
                        ["{0: >{1}}".format(c, int_padding) for c in range(len(classes))]))
            for i in range(len(classes)):
                row = [classes[i].rjust(padding), str(i).rjust(int_padding)]
                row_errors = [table.get((i, j), 0) for j in range(len(classes))]
                row_totals[classes[i]] = sum(row_errors)
                row += ["{0: >{1}}".format(sum(row_errors), int_padding)]
                row += ["{0: >{1}}".format('.' if e == 0 else e, int_padding)
                        for e in row_errors]

                logger.info('  '.join(row))

            std_err = old_div(std(accuracies), math.sqrt(len(accuracies)))
            logger.info('Candidate average CV accuracy: {:.2%} ± {:.2%}'
                        .format(accuracy, std_err * 2))
            loss_std_err = old_div(std(likelihoods), math.sqrt(len(likelihoods)))
            logger.info('Candidate average CV log likelihood: {:.2} ± {:.2}'
                        .format(likelihood, loss_std_err * 2))
            logger.debug('Errors per gold class: {}'.format(row_totals))

        logger.info('Best settings: {}'.format(best_settings))
        logger.info('Accuracy of best settings: {:.2%}'.format(best_accuracy))
        logger.info('Log likelihood of best settings: {:.2}'.format(best_likelihood))
        if scoring == ACCURACY_SCORING:
            self.cv_loss_ = 1 - best_accuracy
        elif scoring == LIKELIHOOD_SCORING:
            self.cv_loss_ = - best_likelihood

        return classifier_type(**best_settings).fit(X, y)

    def extract_meta_features(self, queries):
        """Generates the set of features for the super-learner from a list of queries

        Args:
            queries (list of Query): The queries

        Returns:
            (list of dict): meta features as feature_name => feature_value pairs
        """
        min_lprob = -10.0
        preds = []
        for idx, q in enumerate(queries):
            meta_feats = {}
            for base_model_conf, base_model in self._base_clfs.items():
                p_class, p_prob = base_model.predict_and_log_proba([q])
                meta_feats.update({"{}|class:{}".format(base_model_conf, p_class[0]): 1})
                meta_feats.update({"{}|class:{}|prob".format(base_model_conf, c): max(p, min_lprob)
                                   for c, p in p_prob[0].items()})
            preds.append(meta_feats)
        return preds

    def get_feature_matrix(self, queries, fit=False):
        """Transforms a list of Query objects into a feature matrix.

        Args:
            queries (list of Query): The queries.
            fit (bool): Whether to (re)fit vectorizer with queries

        Returns:
            (numpy.matrix): The feature matrix.
        """
        feats = [self.extract_features(q) for q in queries]
        if fit:
            X = self._feat_vectorizer.fit_transform(feats)
        else:
            X = self._feat_vectorizer.transform(feats)
        return X

    def predict(self, queries):
        """Predicts class labels for a set of queries.

        Args:
            queries (list of Query): The queries

        Returns:
            (list of str): The predicted labels for each query.
        """

        if self._base_clfs:
            preds = self.extract_meta_features(queries)
            X = self._meta_feat_vectorizer.transform(preds)
        else:
            X = self.get_feature_matrix(queries)
        predictions = self._clf.predict(X)
        return self._class_encoder.inverse_transform(predictions)

    # TODO (julius): Log probability is not applicable to all classifier types.
    def predict_log_proba(self, queries):
        """Returns predicted log probability values for a set of queries.

        Args:
            queries (list of Query): The queries.

        Returns:
            (list of list): The predicted labels for each query.
        """
        return self.predict_and_log_proba(queries)[1]

    def predict_and_log_proba(self, queries, verbose=False, gold=None):
        """For a set of queries, return both predictions and log probabilities

        This duplicates functionality from predict() and predict_log_proba(),
        but is included so clients can do both without running the classifier
        more than once.

        Args:
            queries (list of Query): The queries.
            verbose (bool): calculate and print detailed analysis
            gold (list of int): The gold class index for each query. For analysis.

        Returns:
            ((list of str), (list of list)) The first element is the same as in
                predict(), the second element is the same as in
                predict_log_proba().
        """

        # Prediction is somewhat different if we're using ensemble methods.
        if self._base_clfs:
            predictions = self.extract_meta_features(queries)
            X = self._meta_feat_vectorizer.transform(predictions)
        else:
            X = self.get_feature_matrix(queries)
        predictions = []
        log_proba = []
        for i, row in enumerate(self._clf.predict_log_proba(X)):
            class_index = row.argmax()
            predictions.append(self._class_encoder.inverse_transform([class_index])[0])
            log_proba.append(dict(
                (self._class_encoder.inverse_transform([j])[0], row[j])
                for j in range(len(row))))

            if verbose:
                if gold is not None:
                    gold = gold[i]
                self._print_query_inspection(queries[i], class_index, gold)

        # JSON can't reliably encode infinity, so replace it with large number
        for row in log_proba:
            for label in row:
                if row[label] == -Infinity:
                    row[label] = _NEG_INF
        return predictions, log_proba

    def _print_query_inspection(self, query, pred_class, gold_class):
        pred_label = self._class_encoder.inverse_transform([pred_class])[0]

        if self._base_clfs:
            # super-learner
            features = self.extract_meta_features([query])[0]
            vectorizer = self._meta_feat_vectorizer
        else:
            features = self.extract_features(query)
            vectorizer = self._feat_vectorizer

        print("Predicted: " + pred_label)
        columns = 'FEATURE                       \t   VALUE\t  PRED_W\t  PRED_P'

        if gold_class is not None:
            gold_label = self._class_encoder.inverse_transform([gold_class])[0]
            print("Gold:      " + gold_label)
            columns += '\t  GOLD_W\t  GOLD_P\t    DIFF'
        print()
        print(columns)
        print()

        # Get all active features sorted alphabetically by name
        features = sorted(list(features.items()), key=operator.itemgetter(0))
        name_format = '{{0:{}}}'.format(max([len(f[0]) for f in features] + [20]))
        for feature in features:
            feat_name = feature[0]
            feat_value = feature[1]

            # Features we haven't seen before won't be in our vectorizer
            # e.g., an exact match feature for a query we've never seen before
            if feat_name not in vectorizer.vocabulary_:
                continue

            if len(self._class_encoder.classes_) == 2 and pred_class == 1:
                weight = 0
            else:
                weight = self._clf.coef_[pred_class, vectorizer.vocabulary_[feat_name]]
            product = feat_value * weight

            if gold_class is None:
                print(name_format + '\t{1:8.3f}\t{2:8.3f}\t{3:8.3f}'
                      .format(feat_name, feat_value, weight, product))
            else:
                if len(self._class_encoder.classes_) == 2 and gold_class == 1:
                    gold_w = 0
                else:
                    gold_w = self._clf.coef_[gold_class, vectorizer.vocabulary_[feat_name]]
                gold_p = feat_value * gold_w
                diff = gold_p - product

                print(name_format + '\t{1:8.3f}\t{2:8.3f}\t{3:8.3f}\t{4:8.3f}\t{5:8.3f}\t{6:+8.3f}'
                      .format(feat_name, feat_value, weight, product, gold_w, gold_p, diff))
        print()

    def requires_resource(self, resource):
        for f in self.config.features:
            if ('requirements' in FEATURE_NAME_MAP[f].__dict__ and
                    resource in FEATURE_NAME_MAP[f].requirements):
                return True
        return False


def requires(resource):
    """
    Decorator to enforce the resource dependencies of the active feature extractors

    Args:
        resource (str): the key of a classifier resource which must be initialized before
            the given feature extractor is used

    Returns:
        (func): the feature extractor
    """
    def add_resource(func):
        req = func.__dict__.get('requirements', [])
        func.requirements = req + [resource]
        return func

    return add_resource


def mask_numerics(token):
    if token.isdigit():
        return '#NUM'
    else:
        return re.sub(r'\d', '8', token)


@requires(WORD_FREQ_RSC)
def extract_ngrams(lengths=(1,)):
    """
    Extract ngrams of some specified lengths.

    Args:
        lengths (list of int): The ngram length.

    Returns:
        (function) An feature extraction function that takes a query and
            returns ngrams of the specified lengths.
    """
    def _extractor(query, resources):
        tokens = query.split()
        ngram_counter = Counter()
        for length in lengths:
            for i in range(len(tokens) - length + 1):
                ngram = []
                for token in tokens[i:i + length]:
                    # We never want to differentiate between number tokens.
                    # We may need to convert number words too, like "eighty".
                    tok = mask_numerics(token)
                    if tok not in resources[WORD_FREQ_RSC]:
                        tok = 'OOV'
                    ngram.append(tok)
                ngram_counter.update(['ngram:' + '|'.join(ngram)])
        return ngram_counter

    return _extractor


@requires(WORD_FREQ_RSC)
def extract_edge_ngrams(lengths=(1,)):
    """
    Extract ngrams of some specified lengths.

    Args:
        lengths (list of int): The ngram length.

    Returns:
        (function) An feature extraction function that takes a query and
            returns ngrams of the specified lengths at start and end of query.
    """
    def _extractor(query, resources):
        tokens = query.split()
        feats = {}
        for length in lengths:
            if length < len(tokens):
                left_tokens = [mask_numerics(tok) for tok in tokens[:length]]
                left_tokens = [tok if resources[WORD_FREQ_RSC].get(tok, 0) > 1 else 'OOV'
                               for tok in left_tokens]
                right_tokens = [mask_numerics(tok) for tok in tokens[-length:]]
                right_tokens = [tok if resources[WORD_FREQ_RSC].get(tok, 0) > 1 else 'OOV'
                                for tok in right_tokens]
                feats.update({'left-edge|{}:{}'.format(length, '|'.join(left_tokens)): 1})
                feats.update({'right-edge|{}:{}'.format(length, '|'.join(right_tokens)): 1})

        return feats

    return _extractor


@requires(WORD_FREQ_RSC)
def extract_freq(bins=5):
    """
    Extract frequency bin features.

    Args:
        bins (int): The number of frequency bins (besides OOV)

    Returns:
        (function): A feature extraction function that returns the log of the
            count of query tokens within each frequency bin.

    """
    def _extractor(query, resources):
        tokens = query.split()
        freq_dict = resources[WORD_FREQ_RSC]
        max_freq = freq_dict.most_common(1)[0][1]
        freq_features = defaultdict(int)
        for tok in tokens:
            tok = mask_numerics(tok)
            freq = freq_dict.get(tok, 0)
            if freq < 2:
                freq_features['freq|U'] += 1
            else:
                # Bin the frequency with break points at
                # half max, a quarter max, an eighth max, etc.
                freq_bin = int(math.log(max_freq, 2) - math.log(freq, 2))
                if freq_bin < bins:
                    freq_features['freq|{}'.format(freq_bin)] += 1
                else:
                    freq_features['freq|{}'.format(freq_bin)] += 1

        q_len = float(len(tokens))
        for k in freq_features:
            # sublinear
            freq_features[k] = math.log(freq_features[k] + 1, 2)
            # ratio
            freq_features[k] /= q_len
        return freq_features

    return _extractor


@requires(GAZETTEER_RSC)
@requires(WORD_FREQ_RSC)
def extract_gaz_freq():
    """
    Extract frequency bin features for each gazetteer

    Returns:
        (function): A feature extraction function that returns the log of the
            count of query tokens within each gazetteer's frequency bins.
    """
    def _extractor(query, resources):
        tokens = query.split()
        freq_features = defaultdict(int)

        for tok in tokens:
            query_freq = 'OOV' if resources[WORD_FREQ_RSC].get(tok) is None else 'IV'
            for gaz_name, gaz in resources[GAZETTEER_RSC].items():
                freq = len(gaz['index'].get(tok, []))
                if freq > 0:
                    freq_bin = int(old_div(math.log(freq, 2), 2))
                    freq_features['{}|freq|{}'.format(gaz_name, freq_bin)] += 1
                    freq_features['{}&{}|freq|{}'.format(query_freq, gaz_name, freq_bin)] += 1

        q_len = float(len(tokens))
        for k in freq_features:
            # sublinear
            freq_features[k] = math.log(freq_features[k] + 1, 2)
            # ratio
            freq_features[k] /= q_len
        return freq_features

    return _extractor


@requires(GAZETTEER_RSC)
def extract_in_gaz_feature(scaling=1):

    def _extractor(query, resources):
        in_gaz_features = defaultdict(float)

        tokens = query.split()
        ngrams = []
        for i in range(1, (len(tokens) + 1)):
            ngrams.extend(find_ngrams(tokens, i))
        for ngram in ngrams:
            for gaz_name, gaz in resources[GAZETTEER_RSC].items():
                if ngram in gaz['pop_dict']:
                    popularity = gaz['pop_dict'].get(ngram, 0.0)
                    ratio = (old_div(len(ngram), float(len(query)))) * scaling
                    ratio_pop = ratio * popularity
                    in_gaz_features[gaz_name + '_ratio_pop'] += ratio_pop
                    in_gaz_features[gaz_name + '_ratio'] += ratio
                    in_gaz_features[gaz_name + '_pop'] += popularity
                    in_gaz_features[gaz_name + '_exists'] = 1

        return in_gaz_features

    return _extractor


def extract_length():
    """
    Extract length measures (tokens and chars; linear and log) on whole query.

    Returns:
        (function) A feature extraction function that takes a query and
            returns number of tokens and characters on linear and log scales
    """
    # pylint: disable=locally-disabled,unused-argument
    def _extractor(query, resources):
        tokens = len(query.split())
        chars = len(query)
        return {'tokens': tokens,
                'chars': chars,
                'tokens_log': math.log(tokens + 1),
                'chars_log': math.log(chars + 1)}

    return _extractor


@requires(QUERY_FREQ_RSC)
def extract_query_string(scaling=1000):
    """
    Extract whole query string as a feature.

    Returns:
        (function) A feature extraction function that takes a query and
            returns the whole query string for exact matching

    """

    def _extractor(query, resources):
        query_key = u'<{}>'.format(query)
        if query_key not in resources[QUERY_FREQ_RSC]:
            query_key = '<OOV>'
        return {'exact={}'.format(query_key): scaling}

    return _extractor


# Generate all n-gram combinations from a list of strings
def find_ngrams(input_list, n):
    result = []
    for ngram in zip(*[input_list[i:] for i in range(n)]):
        result.append(" ".join(ngram))
    return result


FEATURE_NAME_MAP = {
    'bag-of-words': extract_ngrams,
    'edge-ngrams': extract_edge_ngrams,
    'freq': extract_freq,
    'in-gaz': extract_in_gaz_feature,
    'gaz-freq': extract_gaz_freq,
    'length': extract_length,
    'exact': extract_query_string
}
