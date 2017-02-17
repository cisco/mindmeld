# coding=utf-8
"""
This module contains all code required to perform multinomial classification
of text.
"""
# TODO (julius): Add validation to the model configuration files.

import copy
import itertools
import logging
import math
import re
from numpy import random, bincount, mean, std, Infinity
from collections import Counter
from collections import defaultdict
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.cross_validation import StratifiedKFold, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score as AccuracyScore

_NEG_INF = -1e10


class TextClassifier:
    """A machine learning classifier for text.

    This class manages feature extraction, training, cross-validation, and
    prediction. The design goal is that after providing initial settings like
    hyperparameters, grid-searchable hyperparameters, feature extractors, and
    cross-validation settings, TextClassifier manages all of the details
    involved in training and prediction such that the input to training or
    prediction is Query objects, and the output is class names, and no data
    manipulation is needed from the client.

    Attributes:
        classifier_type (str): The name of the classifier type. Currently
            recognized values are "logreg","dtree", and "svm".
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

    def __init__(self, classifier_type, hyperparams, grid_search_hyperparams,
                 feat_specs, cross_validation_settings):
        self.classifier_type = classifier_type
        self.hyperparams = hyperparams
        self.grid_search_hyperparams = grid_search_hyperparams
        self.feat_specs = feat_specs
        self.cross_validation_settings = cross_validation_settings

        self._is_fit = False
        self._feat_vectorizer = DictVectorizer(sparse=True)
        self._class_encoder = LabelEncoder()
        self._clf = None
        self._resources = {}
        self._queries = {}

    @classmethod
    def from_config(cls, config, model_name):
        """Initializes a TextClassifier instance from config file data.

        Args:
            config (dict): The Python representation of a parsed config file.
            model_name (str): One of the keys in config['models'].

        Returns:
            (TextClassifier): A TextClassifier instance initialized with the
                settings from the config entry given by model_name.
        """
        model_config_entry = config['models'].get(model_name)
        if not model_config_entry:
            error_msg = "Model config does not contain a model named '{}'"
            raise ValueError(error_msg.format(model_name))

        return cls(
            model_config_entry.get('model-type',
                                   model_config_entry.get('classifier-type')),
            model_config_entry.get('model-parameters', {}),
            model_config_entry.get('model-parameter-choices', {}),
            model_config_entry['features'],
            model_config_entry['cross-validation-settings'])

    def __getstate__(self):
        """Returns the information needed pickle an instance of this class.

        By default, pickling removes attributes with names starting with
        underscores. This overrides that behavior.
        """
        attributes = self.__dict__.copy()
        attributes['_resources'] = {'freq': self._resources['freq']}
        return attributes

    def load_resources(self, all_gazeteers):
        self._resources['gaz'] = all_gazeteers

    def extract_features(self, query):
        """Gets all features from a query.

        Args:
            query (Query): A query instance.

        Returns:
            (dict of str: number): A dict of feature names to their values.
        """
        feat_set = {}
        marked_down_query = query.get_normalized_marked_down_query()
        for name, kwargs in self.feat_specs.items():
            feat_extractor = FEATURE_NAME_MAP[name](**kwargs)
            feat_set.update(feat_extractor(marked_down_query, self._resources))
        return feat_set

    def _iter_settings(self):
        """Iterates through all classifier settings.

        Yields:
            (dict): A kwargs dict to be passed to the classifier object. Each
                item yielded is a unique combination of self.hyperparams with
                a choice of settings from self.grid_search_hyperparams.
        """
        if self.grid_search_hyperparams:
            all_settings = copy.deepcopy(self.hyperparams)
            gsh_keys, gsh_values = zip(*self.grid_search_hyperparams.items())
            for settings in itertools.product(*gsh_values):
                all_settings.update(dict(zip(gsh_keys, settings)))
                yield copy.deepcopy(all_settings)
        else:
            yield self.hyperparams

    def _compile_freq_dict(self, queries):
        """Compiles frequency dictionary of query tokens

        """
        tokens = [mask_numerics(tok) for q in queries
                  for tok in q.get_normalized_tokens()]
        freq_dict = Counter(tokens)

        query_dict = Counter([u'<{}>'.format(q.get_normalized_query()) for q in queries])
        for q in query_dict:
            if query_dict[q] < 2:
                query_dict[q] = 0
        query_dict += Counter()
        freq_dict += query_dict

        self._resources['freq'] = freq_dict

    def fit(self, queries, classes):
        """Trains this TextClassifier.

        This method inspects instance attributes to determine the classifier
        object and cross-validation strategy, and then fits the model to the
        training examples passed in.

        Args:
            queries (list of Query): A list of queries.
            classes (list of str): A parallel list to queries. The gold labels
                for each query.

        Returns:
            (TextClassifier): Returns self to match classifier scikit-learn
                interfaces.
        """

        self._compile_freq_dict(queries)

        # Need to shuffle once to prevent order effects
        indices = list(range(len(classes)))
        random.shuffle(indices)
        classes = [classes[i] for i in indices]
        queries = [queries[i] for i in indices]
        self._queries = zip(queries, classes)

        classes_set = set(classes)
        if len(set(classes_set)) <= 1: return None

        X = self.get_feature_matrix(queries, fit=True)
        y = self._class_encoder.fit_transform(classes)

        if self.classifier_type == 'logreg':
            clf_cls = LogisticRegression
        elif self.classifier_type == 'dtree':
            clf_cls = DecisionTreeClassifier
        elif self.classifier_type == 'rforest':
            clf_cls = RandomForestClassifier
        elif self.classifier_type == 'svm':
            clf_cls = SVC
        else:
            raise ValueError('Classifier type "{}" not recognized'
                             .format(self.classifier_type))

        if self.cross_validation_settings['type'] == 'k-fold':
            cv_iterator = self._k_fold_iterator
        elif self.cross_validation_settings['type'] == 'shuffle':
            cv_iterator = self._shuffle_iterator
        else:
            raise ValueError('CV iterator type "{}" not recognized'
                             .format(self.cross_validation_settings['type']))

        self._clf = self._fit_cv(clf_cls, X, y, cv_iterator)

        predictions = self._clf.predict(X)
        acc = AccuracyScore(y, predictions)
        logging.info("Final accuracy on training data: {:.1%}".format(acc))
        for idx in range(len(predictions)):
            if predictions[idx] != y[idx]:
                logging.debug("Class {} mistaken for {}: {}"
                              .format(y[idx], predictions[idx], queries[idx].get_raw_query()))

        return self

    def _k_fold_iterator(self, y):
        k = self.cross_validation_settings['k']
        return StratifiedKFold(y, n_folds=k)

    def _shuffle_iterator(self, y):
        k = self.cross_validation_settings['k']
        n = self.cross_validation_settings.get('n', k)
        return StratifiedShuffleSplit(y, n_iter=n, test_size=1.0/k)

    def _fit_cv(self, classifier_type, X, y, cv_iterator):
        """Trains a classifier with stratified cross-validation.

        Grid searches over hyperparameter settings and finds one with the
        highest held-out accuracy, then uses those settings to train over
        the full dataset.

        Args:
            classifier_type (type): A multinomial classifier type. Must have
                methods fit() and predict(), like LogisticRegression in
                scikit-learn.
            X (numpy.matrix): The feature matrix for a dataset.
            y (numpy.array): The target output values.
            cv_iterator (callable): A cross-validation split generator over y

        Returns:
            (object): An instance of classifier_type.
        """
        logging.info(
            'Fitting text classifier by {} cross-validation with settings: {}'
            .format(self.cross_validation_settings['type'],
                    self.cross_validation_settings))

        best_accuracy = -1.0
        best_settings = None
        classes = self._class_encoder.classes_
        class_count = bincount(y)

        for settings in self._iter_settings():

            logging.info('Fitting text classifier with settings: {}'.format(
                settings))
            if 'class_weight' in settings:
                # translate string keys to class codes
                cw_dict = {self._class_encoder.transform(k): v
                           for k, v in settings['class_weight'].items()}
                settings['class_weight'] = cw_dict
            elif 'class_bias' in settings:
                # interpolate between class_bias=0 => class_weight=None
                # and class_bias=1 => class_weight='balanced'
                class_bias = settings['class_bias']

                # these weights are same as sklearn's class_weight='balanced'
                balanced_w = [len(y) / (float(len(classes)) * c)
                              for c in class_count]
                balanced_tuples = zip(list(range(len(classes))), balanced_w)

                settings['class_weight'] = {c: (1 - class_bias) + class_bias * w
                                            for c, w in balanced_tuples}
                del settings['class_bias']


            clf = classifier_type(**settings)
            scores = []
            errors = []
            padding = max([len(c) for c in classes])
            int_padding = 4

            for train_idx, test_idx in cv_iterator(y):
                clf.fit(X[train_idx], y[train_idx])
                predictions = clf.predict(X[test_idx])
                scores.append(AccuracyScore(y[test_idx], predictions))

                # Produce categorization error analysis
                for idx, (i, j) in enumerate(zip(y[test_idx], predictions)):
                    if i != j:
                        real_idx = test_idx[idx]
                        errors.append((i, j))
                        logging.debug(u"Class {} mistaken for {}: {}".format(
                            i, j, self._queries[real_idx][0].get_raw_query()))
            # format categorization error table

            table = Counter(errors)
            row_totals = {}
            logging.info('Error analysis:'.ljust(padding + 5 + int_padding * 2) +
                          ''.rjust((int_padding + 2) * len(classes) / 2 - 5, '_') +
                          ' predicted ' +
                          ''.rjust((int_padding + 2) * len(classes) / 2 - 5, '_'))
            logging.info('  '.join(['gold'.rjust(padding), 'idx'.rjust(int_padding),
                                    'sum'.rjust(int_padding)] +
                         ["{0: >{1}}".format(c, int_padding) for c in range(len(classes))]))
            for i in range(len(classes)):
                row = [classes[i].rjust(padding), str(i).rjust(int_padding)]
                row_errors = [table.get((i, j), 0) for j in range(len(classes))]
                row_totals[classes[i]] = sum(row_errors)
                row += ["{0: >{1}}".format(sum(row_errors), int_padding)]
                row += ["{0: >{1}}".format('.' if e == 0 else e, int_padding)
                        for e in row_errors]

                logging.info('  '.join(row))

            accuracy_score = mean(scores)
            std_err = std(scores) / math.sqrt(len(scores))
            if accuracy_score > best_accuracy:
                best_accuracy = accuracy_score
                best_settings = settings
            logging.info('Average CV accuracy: {:.2%} ± {:.2%}'.format(
                accuracy_score, std_err*2))
            logging.debug('Errors per gold class: {}'.format(row_totals))
        logging.info('Best accuracy: {:.2%}, settings: {}'.format(
            best_accuracy, best_settings))
        return classifier_type(**best_settings).fit(X, y)

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
            queries (list of Query): The queries.

        Returns:
            (list of str): The predicted labels for each query.
        """
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
            gold (list of int): The gold class index for each query. For analysis.

        Returns:
            ((list of str), (list of list)) The first element is the same as in
                predict(), the second element is the same as in
                predict_log_proba().
        """
        X = self.get_feature_matrix(queries)
        predictions = []
        log_proba = []
        for i, row in enumerate(self._clf.predict_log_proba(X)):
            class_index = row.argmax()
            predictions.append(
                self._class_encoder.inverse_transform([class_index])[0])
            log_proba.append(dict(
                (self._class_encoder.inverse_transform([j])[0], row[j])
                for j in range(len(row))))

            if verbose:
                pred_class = predictions[-1]
                gold = [gold] if isinstance(gold, int) else gold
                print("Predicted: " + pred_class)
                columns = 'FEATURE                       \t   VALUE\t  PRED_W\t  PRED_P'

                if gold is not None:
                    gold_class = self._class_encoder.inverse_transform([gold[i]])
                    if not isinstance(gold_class, str):
                        gold_class = gold_class[0]
                    print("Gold:      " + gold_class)
                    columns += '\t  GOLD_W\t  GOLD_P\t    DIFF'
                print
                print(columns)
                print
                import operator
                # Get all active features sorted alphabetically by name
                features = sorted(
                    self.extract_features(queries[i]).items(),
                    key=operator.itemgetter(0)
                )
                for feature in features:
                    feat_name = feature[0]
                    feat_value = feature[1]

                    # Features we haven't seen before won't be in our vectorizer
                    # e.g., an exact match feature for a query we've never seen before
                    if feat_name not in self._feat_vectorizer.vocabulary_:
                        continue

                    if len(self._class_encoder.classes_) == 2 and class_index == 1:
                        weight = 0
                    else:
                        weight = self._clf.coef_[class_index,
                            self._feat_vectorizer.vocabulary_[feat_name]]
                    product = feat_value * weight

                    if gold is None:
                        print('{0:30}\t{1:8.3f}\t{2:8.3f}\t{3:8.3f}'.format(
                            feat_name, feat_value, weight, product))
                    else:
                        if len(self._class_encoder.classes_) == 2 and gold[i] == 1:
                            gold_w = 0
                        else:
                            gold_w = self._clf.coef_[gold[i],
                                self._feat_vectorizer.vocabulary_[feat_name]]
                        gold_p = feat_value * gold_w
                        diff = gold_p - product

                        print('{0:30}\t{1:8.3f}\t{2:8.3f}\t{3:8.3f}\t{4:8.3f}\t{5:8.3f}\t{6:+8.3f}'.format(
                            feat_name, feat_value, weight, product, gold_w, gold_p, diff))
                print

        # JSON can't reliably encode infinity, so replace it with large number
        for row in log_proba:
            for label in row:
                if row[label] == -Infinity:
                    row[label] = _NEG_INF
        return predictions, log_proba


def mask_numerics(token):
    if token.isdigit():
        return '#NUM'
    else:
        return re.sub('\d', '8', token)


def extract_ngrams(lengths=(1,)):
    """
    Extract ngrams of some specified lengths.

    Args:
        lengths (list of int): The ngram length.

    Returns:
        (function) An feature extraction function that takes a query and
            returns ngrams of the specified lengths.
    """
    def extractor(query, resources):
        tokens = query.split()
        ngram_counter = Counter()
        for length in lengths:
            for i in range(len(tokens) - length + 1):
                ngram = []
                for token in tokens[i:i+length]:
                    # We never want to differentiate between number tokens.
                    # We may need to convert number words too, like "eighty".
                    tok = mask_numerics(token)
                    if tok not in resources['freq']:
                        tok = 'OOV'
                    ngram.append(tok)
                ngram_counter.update(['ngram:' + '|'.join(ngram)])
        return ngram_counter

    return extractor


def extract_edge_ngrams(lengths=(1,)):
    """
    Extract ngrams of some specified lengths.

    Args:
        lengths (list of int): The ngram length.

    Returns:
        (function) An feature extraction function that takes a query and
            returns ngrams of the specified lengths at start and end of query.
    """
    def extractor(query, resources):
        tokens = query.split()
        feats = {}
        for length in lengths:
            if length < len(tokens):
                left_tokens = [mask_numerics(tok) for tok in tokens[:length]]
                left_tokens = [tok if resources['freq'].get(tok, 0) > 1 else 'OOV'
                               for tok in left_tokens]
                right_tokens = [mask_numerics(tok) for tok in tokens[-length:]]
                right_tokens = [tok if resources['freq'].get(tok, 0) > 1 else 'OOV'
                                for tok in right_tokens]
                feats.update({'left-edge|{}:{}'.format(length, '|'.join(left_tokens)): 1})
                feats.update({'right-edge|{}:{}'.format(length, '|'.join(right_tokens)): 1})

        return feats

    return extractor


def extract_freq(bins=5):
    """
    Extract frequency bin features.

    Args:
        bins (int): The number of frequency bins (besides OOV)

    Returns:
        (function): A feature extraction function that returns the log of the
            count of query tokens within each frequency bin.

    """
    def extractor(query, resources, **kwargs):
        tokens = query.split()
        freq_dict = resources['freq']
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
                b = int(math.log(max_freq, 2) - math.log(freq, 2))
                if b < bins:
                    freq_features['freq|{}'.format(b)] += 1
                else:
                    freq_features['freq|{}'.format(bins)] += 1

        q_len = float(len(tokens))
        for k in freq_features:
            # sublinear
            freq_features[k] = math.log(freq_features[k] + 1, 2)
            # ratio
            freq_features[k] /= q_len
        return freq_features

    return extractor


def extract_gaz_freq():
    """
    Extract frequency bin features for each gazetteer

    Args:
        bins (int): The number of frequency bins (besides OOV)

    Returns:
        (function): A feature extraction function that returns the log of the
            count of query tokens within each gazetteer's frequency bins.
    """
    def extractor(query, resources):
        tokens = query.split()
        freq_features = defaultdict(int)
        for tok in tokens:
            query_freq = 'OOV' if resources['freq'].get(tok) is None else 'IV'
            for domain, gazes in resources['gaz'].items():
                for gaz_name, gaz in gazes.items():
                    freq = len(gaz['index'].get(tok, []))
                    if freq > 0:
                        b = int(math.log(freq, 2) / 2)
                        freq_features['{}|freq|{}'.format(gaz_name, b)] += 1
                        freq_features['{}&{}|freq|{}'.format(query_freq, gaz_name, b)] += 1

        q_len = float(len(tokens))
        for k in freq_features:
            # sublinear
            freq_features[k] = math.log(freq_features[k] + 1, 2)
            # ratio
            freq_features[k] /= q_len
        return freq_features

    return extractor


def extract_in_gaz_feature(scaling=1):

    def extractor(query, resources):
        in_gaz_features = defaultdict(float)
        tokens = query.split()
        ngrams = []
        for i in range(1, (len(tokens) + 1)):
            ngrams.extend(find_ngrams(tokens, i))
        for ngram in ngrams:
            for domain, gazes in resources['gaz'].items():
                for gaz_name, gaz in gazes.items():
                    if ngram in gaz['edict']:
                        popularity = gaz['edict'].get(ngram, 0.0)
                        in_gaz_features[domain + '_' + gaz_name + '_ratio_pop'] += (len(ngram)/float(len(query))) * scaling * popularity
                        in_gaz_features[domain + '_' + gaz_name + '_ratio'] += (len(ngram)/float(len(query))) * scaling
                        in_gaz_features[domain + '_' + gaz_name + '_pop'] += popularity
                        in_gaz_features[domain + '_' + gaz_name + '_exists'] = 1

        return in_gaz_features

    return extractor


def extract_length():
    """
    Extract length measures (tokens and chars; linear and log) on whole query.

    Returns:
        (function) A feature extraction function that takes a query and
            returns number of tokens and characters on linear and log scales
    """

    def extractor(query, resources):
        tokens = len(query.split())
        chars = len(query)
        return {'tokens': tokens,
                'chars': chars,
                'tokens_log': math.log(tokens + 1),
                'chars_log': math.log(chars + 1)}

    return extractor


def extract_query_string(scaling=1000):
    """
    Extract whole query string as a feature.

    Returns:
        (function) A feature extraction function that takes a query and
            returns the whole query string for exact matching

    """

    def extractor(query, resources):
        query_key = u'<{}>'.format(query)
        if query_key not in resources['freq']:
            query_key = '<OOV>'
        return {'exact={}'.format(query_key): scaling}

    return extractor

# Generate all n-gram combinations from a list of strings
def find_ngrams(input_list, n):
  result = []
  ngrams = zip(*[input_list[i:] for i in range(n)])
  for ngram in ngrams:
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
