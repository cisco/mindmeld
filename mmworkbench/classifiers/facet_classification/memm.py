"""

"""

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from scipy.sparse import vstack, hstack
import numpy
import logging

from . import util
from . import sequence_features
from .base_model import FacetClassifier

DEFAULT_FEATURES = {
    'bag-of-words': {
        'ngram_lengths_to_start_positions': {
            1: [-2, -1, 0, 1, 2],
            2: [-2, -1, 0, 1]
        }
    },
    'in-gaz-span': {},
    'num-candidates': {
        'start_positions': [-1, 0, 1]
    }
}

class MemmFacetClassifier(FacetClassifier):
    """A maximum-entropy Markov model for facet prediction.

    This class implements a conditional sequence model that predicts tags that
    are a joint representation of numeric and non-numeric facets.

    Attributes:
        feat_specs (dict): A mapping from feature extractor names, as given in
            FEATURE_NAME_MAP, to a kwargs dict, which will be passed into the
            associated feature extractor function.
    """

    def __init__(self, model_parameter_choices=None,
                 cross_validation_settings=None,
                 classifier_settings=None,
                 features=None):

        features = features or DEFAULT_FEATURES

        super(MemmFacetClassifier, self).__init__(model_parameter_choices,
                                                  cross_validation_settings,
                                                  classifier_settings,
                                                  features)

        self._tag_scheme = self.classifier_settings.get('tag-scheme', 'IOB').upper()
        self._feat_selector = self.get_feature_selector()
        self._feat_scaler = self.get_feature_scaler()

        self._clf = None
        self._class_encoder = LabelEncoder()
        self._feat_vectorizer = DictVectorizer()

        self._no_facets = False

    def __getstate__(self):
        """Returns the information needed pickle an instance of this class.

        By default, pickling removes attributes with names starting with
        underscores. This overrides that behavior. For the _resources field,
        we save the resources that are memory intensive
        """
        attributes = self.__dict__.copy()
        saved_resources = ['num_types']
        for key in attributes['_resources'].keys():
            if key not in saved_resources: del attributes['_resources'][key]
        return attributes

    def _extract_query_features(self, query):
        """Extracts feature dicts for each token in a query."""
        feat_seq = [{} for _ in query.get_normalized_tokens()]
        for name, kwargs in self.feat_specs.items():
            feat_extractor = sequence_features.get_feature_template(name)(
                **kwargs)
            util.update_features_sequence(
                feat_seq, feat_extractor(query, self._resources))
        return feat_seq

    def _clf_fit_cv(self, X, y, query_groups, cv_iterator):
        best_score = None
        best_settings = None
        metric = self.cross_validation_settings.get("metric", "accuracy")

        for settings in self._iter_settings():
            logging.info("Fitting facet classifier with settings: {}".format(
                settings))

            self._model_parameters = dict(settings)
            self._clf = LogisticRegression(**self._model_parameters)

            scores = cross_validation.cross_val_score(self._clf, X, y,
                                                      metric,
                                                      cv_iterator(query_groups))

            mean_score = numpy.mean(scores)
            std_err = numpy.std(scores) / numpy.sqrt(len(scores))
            if mean_score > best_score:
                best_score = mean_score
                best_settings = settings
            logging.info('Average tagging CV {} score: {:.2%} +/- {:.2%}'
                         .format(metric, mean_score, std_err*2))

        logging.info('Best score: {:.2%}, settings: {}'.format(
            best_score, best_settings))

        self._model_parameters = best_settings
        self._clf = LogisticRegression(**self._model_parameters)
        self._clf.fit(X, y)

    def fit(self, labeled_queries, domain, intent, facet_names=None, verbose=False):
        # all_features and all_tags are parallel lists of feature dictionaries
        # and their associated gold tags, respectively. They are concatenations
        # of the features and tags across all the input queries.
        all_features = []
        all_tags = []
        all_query_groups = []

        numeric_types = util.get_numeric_types(self._resources['gazetteers'])
        self.load_resources(num_types=numeric_types)

        for idx, query in enumerate(labeled_queries):
            features = self._extract_query_features(query)
            tags = util.get_tags_from_facets(query, self._tag_scheme)
            query_groups = [idx for _ in tags]

            if len(features) > 0:
                # Set the special feature for the identity of the previous tag.
                features[0]['prev-tag'] = util.START_TAG
                for i, tag in enumerate(tags[:-1]):
                    features[i+1]['prev-tag'] = str(tag)

            all_features.extend(features)
            all_tags.extend(tags)
            all_query_groups.extend(query_groups)

        if len(set(all_tags)) == 1:
            self._no_facets = True
        else:
            # Fit the model
            X = self._feat_vectorizer.fit_transform(all_features)
            y = self._class_encoder.fit_transform(all_tags)
            if self._feat_scaler is not None:
                X = self._feat_scaler.fit_transform(X)
            if self._feat_selector is not None:
                X = self._feat_selector.fit_transform(X, y)
            cv_iterator = self.get_cv_iterator()
            choices = max([len(v)
                           for v in self.model_parameter_choices.values()])
            if cv_iterator is None or choices <= 1:
                if choices > 1:
                    logging.warning("Found multiple model parameter choices "
                                    "but no cross-validation type")
                self._model_parameters = dict(self._iter_settings().next())
                self._clf = LogisticRegression(**self._model_parameters)
                self._clf.fit(X, y)
            else:
                self._clf_fit_cv(X, y, all_query_groups, cv_iterator)

    def predict(self, query, domain, intent, facet_names=None, verbose=False):
        facets, num_facets, data = self._predict(query, verbose)
        if verbose:
            self._print_predict_data(data)

        return facets, num_facets, data

    def _predict(self, query, verbose=False):
        if self._no_facets or not query.get_raw_query():
            return [], [], {}
        query_features = self._extract_query_features(query)
        if len(query_features) == 0:
            return [], [], {}
        query_features[0]['prev-tag'] = util.START_TAG

        predicted_tags = []
        predicted_probs = []
        reference_tags = util.get_tags_from_facets(query, self._tag_scheme)
        reference_probs = []
        confidence_data = []
        active_features = []

        # TODO (julius): Use the maximum likelihood Viterbi parse instead of
        # picking the max at each step.
        for i, features in enumerate(query_features):
            feat_vec = self._feat_vectorizer.transform(query_features[i])
            if self._feat_scaler is not None:
                feat_vec = self._feat_scaler.transform(feat_vec)
            if self._feat_selector is not None:
                feat_vec = self._feat_selector.transform(feat_vec)
            prediction = self._clf.predict(feat_vec)
            log_probs = self._clf.predict_log_proba(feat_vec)

            confidence_dump = numpy.array([])
            feat_names = numpy.array([])
            try:
                reference_class = self._class_encoder.transform([reference_tags[i]])
            except ValueError:
                # If the gold parse uses tags that were never observed in the
                # training data, the classifier will not be able to encode the
                # tag or evaluate its probability. To prevent complete failure
                # we use the Outside tag for reference.
                out_tag = '|'.join([util.O_TAG,'', util.O_TAG,''])
                reference_class = self._class_encoder.transform([out_tag])
                logging.warning('Unknown tag {} replaced with {}'.format(
                    reference_tags[i], out_tag))
            if verbose and reference_class != prediction[0]:
                confidence_dump, feat_names = self._decompose_confidence(
                    feat_vec, (prediction[0], reference_class))

            predicted_tag = self._class_encoder.inverse_transform(prediction)[0]
            predicted_tags.append(predicted_tag)
            predicted_probs.append(log_probs[0][prediction[0]])
            reference_probs.append(log_probs[0][reference_class])
            confidence_data.append(confidence_dump)
            active_features.append(feat_names)

            if i + 1 < len(query_features):
                query_features[i+1]['prev-tag'] = predicted_tag

        facets, num_facets = util.get_facets_from_tags(query, predicted_tags)

        # Collect diagnostic details
        data = {'tokens': query.get_normalized_tokens(),
                'features': query_features,
                'tags': predicted_tags,
                'gold_tags': reference_tags,
                'probs': predicted_probs,
                'gold_probs': reference_probs,
                'conf_data': confidence_data,
                'feat_names': active_features}

        return facets, num_facets, data

    @staticmethod
    def _print_predict_data(data):
        """Print diagnostic prediction data for a single query"""

        print('\t'.join(["{:18}".format(s) for s in
                         ['Token', 'Pred Tag', '(Gold Tag)', '(Log Prob)']]))
        print('\t'.join(['-' * 18 for _ in range(4)]))

        for idx, token in enumerate(data['tokens']):
            if data['tags'][idx] == data['gold_tags'][idx]:
                row_dat = [token, data['tags'][idx], '"']
                print('\t'.join(["{:18}".format(s) for s in row_dat]))
                print
            else:
                row_dat = [token, data['tags'][idx], data['gold_tags'][idx],
                       data['gold_probs'][idx] - data['probs'][idx]]
                print('\t'.join(["{:18}".format(s) for s in row_dat]))
                names = data['feat_names'][idx]
                head = ("feat_val", "pred_w", "gold_w",
                        "pred_p", "gold_p", "diff", "name")
                print('\t', '\t'.join(['-' * 8 for _ in range(7)]))
                print('\t', '\t'.join(["{:>8}"] * 6 + ["{}"]).format(*head))
                print('\t', '\t'.join(['-' * 8 for _ in range(7)]))
                format_str = '\t'.join(["{:8.3f}"] * 6 + ["{}"])

                for j, row in enumerate(data['conf_data'][idx].T):
                    row = list(row) + [names[j]]
                    print('\t', format_str.format(*row))
                print('\t', '\t'.join(['-' * 8 for _ in range(7)]))

    def _decompose_confidence(self, feat_vec, classes):

        pointwise_prod = feat_vec.multiply(self._clf.coef_)[classes, :]
        influence = pointwise_prod[1, :] - pointwise_prod[0, :]
        feature_names = numpy.array(self._feat_vectorizer.feature_names_)
        feature_names = feature_names.reshape((1, -1))
        if self._feat_selector is not None:
            feature_names = self._feat_selector.transform(feature_names)

        active_elements = influence.nonzero()
        active_columns = numpy.array(active_elements[1])
        if len(active_columns.shape) > 1:
            # active_columns could be shape (n,) array or (1, n) array
            active_columns = active_columns[0]
        feature_names = feature_names[active_elements]
        if len(feature_names.shape) > 1:
            feature_names = feature_names[0]

        data = vstack([feat_vec[active_elements],
                       self._clf.coef_[numpy.ix_(classes, active_columns)],
                       pointwise_prod[numpy.ix_((0, 1), active_columns)],
                       influence[active_elements]])

        intercepts = self._clf.intercept_[numpy.ix_(classes)]
        intercept_p = intercepts * self._clf.intercept_scaling
        intercept_dat = hstack([self._clf.intercept_scaling,
                                intercepts,
                                intercept_p,
                                intercept_p[1] - intercept_p[0]])
        feature_names = numpy.array(list(feature_names) + [u"Intercepts"])

        data = hstack([data,
                       intercept_dat.T])

        return data.toarray(), feature_names
