# -*- coding: utf-8 -*-
"""This module contains the Memm entity recognizer."""
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from builtins import next, range, str, super
from past.utils import old_div

import logging

from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn import cross_validation
from scipy.sparse import vstack, hstack
import numpy

from . import query_features, tagging
from .base import BaseEntityRecognizer

DEFAULT_FEATURES = {
    'bag-of-words': {
        'ngram_lengths_to_start_positions': {
            1: [-2, -1, 0, 1, 2],
            2: [-2, -1, 0, 1]
        }
    },
    'in-gaz-span': {},
    'sys-candidates': {
        'start_positions': [-1, 0, 1]
    }
}


class MemmModel(BaseEntityRecognizer):
    """A maximum-entropy Markov model for entity recognition.

    This class implements a conditional sequence model that predicts tags that
    are a joint representation of numeric and non-numeric entities.

    Attributes:
        feat_specs (dict): A mapping from feature extractor names, as given in
            FEATURE_NAME_MAP, to a kwargs dict, which will be passed into the
            associated feature extractor function.
    """

    def __init__(self, params_grid=None, cv=None, model_settings=None, features=None):
        features = features or DEFAULT_FEATURES
        super().__init__(params_grid, cv, model_settings, features)

        # Default tag scheme to IOB
        self._tag_scheme = self.classifier_settings.get('tag-scheme', 'IOB').upper()
        self._feat_selector = self.get_feature_selector()
        self._feat_scaler = self.get_feature_scaler()

        self._clf = None
        self._class_encoder = LabelEncoder()
        self._feat_vectorizer = DictVectorizer()

        self._no_entities = False
        self._model_parameters = None
        self.entity_types = set()

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

    def _extract_query_features(self, query):
        """Extracts feature dicts for each token in a query."""
        feat_seq = [{} for _ in query.normalized_tokens]

        for name, kwargs in self.feat_specs.items():
            feat_extractor = sequence_features.get_feature_template(name)(**kwargs)
            update_feat_seq = feat_extractor(query, self._resources)
            for i in range(len(feat_seq)):
                feat_seq[i].update(update_feat_seq[i])

        return feat_seq

    def _clf_fit_cv(self, X, y, query_groups, cv_iterator):
        best_score = None
        best_settings = None
        metric = self.cross_validation_settings.get("metric", "accuracy")

        for settings in self._iter_settings():
            logging.info("Fitting entity recognizer with settings: {}".format(
                settings))

            self._model_parameters = dict(settings)
            self._clf = LogisticRegression(**self._model_parameters)

            scores = cross_validation.cross_val_score(self._clf, X, y,
                                                      metric,
                                                      cv_iterator(query_groups))

            mean_score = numpy.mean(scores)
            std_err = old_div(numpy.std(scores), numpy.sqrt(len(scores)))
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

    def fit(self, labeled_queries, entity_types=None, verbose=False):
        """Trains the model

        Args:
            labeled_queries (list of ProcessedQuery): a list of queries to train on
            entity_types (list): entity types as a filter (defaults to all)
            verbose (boolean): show more debug/diagnostic output

        """

        # all_features and all_tags are parallel lists of feature dictionaries
        # and their associated gold tags, respectively. They are concatenations
        # of the features and tags across all the input queries.
        all_features = []
        all_tags = []
        all_query_groups = []

        system_types = self._get_system_types()
        self.register_resources(sys_types=system_types)

        entity_types = set()
        for idx, query in enumerate(labeled_queries):
            for entity in query.entities:
                entity_types.add(entity.entity.type)

            features = self._extract_query_features(query.query)
            tags = tagging.get_tags_from_entities(query, self._tag_scheme)
            query_groups = [idx for _ in tags]

            if len(features) > 0:
                # Set the special feature for the identity of the previous tag.
                features[0]['prev_tag'] = tagging.START_TAG
                for i, tag in enumerate(tags[:-1]):
                    features[i+1]['prev_tag'] = str(tag)

            all_features.extend(features)
            all_tags.extend(tags)
            all_query_groups.extend(query_groups)

        self.entity_types = frozenset(entity_types)

        if len(set(all_tags)) == 1:
            self._no_entities = True
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
                self._model_parameters = dict(next(self._iter_settings()))
                self._clf = LogisticRegression(**self._model_parameters)
                self._clf.fit(X, y)
            else:
                self._clf_fit_cv(X, y, all_query_groups, cv_iterator)

    def predict(self, query, entity_types=None, verbose=False):
        entities, data = self._predict(query, verbose)
        if verbose:
            self._print_predict_data(data)

        # Assign a confidence value for each entity.
        for entity in entities:
            max_cost = min([data['probs'][i] for i in entity.normalized_token_span])
            entity.entity.confidence = numpy.exp(max_cost)

        if verbose:
            return entities, data
        return entities

    def _predict(self, query, verbose=False):
        if self._no_entities or not query.text:
            return [], {}
        query_features = self._extract_query_features(query)
        if len(query_features) == 0:
            return [], {}
        query_features[0]['prev_tag'] = tagging.START_TAG

        predicted_tags = []
        predicted_probs = []
        # reference_tags = tagging.get_tags_from_entities(query, self._tag_scheme)
        # reference_probs = []
        confidence_data = []
        active_features = []
        all_log_probs = []

        # TODO (julius): Use the maximum likelihood Viterbi parse instead of
        # picking the max at each step.
        for i, features in enumerate(query_features):
            feat_vec = self._feat_vectorizer.transform(features)
            if self._feat_scaler is not None:
                feat_vec = self._feat_scaler.transform(feat_vec)
            if self._feat_selector is not None:
                feat_vec = self._feat_selector.transform(feat_vec)
            prediction = self._clf.predict(feat_vec)
            log_probs = self._clf.predict_log_proba(feat_vec)

            confidence_dump = numpy.array([])
            feat_names = numpy.array([])
            # try:
            #     reference_class = self._class_encoder.transform([reference_tags[i]])
            # except ValueError:
            #     # If the gold parse uses tags that were never observed in the
            #     # training data, the classifier will not be able to encode the
            #     # tag or evaluate its probability. To prevent complete failure
            #     # we use the Outside tag for reference.
            #     out_tag = '|'.join([tagging.O_TAG, '', tagging.O_TAG, ''])
            #     reference_class = self._class_encoder.transform([out_tag])
            #     logging.warning('Unknown tag {} replaced with {}'.format(
            #         reference_tags[i], out_tag))
            # if verbose and reference_class != prediction[0]:
            #     confidence_dump, feat_names = self._decompose_confidence(
            #         feat_vec, (prediction[0], reference_class))

            predicted_tag = self._class_encoder.inverse_transform(prediction)[0]
            predicted_tags.append(predicted_tag)
            predicted_probs.append(log_probs[0][prediction[0]])
            # reference_probs.append(log_probs[0][reference_class])
            confidence_data.append(confidence_dump)

            # Return the log probability information
            tag_labels = self._class_encoder.inverse_transform(list(range(len(log_probs[0]))))
            log_prob_dict = {}
            for j in range(len(tag_labels)):
                log_prob_dict[tag_labels[j]] = log_probs[0][j]
            all_log_probs.append(log_prob_dict)

            active_features.append(feat_names)

            if i + 1 < len(query_features):
                query_features[i+1]['prev_tag'] = predicted_tag

        entities = tagging.get_entities_from_tags(query, predicted_tags)

        # Collect diagnostic details
        data = {'tokens': query.normalized_tokens,
                'features': query_features,
                'tags': predicted_tags,
                # 'gold_tags': reference_tags,
                'probs': predicted_probs,
                # 'gold_probs': reference_probs,
                'conf_data': confidence_data,
                'feat_names': active_features,
                'all_log_probs': all_log_probs}

        return entities, data

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
                print()
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

    def _get_system_types(self):
        sys_types = set()
        for gaz in self._resources['gazetteers'].values():
            sys_types.update(gaz['sys_types'])
        return sys_types


def get_tags_from_entities(query, scheme='IOB'):
    """Get joint app and system IOB tags from a query's entities.

    Args:
        query (ProcessedQuery): An annotated query instance.

    Returns:
        (list of str): The tags for each token in the query. A tag has four
            parts separated by '|'. The first two are the IOB status for
            app entities followed by the type of app entity or
            '' if the IOB status is 'O'. The last two are like the first two,
            but for system facets.
    """

    # Normal entities
    app_entities = [e for e in query.entities if not e.entity.is_system_entity]
    iobs, app_types = _get_tags_from_entities(query, app_entities, scheme)

    # System entities
    # This algorithm assumes that the query system entities are well-formed and
    # only occur as standalone or fully inside an app entity.
    sys_entities = [e for e in query.entities if e.entity.is_system_entity]
    sys_iobs, sys_types = _get_tags_from_entities(query, sys_entities, scheme)

    tags = ['|'.join(args) for args in
            zip(iobs, app_types, sys_iobs, sys_types)]

    return tags


def _get_tags_from_entities(query, entities, scheme='IOB'):
    normalized_tokens = query.query.normalized_tokens
    iobs = [O_TAG for _ in normalized_tokens]
    types = ['' for _ in normalized_tokens]

    # tag I and type for all tag schemes
    for entity in entities:

        for i in entity.normalized_token_span:
            iobs[i] = I_TAG
            types[i] = entity.entity.type

    # Replace I with B/E/S when appropriate
    if scheme in ('IOB', 'IOBES'):
        for entity in entities:
            iobs[entity.normalized_token_span.start] = B_TAG
    if scheme == 'IOBES':
        for entity in entities:
            if len(entity.normalized_token_span) == 1:
                iobs[entity.normalized_token_span.end] = S_TAG
            else:
                iobs[entity.normalized_token_span.end] = E_TAG

    return iobs, types


def get_entities_from_tags(query, tags):
    """From a set of joint IOB tags, parse the app and system entities.

    This performs the reverse operation of get_tags_from_entities.

    Args:
        query (Query): Any query instance.
        tags (list of str): Joint app and system tags, like those
            created by get_tags_from_entities.

    Returns:
        (list of QueryEntity) The tuple containing the list of entities.
    """

    normalized_tokens = query.normalized_tokens

    entities = []

    def _append_entity(token_start, entity_type, tokens):
        prefix = ' '.join(normalized_tokens[:token_start])
        # If there is a prefix, we have to add one for the whitespace
        start = len(prefix) + 1 if len(prefix) else 0
        end = start - 1 + len(' '.join(tokens))
        norm_span = Span(start, end)
        entity = QueryEntity.from_query(query, normalized_span=norm_span, entity_type=entity_type)
        entities.append(entity)
        logger.debug("Appended {}".format(entity))

    def _append_system_entity(token_start, token_end, entity_type):
        msg = "Looking for '{}' between {} and {}"
        logger.debug(msg.format(entity_type, token_start, token_end))
        prefix = ' '.join(normalized_tokens[:token_start])
        # If there is a prefix, we have to add one for the whitespace
        start = len(prefix) + 1 if len(prefix) else 0
        end = start - 1 + len(' '.join(normalized_tokens[token_start:token_end]))
        norm_span = Span(start, end)

        span = query.transform_span(norm_span, TEXT_FORM_NORMALIZED, TEXT_FORM_RAW)

        try:
            entity = resolve_system_entity(query, entity_type, span)
            entities.append(entity)
            logger.debug("Appended system entity {}".format(entity))
        except SystemEntityResolutionError:
            msg = "Found no matching system entity {}-{}, {!r}"
            logger.debug(msg.format(token_start, token_end, entity_type))

    entity_tokens = []
    entity_start = None
    prev_ent_type = ''
    sys_entity_start = None
    prev_sys_type = ''

    for tag_idx, tag in enumerate(tags):
        iob, ent_type, sys_iob, sys_type = tag.split('|')

        # Close sysem entity and reset if the tag indicates a new entity
        if (sys_entity_start is not None and
                (sys_iob in (O_TAG, B_TAG, S_TAG) or sys_type != prev_sys_type)):
            logger.debug("System entity closed at prev")
            _append_system_entity(sys_entity_start, tag_idx, prev_sys_type)
            sys_entity_start = None
            prev_sys_type = ''

        # Close regular facet and reset if the tag indicates a new facet
        if (entity_start is not None and
                (iob in (O_TAG, B_TAG, S_TAG) or ent_type != prev_ent_type)):
            logger.debug("Entity closed at prev")
            _append_entity(entity_start, prev_ent_type, entity_tokens)
            entity_start = None
            prev_ent_type = ''
            entity_tokens = []

            # Cut short any numeric facet that might continue beyond the facet
            if sys_entity_start is not None:
                _append_system_entity(sys_entity_start, tag_idx, prev_sys_type)
            sys_entity_start = None
            prev_sys_type = ''

        # Check if a regular facet has started
        if iob in (B_TAG, S_TAG) or ent_type not in ('', prev_ent_type):
            entity_start = tag_idx
        # Check if a numeric facet has started
        if sys_iob in (B_TAG, S_TAG) or sys_type not in ('', prev_sys_type):
            sys_entity_start = tag_idx

        # Append the current token to the current entity, if applicable.
        if iob != O_TAG and entity_start is not None:
            entity_tokens.append(normalized_tokens[tag_idx])

        # Close the numeric facet if the tag indicates it closed
        if (sys_entity_start is not None and
                sys_iob in (E_TAG, S_TAG)):
            logger.debug("System entity closed here")
            _append_system_entity(sys_entity_start, tag_idx+1, sys_type)
            sys_entity_start = None
            sys_type = ''

        # Close the regular facet if the tag indicates it closed
        if (entity_start is not None and
                iob in (E_TAG, S_TAG)):
            logger.debug("Entity closed here")
            _append_entity(entity_start, ent_type, entity_tokens)
            entity_start = None
            ent_type = ''
            entity_tokens = []

            # Cut short any numeric facet that might continue beyond the facet
            if sys_entity_start is not None:
                _append_system_entity(sys_entity_start, tag_idx+1, sys_type)
            sys_entity_start = None
            sys_type = ''

        prev_ent_type = ent_type
        prev_sys_type = sys_type

    # Handle facets that end with the end of the query
    if entity_start is not None:
        logger.debug("Entity closed at end: {}".format(entity_tokens))
        _append_entity(entity_start, prev_ent_type, entity_tokens)
    else:
        logger.debug("Entity did not end: {}".format(entity_start))
    if sys_entity_start is not None:
        _append_system_entity(sys_entity_start, len(tags), prev_sys_type)

    return entities
