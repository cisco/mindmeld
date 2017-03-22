# -*- coding: utf-8 -*-
"""

"""
from __future__ import unicode_literals

from builtins import range
from builtins import object
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

OUT_OF_BOUNDS_TOKEN = '<$>'


def get_ngram(tokens, start, length):
    """Gets a ngram from a list of tokens.

    Handles out-of-bounds token positions with a special character.

    Args:
        tokens (list of str): Word tokens.
        start (int): The index of the desired ngram's start position.
        length (int): The length of the n-gram, e.g. 1 for unigram, etc.

    Returns:
        (str) An n-gram in the input token list.
    """

    ngram_tokens = []
    for index in range(start, start+length):
        token = (OUT_OF_BOUNDS_TOKEN if index < 0 or index >= len(tokens)
                 else tokens[index])
        ngram_tokens.append(token)
    return ' '.join(ngram_tokens)


class MaxentRoleModel(object):

    def __init__(self):
        self._resources = {}
        self.feat_specs = {}
        self._clf = {}
        self._class_encoders = {}
        self._feat_vectorizers = {}
        # Map from entity types with only one role to the name of that role.
        self._only_one_role = {}

    def __getstate__(self):
        """Returns the information needed pickle an instance of this class.

        By default, pickling removes attributes with names starting with
        underscores. This overrides that behavior. For the _resources field,
        we save the resources that are memory intensive
        """
        attributes = dict(self.__dict__)
        saved_resources = []
        attributes['_resources'] = dict((resource, self._resources[resource])
                                        for resource in saved_resources)
        return attributes

    def register_resources(self, **kwargs):
        """

        Args:
            **kwargs: dictionary of resources to load

        """
        self._resources.update(kwargs)

    def extract_features(self, query, entities, entity_index):
        """Extracts feature dicts for each token in a query."""
        features = {}

        for name, kwargs in self.feat_specs.items():
            feat_extractor = FEATURE_NAME_MAP[name](**kwargs)
            features.update(feat_extractor(query, entities, entity_index, self._resources))
        return features

    def fit(self, labeled_queries, domain, intent, entity_names=None, verbose=False):
        training_examples = {}

        for query in labeled_queries:
            entities = query.get_gold_entities()
            for i, entity in enumerate(entities):
                if 'role' in entity:
                    training_examples.setdefault(entity['type'], []).append(
                        (self.extract_features(query, entities, i), entity['role']))

        for entity_type, examples in training_examples.items():
            feat_vectorizer = DictVectorizer()
            class_encoder = LabelEncoder()
            classifier = LogisticRegression()
            X = feat_vectorizer.fit_transform([e[0] for e in examples])
            classes = [e[1] for e in examples]
            if len(set(classes)) == 1:
                self._only_one_role[entity_type] = classes[0]
            else:
                Y = class_encoder.fit_transform([e[1] for e in examples])
                classifier.fit(X, Y)
                self._clf[entity_type] = classifier
                self._class_encoders[entity_type] = class_encoder
                self._feat_vectorizers[entity_type] = feat_vectorizer

    def predict(self, query, domain, intent, entities, verbose=False):
        roles = []
        for i, entity in enumerate(entities):
            if entity['type'] in self._only_one_role:
                roles.append(self._only_one_role[entity['type']])
            elif entity['type'] not in self._clf:
                roles.append(None)
            else:
                features = self.extract_features(query, entities, i)
                feat_vec = self._feat_vectorizers[entity['type']].transform(features)
                Y = self._clf[entity['type']].predict(feat_vec)[0]
                prediction = self._class_encoders[entity['type']].inverse_transform(Y)
                roles.append(prediction)
        return roles


def extract_in_gaz_features():
    def extractor(query, entities, entity_index, resources):
        features = {}
        current_entity = entities[entity_index]

        domain_gazes = resources['gazetteers']

        for gaz_name, gaz in domain_gazes.items():
            if current_entity['entity'] in gaz['edict']:
                feat_name = 'in-gaz|gaz:{}'.format(gaz_name)
                features[feat_name] = 1

        return features

    return extractor


def extract_bag_of_words_before_features(ngram_lengths_to_start_positions):
    """Returns a bag-of-words feature extractor.

    Args:
        ngram_lengths_to_start_positions (dict):

    Returns:
        (function) The feature extractor.
    """
    def extractor(query, entities, entity_index, resources):
        features = {}
        tokens = query.get_normalized_tokens()
        current_entity = entities[entity_index]
        current_entity_index = current_entity['start']

        for length, starts in ngram_lengths_to_start_positions.items():
            for start in starts:
                feat_name = 'bag-of-words-before|length:{}|pos:{}'.format(
                        length, start)
                features[feat_name] = get_ngram(tokens, current_entity_index + start, length)

        return features

    return extractor


def extract_bag_of_words_after_features(ngram_lengths_to_start_positions):
    """Returns a bag-of-words feature extractor.

    Args:
        ngram_lengths_to_start_positions (dict):

    Returns:
        (function) The feature extractor.
    """
    def extractor(query, entities, entity_index, resources):
        features = {}
        tokens = query.get_normalized_tokens()
        current_entity = entities[entity_index]
        current_entity_index = current_entity['end']

        for length, starts in ngram_lengths_to_start_positions.items():
            for start in starts:
                feat_name = 'bag-of-words-after|length:{}|pos:{}'.format(
                        length, start)
                features[feat_name] = get_ngram(tokens, current_entity_index + start, length)

        return features

    return extractor


def extract_numeric_candidate_features():
    def extractor(query, entity_index, resources):
        feat_seq = [{} for _ in query.get_normalized_tokens()]
        num_entities = query.get_candidate_numeric_entities(['time', 'interval'])
        for f in num_entities:
            for i in range(f['start'], f['end']+1):
                feat_name = 'num-candidate|type:{}'.format(f['type'])
                feat_seq[i][feat_name] = 1
        return feat_seq

    return extractor


def extract_other_entities_features():
    def extractor(query, entities, entity_index, resources):
        features = {}
        for i, entity in enumerate(entities):
            if i == entity_index:
                continue
            feat_name = 'other-entities|entity_type:{}'.format(entity['type'])
            features[feat_name] = 1

        return features

    return extractor


def extract_operator_value_features():
    def extractor(query, entities, entity_index, resources):
        features = {}
        for i, entity in enumerate(entities):
            if i == entity_index:
                continue
            if entity['type'] == 'operator':
                feat_name = 'operator-entities|value:{}'.format(entity['entity'])
                features[feat_name] = 1

        return features

    return extractor


# TODO: refactor to have an app level role classifier config and make
# entity type/entity features more genaric. App specific features are
# extract_operator_value_features, extract_age_features, and extract_artist_only_feature
def extract_age_features():
    def extractor(query, entities, entity_index, resources):
        features = {}
        for i, entity in enumerate(entities):
            if i == entity_index:
                continue
            if entity['type'] == 'size':
                feat_name = 'age-entities|value:{}'.format(entity['entity'])
                features[feat_name] = 1

        return features

    return extractor


FEATURE_NAME_MAP = {
    'bag-of-words-before': extract_bag_of_words_before_features,
    'bag-of-words-after': extract_bag_of_words_after_features,
    'in-gaz': extract_in_gaz_features,
    'other-entities': extract_other_entities_features,
    'operator-entities': extract_operator_value_features,
    'age-entities': extract_age_features
}
