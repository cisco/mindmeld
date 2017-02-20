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


class MaxentRoleClassifier(object):

    def __init__(self):
        self._resources = {}
        self.feat_specs = {
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
            'extract-artist-only': {},
            'age-entities': {}
        }

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

    def load_resources(self, **kwargs):
        """

        Args:
            **kwargs: dictionary of resources to load

        """
        self._resources.update(kwargs)

    def extract_features(self, query, facets, facet_index):
        """Extracts feature dicts for each token in a query."""
        features = {}

        for name, kwargs in self.feat_specs.items():
            feat_extractor = FEATURE_NAME_MAP[name](**kwargs)
            features.update(feat_extractor(query, facets, facet_index, self._resources))
        return features

    def fit(self, labeled_queries, domain, intent, facet_names=None, verbose=False):
        training_examples = {}

        for query in labeled_queries:
            facets = query.get_gold_facets()
            for i, facet in enumerate(facets):
                if 'role' in facet:
                    training_examples.setdefault(facet['type'], []).append(
                        (self.extract_features(query, facets, i), facet['role']))

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

    def predict(self, query, domain, intent, facets, verbose=False):
        roles = []
        for i, facet in enumerate(facets):
            if facet['type'] in self._only_one_role:
                roles.append(self._only_one_role[facet['type']])
            elif facet['type'] not in self._clf:
                roles.append(None)
            else:
                features = self.extract_features(query, facets, i)
                feat_vec = self._feat_vectorizers[facet['type']].transform(features)
                Y = self._clf[facet['type']].predict(feat_vec)[0]
                prediction = self._class_encoders[facet['type']].inverse_transform(Y)
                roles.append(prediction)
        return roles


def extract_in_gaz_features():
    def extractor(query, facets, facet_index, resources):
        features = {}
        current_entity = facets[facet_index]

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
    def extractor(query, facets, facet_index, resources):
        features = {}
        tokens = query.get_normalized_tokens()
        current_facet = facets[facet_index]
        current_facet_index = current_facet['start']

        for length, starts in ngram_lengths_to_start_positions.items():
            for start in starts:
                feat_name = 'bag-of-words-before|length:{}|pos:{}'.format(
                        length, start)
                features[feat_name] = get_ngram(tokens, current_facet_index + start, length)

        return features

    return extractor


def extract_bag_of_words_after_features(ngram_lengths_to_start_positions):
    """Returns a bag-of-words feature extractor.

    Args:
        ngram_lengths_to_start_positions (dict):

    Returns:
        (function) The feature extractor.
    """
    def extractor(query, facets, facet_index, resources):
        features = {}
        tokens = query.get_normalized_tokens()
        current_facet = facets[facet_index]
        current_facet_index = current_facet['end']

        for length, starts in ngram_lengths_to_start_positions.items():
            for start in starts:
                feat_name = 'bag-of-words-after|length:{}|pos:{}'.format(
                        length, start)
                features[feat_name] = get_ngram(tokens, current_facet_index + start, length)

        return features

    return extractor


def extract_numeric_candidate_features():
    def extractor(query, facet_index, resources):
        feat_seq = [{} for _ in query.get_normalized_tokens()]
        num_facets = query.get_candidate_numeric_facets(['time', 'interval'])
        for f in num_facets:
            for i in range(f['start'], f['end']+1):
                feat_name = 'num-candidate|type:{}'.format(f['type'])
                feat_seq[i][feat_name] = 1
        return feat_seq

    return extractor


def extract_other_entities_features():
    def extractor(query, facets, facet_index, resources):
        features = {}
        for i, facet in enumerate(facets):
            if i == facet_index:
                continue
            feat_name = 'other-entities|entity_type:{}'.format(facet['type'])
            features[feat_name] = 1

        return features

    return extractor


def extract_operator_value_features():
    def extractor(query, facets, facet_index, resources):
        features = {}
        for i, facet in enumerate(facets):
            if i == facet_index:
                continue
            if facet['type'] == 'operator':
                feat_name = 'operator-entities|value:{}'.format(facet['entity'])
                features[feat_name] = 1

        return features

    return extractor


# TODO: refactor to have an app level role classifier config and make
# facet type/entity features more genaric. App specific features are
# extract_operator_value_features, extract_age_features, and extract_artist_only_feature
def extract_age_features():
    def extractor(query, facets, facet_index, resources):
        features = {}
        for i, facet in enumerate(facets):
            if i == facet_index:
                continue
            if facet['type'] == 'size':
                feat_name = 'age-entities|value:{}'.format(facet['entity'])
                features[feat_name] = 1

        return features

    return extractor


def extract_artist_only_feature():
    def extractor(query, facets, facet_index, resources):
        features = {}
        allowed_facets = {'artist', 'action'}
        found_extra_facet = False
        for i, facet in enumerate(facets):
            if facet['type'] not in allowed_facets:
                found_extra_facet = True
                break
        if not found_extra_facet:
            features['artist-only'] = 1

        return features

    return extractor


FEATURE_NAME_MAP = {
    'bag-of-words-before': extract_bag_of_words_before_features,
    'bag-of-words-after': extract_bag_of_words_after_features,
    'in-gaz': extract_in_gaz_features,
    'other-entities': extract_other_entities_features,
    'operator-entities': extract_operator_value_features,
    'extract-artist-only': extract_artist_only_feature,
    'age-entities': extract_age_features
}
