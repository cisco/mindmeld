# -*- coding: utf-8 -*-
"""This module contains feature extractors for entities"""
from __future__ import absolute_import, unicode_literals

from .helpers import GAZETTEER_RSC, register_features, get_ngram


def extract_in_gaz_features():
    def extractor(example, resources):
        _, entities, entity_index = example
        features = {}
        current_entity = entities[entity_index]

        domain_gazes = resources[GAZETTEER_RSC]

        for gaz_name, gaz in domain_gazes.items():
            if current_entity.normalized_text in gaz['pop_dict']:
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
    def extractor(example, resources):
        query, entities, entity_index = example
        features = {}
        tokens = query.normalized_tokens
        current_entity = entities[entity_index]
        current_entity_token_start = current_entity.token_span.start

        for length, starts in ngram_lengths_to_start_positions.items():
            for start in starts:
                feat_name = 'bag-of-words-before|length:{}|pos:{}'.format(length, start)
                features[feat_name] = get_ngram(tokens, current_entity_token_start + start, length)

        return features

    return extractor


def extract_bag_of_words_after_features(ngram_lengths_to_start_positions):
    """Returns a bag-of-words feature extractor.

    Args:
        ngram_lengths_to_start_positions (dict):

    Returns:
        (function) The feature extractor.
    """
    def extractor(example, resources):
        query, entities, entity_index = example
        features = {}
        tokens = query.normalized_tokens
        current_entity = entities[entity_index]
        current_entity_token_end = current_entity.token_span.end

        for length, starts in ngram_lengths_to_start_positions.items():
            for start in starts:
                feat_name = 'bag-of-words-after|length:{}|pos:{}'.format(length, start)
                features[feat_name] = get_ngram(tokens, current_entity_token_end + start, length)

        return features

    return extractor


def extract_numeric_candidate_features():
    def extractor(example, resources):
        query, _, _ = example
        feat_seq = [{}] * len(query.normalized_tokens)
        sys_entities = query.get_system_entity_candidates(['time', 'interval'])
        for ent in sys_entities:
            for i in ent.token_span:
                feat_name = 'num-candidate|type:{}'.format(ent.entity.type)
                feat_seq[i][feat_name] = 1
        return feat_seq

    return extractor


def extract_other_entities_features():
    def extractor(example, resources):
        _, entities, entity_index = example
        features = {}
        for idx, entity in enumerate(entities):
            if idx == entity_index:
                continue
            feat_name = 'other-entities|entity_type:{}'.format(entity.entity.type)
            features[feat_name] = 1

        return features

    return extractor


def extract_operator_value_features():
    def extractor(example, resources):
        _, entities, entity_index = example
        features = {}
        for idx, entity in enumerate(entities):
            if idx == entity_index:
                continue
            if entity.entity.type == 'operator':
                feat_name = 'operator-entities|value:{}'.format(entity.normalized_text)
                features[feat_name] = 1

        return features

    return extractor


# TODO: refactor to have an app level role classifier config and make
# entity type/entity features more generic. App specific features are
# extract_operator_value_features, extract_age_features
def extract_age_features():
    def extractor(example, resources):
        _, entities, entity_index = example
        features = {}
        for i, entity in enumerate(entities):
            if i == entity_index:
                continue
            if entity.entity.type == 'size':
                feat_name = 'age-entities|value:{}'.format(entity.normalized_text)
                features[feat_name] = 1

        return features

    return extractor


register_features('entity', {
    'bag-of-words-before': extract_bag_of_words_before_features,
    'bag-of-words-after': extract_bag_of_words_after_features,
    'in-gaz': extract_in_gaz_features,
    'other-entities': extract_other_entities_features,
    'operator-entities': extract_operator_value_features,
    'age-entities': extract_age_features
})
