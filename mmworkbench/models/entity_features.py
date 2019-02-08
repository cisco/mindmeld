# -*- coding: utf-8 -*-
"""This module contains feature extractors for entities"""
from .helpers import GAZETTEER_RSC, register_entity_feature, get_ngram, requires


@register_entity_feature(feature_name='in-gaz')
@requires(GAZETTEER_RSC)
def extract_in_gaz_features(**args):
    def _extractor(example, resources):
        _, entities, entity_index = example
        features = {}
        current_entity = entities[entity_index]

        domain_gazes = resources[GAZETTEER_RSC]

        for gaz_name, gaz in domain_gazes.items():
            if current_entity.normalized_text in gaz['pop_dict']:
                feat_name = 'in_gaz|type:{}'.format(gaz_name)
                features[feat_name] = 1

        return features

    return _extractor


@register_entity_feature(feature_name='bag-of-words-before')
def extract_bag_of_words_before_features(ngram_lengths_to_start_positions, **args):
    """Returns a bag-of-words feature extractor.

    Args:
        ngram_lengths_to_start_positions (dict):

    Returns:
        (function) The feature extractor.
    """
    def _extractor(example, resources):
        query, entities, entity_index = example
        features = {}
        tokens = query.normalized_tokens
        current_entity = entities[entity_index]
        current_entity_token_start = current_entity.token_span.start

        for length, starts in ngram_lengths_to_start_positions.items():
            for start in starts:
                feat_name = 'bag_of_words|ngram_before|length:{}|pos:{}'.format(length, start)
                features[feat_name] = get_ngram(tokens, current_entity_token_start + start, length)

        return features

    return _extractor


@register_entity_feature(feature_name='bag-of-words-after')
def extract_bag_of_words_after_features(ngram_lengths_to_start_positions, **args):
    """Returns a bag-of-words feature extractor.

    Args:
        ngram_lengths_to_start_positions (dict):

    Returns:
        (function) The feature extractor.
    """
    def _extractor(example, resources):
        query, entities, entity_index = example
        features = {}
        tokens = query.normalized_tokens
        current_entity = entities[entity_index]
        current_entity_token_end = current_entity.token_span.end

        for length, starts in ngram_lengths_to_start_positions.items():
            for start in starts:
                feat_name = 'bag_of_words|ngram_after|length:{}|pos:{}'.format(length, start)
                features[feat_name] = get_ngram(tokens, current_entity_token_end + start, length)

        return features

    return _extractor


@register_entity_feature(feature_name='numeric')
def extract_numeric_candidate_features(**args):
    def _extractor(example, resources):
        query, _, _ = example
        feat_seq = [{}] * len(query.normalized_tokens)
        sys_entities = query.get_system_entity_candidates(['time', 'interval'])
        for ent in sys_entities:
            for i in ent.token_span:
                feat_name = 'sys_candidate|type:{}'.format(ent.entity.type)
                feat_seq[i][feat_name] = 1
        return feat_seq

    return _extractor


@register_entity_feature(feature_name='other-entities')
def extract_other_entities_features(**args):
    def _extractor(example, resources):
        _, entities, entity_index = example
        features = {}
        for idx, entity in enumerate(entities):
            if idx == entity_index:
                continue
            feat_name = 'other_entities|type:{}'.format(entity.entity.type)
            features[feat_name] = 1

        return features

    return _extractor
