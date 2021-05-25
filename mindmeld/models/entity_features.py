# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This module contains feature extractors for entities"""
from .helpers import GAZETTEER_RSC, get_ngram, register_entity_feature, requires
from ..core import QueryEntity, Query, NestedEntity


@register_entity_feature(feature_name="in-gaz")
@requires(GAZETTEER_RSC)
def extract_in_gaz_features(**kwargs):
    """Returns a feature extractor that finds any gazetteer matches against the input query"""
    del kwargs

    def _extractor(example, resources):
        _, entities, entity_index = example
        features = {}
        current_entity = entities[entity_index]

        domain_gazes = resources[GAZETTEER_RSC]

        for gaz_name, gaz in domain_gazes.items():
            if isinstance(current_entity, (QueryEntity, NestedEntity)):
                normalized_tokens = tuple(current_entity.normalized_text.split())
            elif isinstance(current_entity, Query):
                normalized_tokens = current_entity.normalized_tokens
            else:
                raise TypeError(
                    f"{current_entity} is of unknown type, expected Query, "
                    f"NestedEntity or QueryEntity type")

            if normalized_tokens in gaz["pop_dict"]:
                feat_name = "in_gaz|type:{}".format(gaz_name)
                features[feat_name] = 1

        return features

    return _extractor


@register_entity_feature(feature_name="bag-of-words-before")
def extract_bag_of_words_before_features(ngram_lengths_to_start_positions, **kwargs):
    """Returns a bag-of-words feature extractor.

    Args:
        ngram_lengths_to_start_positions (dict):

    Returns:
        (function) The feature extractor.
    """
    del kwargs

    def _extractor(example, resources):
        del resources
        query, entities, entity_index = example
        features = {}
        tokens = query.normalized_tokens
        current_entity = entities[entity_index]
        current_entity_token_start = current_entity.token_span.start

        for length, starts in ngram_lengths_to_start_positions.items():
            for start in starts:
                feat_name = "bag_of_words|ngram_before|length:{}|pos:{}".format(
                    length, start
                )
                features[feat_name] = get_ngram(
                    tokens, current_entity_token_start + start, length
                )

        return features

    return _extractor


@register_entity_feature(feature_name="bag-of-words-after")
def extract_bag_of_words_after_features(ngram_lengths_to_start_positions, **kwargs):
    """Returns a bag-of-words feature extractor.

    Args:
        ngram_lengths_to_start_positions (dict):

    Returns:
        (function) The feature extractor.
    """
    del kwargs

    def _extractor(example, resources):
        del resources
        query, entities, entity_index = example
        features = {}
        tokens = query.normalized_tokens
        current_entity = entities[entity_index]
        current_entity_token_end = current_entity.token_span.end

        for length, starts in ngram_lengths_to_start_positions.items():
            for start in starts:
                feat_name = "bag_of_words|ngram_after|length:{}|pos:{}".format(
                    length, start
                )
                features[feat_name] = get_ngram(
                    tokens, current_entity_token_end + start, length
                )

        return features

    return _extractor


@register_entity_feature(feature_name="numeric")
def extract_numeric_candidate_features(**kwargs):
    """Returns a feature extractor that generates features indicating the presence
    of the ``sys_time`` and ``sys_interval`` numeric entities. These numeric entities are
    identified by duckling, the numerical entity recognition service and boosted by
    training data containing the entity labels. Used by the role classifier when the
    ``'numeric'`` feature is specified in the config."""
    del kwargs

    def _extractor(example, resources):
        del resources
        query, _, _ = example
        feat_seq = {}
        sys_entities = query.get_system_entity_candidates(["sys_time", "sys_interval"])
        for ent in sys_entities:
            for i in ent.token_span:
                feat_name = "sys_candidate|type:{}|pos:{}".format(ent.entity.type, i)
                feat_seq[feat_name] = 1
        return feat_seq

    return _extractor


@register_entity_feature(feature_name="other-entities")
def extract_other_entities_features(**kwargs):
    """Returns a feature extractor for all other entities apart from the current entity.\
    Used by the role classifier when the ``'other-entities'`` feature is specified in \
    the config."""
    del kwargs

    def _extractor(example, resources):
        del resources
        _, entities, entity_index = example
        features = {}
        for idx, entity in enumerate(entities):
            if idx == entity_index:
                continue
            feat_name = "other_entities|type:{}".format(entity.entity.type)
            features[feat_name] = 1

        return features

    return _extractor
