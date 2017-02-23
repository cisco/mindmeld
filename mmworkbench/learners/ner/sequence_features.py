# -*- coding: utf-8 -*-
"""This module contains feature extractors for sequence models"""
from __future__ import absolute_import
from __future__ import unicode_literals
from __future__ import division
from builtins import range
from past.utils import old_div

import math
import re

from . import tagging

from ...core import resolve_entity_conflicts


def get_feature_template(template):
    return FEATURE_NAME_MAP[template]


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
        token = (tagging.OUT_OF_BOUNDS_TOKEN if index < 0 or index >= len(tokens)
                 else tokens[index])
        ngram_tokens.append(token)
    return ' '.join(ngram_tokens)


def extract_in_gaz_span_features():
    """Returns a feature extractor for properties of spans in gazetteers
    """
    def _extractor(query, resources):
        def _get_span_features(query, gazes, start, end, ftype, entity):
            tokens = [re.sub(r'\d', '0', t) for t in query.normalized_tokens]
            feat_seq = [{} for _ in tokens]

            pop = gazes[ftype]['pop_dict'][entity]
            p_total = old_div(math.log(sum([g['total_entities'] for g in gazes.values()]) + 1), 2)

            p_ftype = math.log(gazes[ftype]['total_entities'] + 1)
            p_entity = math.log(sum([len(g['index'][entity])
                                     for g in gazes.values()]) + 1)
            p_joint = math.log(len(gazes[ftype]['index'][entity]) + 1)
            for i in range(start, end):

                # Generic non-positional features
                feat_prefix = 'in-gaz|type:{}'.format(ftype)

                # Basic existence features
                feat_seq[i][feat_prefix] = 1
                # Features for ngram before the span
                feat_name = feat_prefix + '|ngram-before|length:{}'.format(1)
                feat_seq[i][feat_name] = get_ngram(tokens, start-1, 1)
                # Features for ngram after the span
                feat_name = feat_prefix + '|ngram-after|length:{}'.format(1)
                feat_seq[i][feat_name] = get_ngram(tokens, end, 1)
                # Features for ngram at start of span
                feat_name = feat_prefix + '|ngram-first|length:{}'.format(1)
                feat_seq[i][feat_name] = get_ngram(tokens, start, 1)
                # Features for ngram at end of span
                feat_name = feat_prefix + '|ngram-last|length:{}'.format(1)
                feat_seq[i][feat_name] = get_ngram(tokens, end-1, 1)

                # Popularity features
                feat_name = feat_prefix + '|pop'
                feat_seq[i][feat_name] = pop

                # Character length features
                feat_name = feat_prefix + '|log-char-len'
                feat_seq[i][feat_name] = math.log(len(entity))
                feat_name = feat_prefix + '|pct-char-len'
                feat_seq[i][feat_name] = float(len(entity)) / float(len(' '.join(tokens)))

                # entity PMI and conditional prob
                feat_name = feat_prefix + '|pmi'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype - p_entity
                feat_name = feat_prefix + '|p_fe'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity
                feat_name = feat_prefix + '|p_ef'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype

                # Positional features
                # Used to distinguish among B/I/E/S tags
                if i == start:
                    pos_attr = 'start'
                elif i == end-1:
                    pos_attr = 'end'
                else:
                    pos_attr = 'cont'

                feat_prefix = 'in-gaz|type:{}|pos:{}'.format(ftype, pos_attr)

                # Basic existence features
                feat_seq[i][feat_prefix] = 1
                # Features for ngram before the span
                feat_name = feat_prefix + '|ngram-before|length:{}'.format(1)
                feat_seq[i][feat_name] = get_ngram(tokens, start-1, 1)
                # Features for ngram after the span
                feat_name = feat_prefix + '|ngram-after|length:{}'.format(1)
                feat_seq[i][feat_name] = get_ngram(tokens, end, 1)
                # Features for ngram at start of span
                feat_name = feat_prefix + '|ngram-first|length:{}'.format(1)
                feat_seq[i][feat_name] = get_ngram(tokens, start, 1)
                # Features for ngram at end of span
                feat_name = feat_prefix + '|ngram-last|length:{}'.format(1)
                feat_seq[i][feat_name] = get_ngram(tokens, end-1, 1)

                # Popularity features
                feat_name = feat_prefix + '|pop'
                feat_seq[i][feat_name] = pop
                # Character length features
                feat_name = feat_prefix + '|log-char-len'
                feat_seq[i][feat_name] = math.log(len(entity))
                feat_name = feat_prefix + '|pct-char-len'
                feat_seq[i][feat_name] = (old_div(float(len(entity)),
                                          len(' '.join(tokens))))

                feat_name = feat_prefix + '|pmi'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype - p_entity
                feat_name = feat_prefix + '|p_fe'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity
                feat_name = feat_prefix + '|p_ef'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype

            # End of span feature
            if end < len(tokens):
                feat_prefix = 'in-gaz|prev|type:{}'.format(ftype)
                feat_name = feat_prefix
                feat_seq[end][feat_name] = 1
                feat_name = feat_prefix + '|log-char-len'
                feat_seq[end][feat_name] = math.log(len(entity))
                feat_name = feat_prefix + '|pct-char-len'
                feat_seq[end][feat_name] = old_div(float(len(entity)), len(' '.join(tokens)))
                feat_name = feat_prefix + '|pmi'
                feat_seq[end][feat_name] = p_total + p_joint - p_ftype - p_entity
                feat_name = feat_prefix + '|p_fe'
                feat_seq[end][feat_name] = p_total + p_joint - p_entity
                feat_name = feat_prefix + '|p_ef'
                feat_seq[end][feat_name] = p_total + p_joint - p_ftype

            return feat_seq

        def get_exact_span_conflict_features(query, gazes, start, end, ent_type_1, ent_type_2,
                                             entity_text):
            feat_seq = [{} for _ in query.normalized_tokens]
            for i in range(start, end):

                feat_prefix = (
                    'in-gaz|conflict|exact|type1:{}|type2:{}'
                    .format(ent_type_1, ent_type_2))

                p_ent_type_1 = math.log(gazes[ent_type_1]['total_entities'] + 1)
                p_ent_type_2 = math.log(gazes[ent_type_2]['total_entities'] + 1)
                p_joint_1 = math.log(len(gazes[ent_type_1]['index'][entity_text]) + 1)
                p_joint_2 = math.log(len(gazes[ent_type_2]['index'][entity_text]) + 1)

                pop_1 = gazes[ent_type_1]['pop_dict'][entity_text]
                pop_2 = gazes[ent_type_2]['pop_dict'][entity_text]

                # Generic non-positional features
                feat_seq[i][feat_prefix] = 1
                feat_name = feat_prefix + '|diff-pop'
                feat_seq[i][feat_name] = pop_1 - pop_2
                feat_name = feat_prefix + '|diff-pmi'
                feat_seq[i][feat_name] = p_ent_type_2 - p_ent_type_1 - p_joint_2 + p_joint_1
                feat_name = feat_prefix + '|diff-p_fe'
                feat_seq[i][feat_name] = p_joint_1 - p_joint_2

            return feat_seq

        def get_gaz_spans(query, domain_gazes, sys_types):
            """Collect tuples of (start index, end index, ngram, facet type)
            tracking ngrams that match with the entity gazetteer data
            """
            in_gaz_spans = []
            tokens = query.normalized_tokens

            # Collect ngrams of plain normalized ngrams
            for start in range(len(tokens)):
                for end in range(start+1, len(tokens)+1):
                    for gaz_name, gaz in domain_gazes.items():
                        ngram = ' '.join(tokens[start:end])
                        if ngram in gaz['pop_dict']:
                            in_gaz_spans.append((start, end, gaz_name, ngram))

            # Check ngrams with flattened numerics against the gazetteer
            # This algorithm iterates through each pair of numeric facets
            # and through every ngram that includes the entire facet span.
            # This limits regular facets to contain at most two numeric facets
            system_entities = query.get_system_entity_candidates(sys_types)

            for gaz_name, gaz in domain_gazes.items():
                for i, num_facet_i in enumerate(system_entities):
                    if num_facet_i['type'] not in gaz['sys_types']:
                        continue
                    # logging.debug('Looking for [{}|num:{}] in {} gazetteer '
                    #               'with known numeric types {}'
                    #               .format(num_facet_i['entity'],
                    #                       num_facet_i['type'],
                    #                       gaz_name, list(gaz['sys_types'])))

                    # Collect ngrams that include all of num_facet_i
                    for start in range(num_facet_i['start']+1):
                        for end in range(num_facet_i['end']+1, len(tokens)+1):
                            ngram, ntoks = get_flattened_ngram(tokens, start, end, num_facet_i, 0)
                            if ngram in gaz['pop_dict']:
                                in_gaz_spans.append((start, end, gaz_name, ngram))

                            # Check if we can fit any other num_facet_j between
                            # num_facet_i and the edge of the ngram
                            for j, num_facet_j in enumerate(system_entities[i+1:]):
                                if (num_facet_j['type'] in gaz['sys_types']
                                    and (start <= num_facet_j['start'])
                                    and (num_facet_j['end'] < end)
                                    and (num_facet_j['end'] < num_facet_i['start']
                                         or num_facet_i['end'] < num_facet_j['start'])):
                                    ngram, ntoks2 = get_flattened_ngram(
                                        ntoks, start, end, num_facet_j, start)
                                    if ngram in gaz['pop_dict']:
                                        in_gaz_spans.append((start, end, gaz_name, ngram))

            return in_gaz_spans

        def get_flattened_ngram(tokens, start, end, num_facet, offset):
            flattened_token = '@' + num_facet['type'] + '@'
            ntoks = (tokens[start-offset:num_facet['start']-offset] +
                     [flattened_token] +
                     [None]*(num_facet['end']-num_facet['start']) +
                     tokens[num_facet['end']+1-offset:end-offset])
            ngram = ' '.join([t for t in ntoks if t is not None])
            return ngram, ntoks

        gazetteers = resources['gazetteers']
        feat_seq = [{} for _ in query.normalized_tokens]
        sys_types = set()
        for gaz in gazetteers.values():
            sys_types.update(gaz['sys_types'])
        sys_types = list(sys_types)

        in_gaz_spans = get_gaz_spans(query, gazetteers, sys_types)

        # Sort the spans by their indices. The algorithm below assumes this
        # sort order.
        in_gaz_spans.sort()
        while in_gaz_spans:
            span = in_gaz_spans.pop(0)
            span_feat_seq = _get_span_features(query, gazetteers, *span)
            update_features_sequence(feat_seq, span_feat_seq)
            # logging.debug(span_feat_seq)

            for other_span in in_gaz_spans:
                if other_span[0] >= span[1]:
                    break
                # For now, if two spans of the same type start at the same
                # place, take the longer one.
                if other_span[0] == span[0] and other_span[2] == span[2]:
                    continue
                if span[0] == other_span[0]:
                    if span[1] == other_span[1]:
                        cmp_span_features = get_exact_span_conflict_features(
                            query, domain_gazes, span[0], span[1], span[2],
                            other_span[2], span[3])
                        update_features_sequence(feat_seq, cmp_span_features)

        return feat_seq

    return _extractor


def extract_in_gaz_ngram_features():
    """Returns a feature extractor for surrounding ngrams in gazetteers
    """
    def _extractor(query, resources):

        def get_ngram_gaz_features(query, gazes, ftype):
            tokens = query.normalized_tokens
            feat_seq = [{} for _ in tokens]

            for i in range(len(feat_seq)):
                feat_prefix = 'in-gaz-ngram|type:{}'.format(ftype)
                feat_name = feat_prefix + '|idf-0'
                feat_seq[i][feat_name] = math.log(
                    len(gazes[ftype]['index'][get_ngram(tokens, i, 1)]) + 1)
                feat_name = feat_prefix + '|idf-1'
                feat_seq[i][feat_name] = math.log(
                    len(gazes[ftype]['index'][get_ngram(tokens, i-1, 2)]) + 1)
                feat_name = feat_prefix + '|idf+1'
                feat_seq[i][feat_name] = math.log(
                    len(gazes[ftype]['index'][get_ngram(tokens, i, 2)]) + 1)

                # entity PMI and conditional prob
                p_total = old_div(math.log(sum([g['total_entities']
                                                for g in gazes.values()]) + 1), 2)
                p_ftype = math.log(gazes[ftype]['total_entities'] + 1)
                p_ngram = math.log(sum([len(g['index'][get_ngram(tokens, i, 1)])
                                        for g in gazes.values()]) + 1)
                p_joint = math.log(len(gazes[ftype]['index'][get_ngram(tokens, i, 1)]) + 1)
                feat_name = feat_prefix + '|pmi_1'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype - p_ngram
                feat_name = feat_prefix + '|p_fe_1'
                feat_seq[i][feat_name] = p_total + p_joint - p_ngram
                feat_name = feat_prefix + '|p_ef_1'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype

                p_ngram = math.log(sum([len(g['index'][get_ngram(tokens, i-1, 2)])
                                        for g in gazes.values()]) + 1)
                p_joint = math.log(len(gazes[ftype]['index'][get_ngram(tokens, i-1, 2)]) + 1)
                feat_name = feat_prefix + '|pmi-2'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype - p_ngram
                feat_name = feat_prefix + '|p_fe-2'
                feat_seq[i][feat_name] = p_total + p_joint - p_ngram
                feat_name = feat_prefix + '|p_ef-2'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype

                p_ngram = math.log(sum([len(g['index'][get_ngram(tokens, i, 2)])
                                        for g in gazes.values()]) + 1)
                p_joint = math.log(len(gazes[ftype]['index'][get_ngram(tokens, i, 2)]) + 1)
                feat_name = feat_prefix + '|pmi+2'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype - p_ngram
                feat_name = feat_prefix + '|p_fe+2'
                feat_seq[i][feat_name] = p_total + p_joint - p_ngram
                feat_name = feat_prefix + '|p_ef+2'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype

                p_ngram = math.log(sum([len(g['index'][get_ngram(tokens, i-1, 3)])
                                        for g in gazes.values()]) + 1)
                p_joint = math.log(len(gazes[ftype]['index'][get_ngram(tokens, i-1, 3)]) + 1)
                feat_name = feat_prefix + '|pmi_3'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype - p_ngram
                feat_name = feat_prefix + '|p_fe_3'
                feat_seq[i][feat_name] = p_total + p_joint - p_ngram
                feat_name = feat_prefix + '|p_ef_3'
                feat_seq[i][feat_name] = p_total + p_joint - p_ftype

            return feat_seq

        domain_gazes = resources['gazetteers']
        tokens = query.normalized_tokens
        feat_seq = [{} for _ in tokens]

        for ftype in domain_gazes:
            feats = get_ngram_gaz_features(query, domain_gazes, ftype)
            update_features_sequence(feat_seq, feats)

        return feat_seq

    return _extractor


def extract_bag_of_words_features(ngram_lengths_to_start_positions):
    """Returns a bag-of-words feature extractor.
    Args:
        ngram_lengths_to_start_positions (dict):
    Returns:
        (function) The feature extractor.
    """
    # pylint: disable=locally-disabled,unused-argument
    def _extractor(query, resources):
        tokens = query.normalized_tokens
        tokens = [re.sub(r'\d', '0', t) for t in tokens]
        feat_seq = [{} for _ in tokens]
        for i in range(len(tokens)):
            for length, starts in ngram_lengths_to_start_positions.items():
                for start in starts:
                    feat_name = 'bag-of-words|length:{}|pos:{}'.format(
                        length, start)
                    feat_seq[i][feat_name] = get_ngram(tokens, i+int(start), int(length))
        return feat_seq

    return _extractor


def extract_sys_candidate_features(start_positions=(0,)):
    """Return an extractor for features based on a heuristic guess of numeric
    candidates at/near the current token.
    Args:
        start_positions (tuple): positions relative to current token (=0)
    Returns:
        (function) The feature extractor.
    """
    def _extractor(query, resources):
        feat_seq = [{} for _ in query.normalized_tokens]
        system_entities = query.get_system_entity_candidates(resources['sys_types'])
        resolve_entity_conflicts([system_entities])
        for entity in system_entities:
            for i in range(entity['start'], entity['end']+1):
                for j in start_positions:
                    if 0 < i-j < len(feat_seq):
                        feat_name = 'sys-candidate|type:{}:{}|pos:{}'.format(
                            entity['type'], entity.get('grain'), j)
                        feat_seq[i-j][feat_name] = 1
                        feat_name = 'sys-candidate|type:{}:{}|pos:{}|log-len'.format(
                            entity['type'], entity.get('grain'), j)
                        feat_seq[i-j][feat_name] = math.log(len(entity['entity']))
        return feat_seq

    return _extractor


def update_features_sequence(feat_seq, update_feat_seq):
    """Update a list of features with another parallel list of features.

    Args:
        feat_seq (list of dict): The original list of feature dicts which gets
            mutated.
        update_feat_seq (list of dict): The list of features to update with.
    """
    for i in range(len(feat_seq)):
        feat_seq[i].update(update_feat_seq[i])


FEATURE_NAME_MAP = {
    'bag-of-words': extract_bag_of_words_features,
    'in-gaz-span': extract_in_gaz_span_features,
    'in-gaz-ngram': extract_in_gaz_ngram_features,
    'sys-candidates': extract_sys_candidate_features
}
