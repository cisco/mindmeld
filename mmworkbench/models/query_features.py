# -*- coding: utf-8 -*-
"""This module contains feature extractors for queries"""
from __future__ import absolute_import, unicode_literals, division
from builtins import range, zip
from past.utils import old_div

from collections import Counter, defaultdict
import math
import re

from ..core import resolve_entity_conflicts
from .helpers import (GAZETTEER_RSC, QUERY_FREQ_RSC, SYS_TYPES_RSC, WORD_FREQ_RSC,
                      register_features, mask_numerics, get_ngram, requires)

# TODO: clean this up a LOT


@requires(GAZETTEER_RSC)
def extract_in_gaz_span_features():
    """Returns a feature extractor for properties of spans in gazetteers
    """
    def _extractor(query, resources):
        def _get_span_features(query, gazes, start, end, entity_type, entity):
            tokens = [re.sub(r'\d', '0', t) for t in query.normalized_tokens]
            feat_seq = [{} for _ in tokens]

            pop = gazes[entity_type]['pop_dict'][entity]
            p_total = old_div(math.log(sum([g['total_entities'] for g in gazes.values()]) + 1), 2)

            p_entity_type = math.log(gazes[entity_type]['total_entities'] + 1)
            p_entity = math.log(sum([len(g['index'][entity])
                                     for g in gazes.values()]) + 1)
            p_joint = math.log(len(gazes[entity_type]['index'][entity]) + 1)
            for i in range(start, end):

                # Generic non-positional features
                feat_prefix = 'in-gaz|type:{}'.format(entity_type)

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
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type - p_entity
                feat_name = feat_prefix + '|p_fe'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity
                feat_name = feat_prefix + '|p_ef'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type

                # Positional features
                # Used to distinguish among B/I/E/S tags
                if i == start:
                    pos_attr = 'start'
                elif i == end-1:
                    pos_attr = 'end'
                else:
                    pos_attr = 'cont'

                feat_prefix = 'in-gaz|type:{}|pos:{}'.format(entity_type, pos_attr)

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
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type - p_entity
                feat_name = feat_prefix + '|p_fe'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity
                feat_name = feat_prefix + '|p_ef'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type

            # End of span feature
            if end < len(tokens):
                feat_prefix = 'in-gaz|prev|type:{}'.format(entity_type)
                feat_name = feat_prefix
                feat_seq[end][feat_name] = 1
                feat_name = feat_prefix + '|log-char-len'
                feat_seq[end][feat_name] = math.log(len(entity))
                feat_name = feat_prefix + '|pct-char-len'
                feat_seq[end][feat_name] = old_div(float(len(entity)), len(' '.join(tokens)))
                feat_name = feat_prefix + '|pmi'
                feat_seq[end][feat_name] = p_total + p_joint - p_entity_type - p_entity
                feat_name = feat_prefix + '|p_fe'
                feat_seq[end][feat_name] = p_total + p_joint - p_entity
                feat_name = feat_prefix + '|p_ef'
                feat_seq[end][feat_name] = p_total + p_joint - p_entity_type

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

        # TODO: clean up this method -- currently the parts involving sys_types are
        # completely broken
        def get_gaz_spans(query, gazetteers, sys_types):
            """Collect tuples of (start index, end index, ngram, entity type)
            tracking ngrams that match with the entity gazetteer data
            """
            in_gaz_spans = []
            tokens = query.normalized_tokens

            # Collect ngrams of plain normalized ngrams
            for start in range(len(tokens)):
                for end in range(start+1, len(tokens)+1):
                    for gaz_name, gaz in gazetteers.items():
                        ngram = ' '.join(tokens[start:end])
                        if ngram in gaz['pop_dict']:
                            in_gaz_spans.append((start, end, gaz_name, ngram))

            # Check ngrams with flattened numerics against the gazetteer
            # This algorithm iterates through each pair of numeric entities
            # and through every ngram that includes the entire entity span.
            # This limits regular entities to contain at most two numeric entities
            system_entities = query.get_system_entity_candidates(sys_types)

            for gaz_name, gaz in gazetteers.items():
                for i, num_entity_i in enumerate(system_entities):
                    if num_entity_i['type'] not in gaz['sys_types']:
                        continue
                    # logging.debug('Looking for [{}|num:{}] in {} gazetteer '
                    #               'with known numeric types {}'
                    #               .format(num_entity_i['entity'],
                    #                       num_entity_i['type'],
                    #                       gaz_name, list(gaz['sys_types'])))

                    # Collect ngrams that include all of num_entity_i
                    for start in range(num_entity_i['start']+1):
                        for end in range(num_entity_i['end']+1, len(tokens)+1):
                            ngram, ntoks = get_flattened_ngram(tokens, start, end, num_entity_i, 0)
                            if ngram in gaz['pop_dict']:
                                in_gaz_spans.append((start, end, gaz_name, ngram))

                            # Check if we can fit any other num_entity_j between
                            # num_entity_i and the edge of the ngram
                            for j, num_entity_j in enumerate(system_entities[i+1:]):
                                if (num_entity_j['type'] in gaz['sys_types']
                                    and (start <= num_entity_j['start'])
                                    and (num_entity_j['end'] < end)
                                    and (num_entity_j['end'] < num_entity_i['start']
                                         or num_entity_i['end'] < num_entity_j['start'])):
                                    ngram, ntoks2 = get_flattened_ngram(
                                        ntoks, start, end, num_entity_j, start)
                                    if ngram in gaz['pop_dict']:
                                        in_gaz_spans.append((start, end, gaz_name, ngram))

            return in_gaz_spans

        def get_flattened_ngram(tokens, start, end, num_entity, offset):
            flattened_token = '@' + num_entity['type'] + '@'
            ntoks = (tokens[start-offset:num_entity['start']-offset] +
                     [flattened_token] +
                     [None]*(num_entity['end']-num_entity['start']) +
                     tokens[num_entity['end']+1-offset:end-offset])
            ngram = ' '.join([t for t in ntoks if t is not None])
            return ngram, ntoks

        gazetteers = resources[GAZETTEER_RSC]
        feat_seq = [{} for _ in query.normalized_tokens]
        sys_types = set()
        for gaz in gazetteers.values():
            sys_types.update(gaz['sys_types'])

        in_gaz_spans = get_gaz_spans(query, gazetteers, list(sys_types))

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
                            query, gazetteers, span[0], span[1], span[2],
                            other_span[2], span[3])
                        update_features_sequence(feat_seq, cmp_span_features)

        return feat_seq

    return _extractor


@requires(GAZETTEER_RSC)
def extract_in_gaz_ngram_features():
    """Returns a feature extractor for surrounding ngrams in gazetteers
    """
    def _extractor(query, resources):

        def get_ngram_gaz_features(query, gazes, entity_type):
            tokens = query.normalized_tokens
            feat_seq = [{} for _ in tokens]

            for i in range(len(feat_seq)):
                feat_prefix = 'in-gaz-ngram|type:{}'.format(entity_type)
                feat_name = feat_prefix + '|idf-0'
                feat_seq[i][feat_name] = math.log(
                    len(gazes[entity_type]['index'][get_ngram(tokens, i, 1)]) + 1)
                feat_name = feat_prefix + '|idf-1'
                feat_seq[i][feat_name] = math.log(
                    len(gazes[entity_type]['index'][get_ngram(tokens, i - 1, 2)]) + 1)
                feat_name = feat_prefix + '|idf+1'
                feat_seq[i][feat_name] = math.log(
                    len(gazes[entity_type]['index'][get_ngram(tokens, i, 2)]) + 1)

                # entity PMI and conditional prob
                p_total = old_div(math.log(sum([g['total_entities']
                                                for g in gazes.values()]) + 1), 2)
                p_entity_type = math.log(gazes[entity_type]['total_entities'] + 1)
                p_ngram = math.log(sum([len(g['index'][get_ngram(tokens, i, 1)])
                                        for g in gazes.values()]) + 1)
                p_joint = math.log(len(gazes[entity_type]['index'][get_ngram(tokens, i, 1)]) + 1)
                feat_name = feat_prefix + '|pmi_1'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type - p_ngram
                feat_name = feat_prefix + '|p_fe_1'
                feat_seq[i][feat_name] = p_total + p_joint - p_ngram
                feat_name = feat_prefix + '|p_ef_1'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type

                p_ngram = math.log(sum([len(g['index'][get_ngram(tokens, i-1, 2)])
                                        for g in gazes.values()]) + 1)
                p_joint = math.log(len(gazes[entity_type]['index']
                                       [get_ngram(tokens, i - 1, 2)]) + 1)
                feat_name = feat_prefix + '|pmi-2'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type - p_ngram
                feat_name = feat_prefix + '|p_fe-2'
                feat_seq[i][feat_name] = p_total + p_joint - p_ngram
                feat_name = feat_prefix + '|p_ef-2'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type

                p_ngram = math.log(sum([len(g['index'][get_ngram(tokens, i, 2)])
                                        for g in gazes.values()]) + 1)
                p_joint = math.log(len(gazes[entity_type]['index'][get_ngram(tokens, i, 2)]) + 1)
                feat_name = feat_prefix + '|pmi+2'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type - p_ngram
                feat_name = feat_prefix + '|p_fe+2'
                feat_seq[i][feat_name] = p_total + p_joint - p_ngram
                feat_name = feat_prefix + '|p_ef+2'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type

                p_ngram = math.log(sum([len(g['index'][get_ngram(tokens, i-1, 3)])
                                        for g in gazes.values()]) + 1)
                p_joint = math.log(len(gazes[entity_type]['index']
                                       [get_ngram(tokens, i - 1, 3)]) + 1)
                feat_name = feat_prefix + '|pmi_3'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type - p_ngram
                feat_name = feat_prefix + '|p_fe_3'
                feat_seq[i][feat_name] = p_total + p_joint - p_ngram
                feat_name = feat_prefix + '|p_ef_3'
                feat_seq[i][feat_name] = p_total + p_joint - p_entity_type

            return feat_seq

        gazetteers = resources[GAZETTEER_RSC]
        tokens = query.normalized_tokens
        feat_seq = [{} for _ in tokens]

        for entity_type in gazetteers:
            feats = get_ngram_gaz_features(query, gazetteers, entity_type)
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


def char_ngrams(n, word):
    char_gram = [''.join(ngram) for ngram in zip(*[word[i:] for i in range(n)])]
    return char_gram


def extract_char_ngrams_features(ngram_lengths_to_start_positions):
    """Returns a character n-gram feature extractor.
        Args:
            ngram_lengths_to_start_positions (dict):
            The window of tokens to be considered relative to the
            current token while extracting char n-grams
        Returns:
            (function) The feature extractor.
        """
    def _extractor(query, resources):
        tokens = query.normalized_tokens
        # normalize digits
        tokens = [re.sub(r'\d', '0', t) for t in tokens]
        feat_seq = [{} for _ in tokens]
        for i in range(len(tokens)):
            for length, starts in ngram_lengths_to_start_positions.items():
                for start in starts:
                    if i+int(start) < len(tokens):
                        ngrams = char_ngrams(int(length), tokens[i+int(start)])
                        for j, c_gram in enumerate(ngrams):
                            feat_name = 'char-ngrams|length:{}|pos:{}|sub-pos:{}'.format(
                                length, start, j)
                            feat_seq[i][feat_name] = c_gram
        return feat_seq
    return _extractor


@requires(SYS_TYPES_RSC)
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
        system_entities = query.get_system_entity_candidates(resources[SYS_TYPES_RSC])
        resolve_entity_conflicts([system_entities])
        for entity in system_entities:
            for i in entity.token_span:
                for j in start_positions:
                    if 0 < i-j < len(feat_seq):
                        feat_name = 'sys-candidate|type:{}:{}|pos:{}'.format(
                            entity.entity.type, entity.entity.value.get('grain'), j)
                        feat_seq[i-j][feat_name] = 1
                        feat_name = 'sys-candidate|type:{}:{}|pos:{}|log-len'.format(
                            entity.entity.type, entity.entity.value.get('grain'), j)
                        feat_seq[i-j][feat_name] = math.log(len(entity.normalized_text))
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


def extract_char_ngrams(lengths=(1,)):
    """
        Extract character ngrams of specified lengths.

        Args:
            lengths (list of int): The ngram length.

        Returns:
            (function) An feature extraction function that takes a query and
                returns character ngrams of specified lengths.
        """
    def _extractor(query, resources):
        query_text = query.normalized_text
        ngram_counter = Counter()
        for length in lengths:
            for i in range(len(query_text) - length + 1):
                char_ngram = []
                for token in query_text[i:i + length]:
                    char_ngram.append(token)
                ngram_counter.update(['char_ngram:' + '|'.join(char_ngram)])
        return ngram_counter
    return _extractor


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
        tokens = query.normalized_tokens
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
        tokens = query.normalized_tokens
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
        tokens = query.normalized_tokens
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
        tokens = query.normalized_tokens
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

        norm_text = query.normalized_text
        tokens = query.normalized_tokens
        ngrams = []
        for i in range(1, (len(tokens) + 1)):
            ngrams.extend(find_ngrams(tokens, i))
        for ngram in ngrams:
            for gaz_name, gaz in resources[GAZETTEER_RSC].items():
                if ngram in gaz['pop_dict']:
                    popularity = gaz['pop_dict'].get(ngram, 0.0)
                    ratio = (old_div(len(ngram), float(len(norm_text)))) * scaling
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
        tokens = len(query.normalized_tokens)
        chars = len(query.normalized_text)
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
        query_key = '<{}>'.format(query.normalized_text)
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


register_features('query', {
    'bag-of-words': extract_ngrams,
    'edge-ngrams': extract_edge_ngrams,
    'char-ngrams': extract_char_ngrams,
    'freq': extract_freq,
    'in-gaz': extract_in_gaz_feature,
    'gaz-freq': extract_gaz_freq,
    'length': extract_length,
    'exact': extract_query_string,
    'bag-of-words-seq': extract_bag_of_words_features,
    'in-gaz-span-seq': extract_in_gaz_span_features,
    'in-gaz-ngram-seq': extract_in_gaz_ngram_features,
    'sys-candidates-seq': extract_sys_candidate_features,
    'char-ngrams-seq': extract_char_ngrams_features
})
