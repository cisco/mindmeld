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

"""This module contains feature extractors for queries"""
import math
import re
from collections import Counter, defaultdict

from .helpers import (
    CHAR_NGRAM_FREQ_RSC,
    DEFAULT_SYS_ENTITIES,
    ENABLE_STEMMING,
    GAZETTEER_RSC,
    OUT_OF_BOUNDS_TOKEN,
    OUT_OF_VOCABULARY,
    IN_VOCABULARY,
    QUERY_FREQ_RSC,
    SYS_TYPES_RSC,
    WORD_FREQ_RSC,
    WORD_NGRAM_FREQ_RSC,
    SENTIMENT_ANALYZER,
    get_ngram,
    mask_numerics,
    register_query_feature,
    requires,
    get_ngrams_upto_n,
)

from ..gazetteer import NestedGazetteer


@register_query_feature(feature_name="in-gaz-span-seq")
@requires(GAZETTEER_RSC)
def extract_in_gaz_span_features(**kwargs):
    """Returns a feature extractor for properties of spans in gazetteers"""
    del kwargs

    def _extractor(query, resources):
        def _get_span_features(query, all_gazes, current_gaz):
            tokens = [re.sub(r"\d", "0", t) for t in query.normalized_tokens]
            feature_sequence = [{} for _ in tokens]

            pop = all_gazes[current_gaz.gaz_name]["pop_dict"][current_gaz.token_ngram]
            p_total = (
                math.log(sum([g["total_entities"] for g in all_gazes.values()]) + 1) / 2
            )
            p_entity_type = math.log(all_gazes[current_gaz.gaz_name]["total_entities"] + 1)
            p_entity = math.log(
                sum([len(g["index"][current_gaz.raw_ngram]) for g in all_gazes.values()]) + 1
            )
            p_joint = math.log(len(
                all_gazes[current_gaz.gaz_name]["index"][current_gaz.raw_ngram]) + 1)

            for i in range(current_gaz.start_token_index, current_gaz.end_token_index_plus_one):
                # Generic non-positional features
                gaz_feat_prefix = "in_gaz|type:{}".format(current_gaz.gaz_name)

                # Basic existence features
                feature_sequence[i][gaz_feat_prefix] = 1

                # Used to distinguish among B/I/E/S tags
                if i == current_gaz.start_token_index:
                    pos_attr = "start"
                elif i == current_gaz.end_token_index_plus_one - 1:
                    pos_attr = "end"
                else:
                    pos_attr = "cont"

                # Basic existence features
                positional_gaz_prefix = "in_gaz|type:{}|segment:{}".format(
                    current_gaz.gaz_name, pos_attr
                )

                # Basic Positional features
                feature_sequence[i][positional_gaz_prefix] = 1

                features = {
                    # Features for ngram before the span
                    "|ngram_before|length:{}".format(1): get_ngram(
                        tokens, current_gaz.start_token_index - 1, 1
                    ),
                    # Features for ngram after the span
                    "|ngram_after|length:{}".format(1): get_ngram(
                        tokens, current_gaz.end_token_index_plus_one, 1),
                    # Features for ngram at start of span
                    "|ngram_first|length:{}".format(1): get_ngram(
                        tokens, current_gaz.start_token_index, 1),
                    # Features for ngram at end of span
                    "|ngram_last|length:{}".format(1): get_ngram(
                        tokens, current_gaz.end_token_index_plus_one - 1, 1),
                    # Popularity features
                    "|pop": pop,
                    # Character length features
                    "|log_char_len": math.log(len(current_gaz.raw_ngram)),
                    "|pct_char_len": len(current_gaz.raw_ngram) / len(" ".join(tokens)),
                    # entity PMI and conditional prob
                    "|pmi": p_total + p_joint - p_entity_type - p_entity,
                    "|class_prob": p_total + p_joint - p_entity,
                    "|output_prob": p_total + p_joint - p_entity_type,
                }

                for key, value in features.items():
                    for prefix in [gaz_feat_prefix, positional_gaz_prefix]:
                        feature_sequence[i][prefix + key] = value

            # End of span feature
            if current_gaz.end_token_index_plus_one < len(tokens):
                feat_prefix = "in-gaz|type:{}|signal_entity_end".format(current_gaz.gaz_name)
                feature_sequence[current_gaz.end_token_index_plus_one][feat_prefix] = 1

                span_features = {
                    "|log_char_len": math.log(len(current_gaz.raw_ngram)),
                    "|pct_char_len": len(current_gaz.raw_ngram) / len(" ".join(tokens)),
                    "|pmi": p_total + p_joint - p_entity_type - p_entity,
                    "|class_prob": p_total + p_joint - p_entity,
                    "|output_prob": p_total + p_joint - p_entity_type,
                }

                for key, value in span_features.items():
                    feature_sequence[current_gaz.end_token_index_plus_one][
                        feat_prefix + key] = value

            return feature_sequence

        def get_exact_span_conflict_features(query, gazes, nested_gaz, other_nested_gaz):
            feature_sequence = [{} for _ in query.normalized_tokens]
            for i in range(nested_gaz.start_token_index, nested_gaz.end_token_index_plus_one):
                feat_prefix = "in-gaz|conflict:exact|type1:{}|type2:{}".format(
                    nested_gaz.gaz_name, other_nested_gaz.gaz_name
                )

                p_ent_type_1 = math.log(gazes[nested_gaz.gaz_name]["total_entities"] + 1)
                p_ent_type_2 = math.log(gazes[other_nested_gaz.gaz_name]["total_entities"] + 1)
                p_joint_1 = math.log(
                    len(gazes[nested_gaz.gaz_name]["index"][nested_gaz.raw_ngram]) + 1)
                p_joint_2 = math.log(
                    len(gazes[other_nested_gaz.gaz_name]["index"][nested_gaz.raw_ngram]) + 1)

                pop_1 = gazes[nested_gaz.gaz_name]["pop_dict"][nested_gaz.token_ngram]
                pop_2 = gazes[other_nested_gaz.gaz_name]["pop_dict"][nested_gaz.token_ngram]

                # Generic non-positional features
                feature_sequence[i][feat_prefix] = 1

                features = {
                    "|diff_pop": pop_1 - pop_2,
                    "|diff_pmi": p_ent_type_2 - p_ent_type_1 - p_joint_2 + p_joint_1,
                    "|diff_class_prob": p_joint_1 - p_joint_2,
                }

                for key, value in features.items():
                    feature_sequence[i][feat_prefix + key] = value

            return feature_sequence

        def get_gaz_spans(query, gazetteers):
            """Collect tuples of (start index, end index, ngram, entity type)
            tracking ngrams that match with the entity gazetteer data
            """
            nested_gazes = []
            tokens = query.get_verbose_normalized_tokens()
            # Collect ngrams of plain normalized ngrams
            for start_index, _ in enumerate(tokens):
                for end_index_plus_one in range(start_index + 1, len(tokens) + 1):
                    for gaz_name, gaz in gazetteers.items():
                        token_ngram, raw_ngram, _ = query.get_token_ngram_raw_ngram_span(
                            tokens, start_index, end_index_plus_one - 1)
                        if token_ngram and token_ngram in gaz["pop_dict"]:
                            nested_gazes.append(NestedGazetteer(
                                start_index, end_index_plus_one,
                                gaz_name, token_ngram, raw_ngram))
            return nested_gazes

        gazetteers = resources[GAZETTEER_RSC]
        feat_seq = [{} for _ in query.normalized_tokens]
        in_gaz_spans = get_gaz_spans(query, gazetteers)

        # Sort the spans by their indices. The algorithm below assumes this
        # sort order.
        in_gaz_spans.sort()
        while in_gaz_spans:
            nested_gaz = in_gaz_spans.pop(0)
            span_feat_seq = _get_span_features(query, gazetteers, nested_gaz)
            update_features_sequence(feat_seq, span_feat_seq)

            for other_nested_gaz in in_gaz_spans:
                if other_nested_gaz.start_token_index >= nested_gaz.end_token_index_plus_one:
                    break
                # For now, if two spans of the same type start at the same
                # place, take the longer one.
                if other_nested_gaz.start_token_index == nested_gaz.start_token_index and \
                        other_nested_gaz.gaz_name == nested_gaz.gaz_name:
                    continue
                if nested_gaz.start_token_index == other_nested_gaz.start_token_index:
                    if nested_gaz.end_token_index_plus_one == \
                            other_nested_gaz.end_token_index_plus_one:
                        cmp_span_features = get_exact_span_conflict_features(
                            query,
                            gazetteers,
                            nested_gaz,
                            other_nested_gaz,
                        )
                        update_features_sequence(feat_seq, cmp_span_features)
        return feat_seq

    return _extractor


@register_query_feature(feature_name="in-gaz-ngram-seq")
@requires(GAZETTEER_RSC)
def extract_in_gaz_ngram_features(**kwargs):
    """Returns a feature extractor for surrounding ngrams in gazetteers"""
    del kwargs

    def _extractor(query, resources):
        def get_ngram_gaz_features(query, gazes, entity_type):
            tokens = query.normalized_tokens
            feat_seq = [{} for _ in tokens]

            for i, _ in enumerate(feat_seq):
                feat_prefix = "in_gaz|type:{}|ngram".format(entity_type)

                # entity PMI and conditional prob
                p_total = (
                    math.log(sum([g["total_entities"] for g in gazes.values()]) + 1) / 2
                )
                p_entity_type = math.log(gazes[entity_type]["total_entities"] + 1)

                features = {
                    "|length:{}|pos:{}|idf".format(1, 0): math.log(
                        len(gazes[entity_type]["index"][get_ngram(tokens, i, 1)]) + 1
                    ),
                    "|length:{}|pos:{}|idf".format(2, -1): math.log(
                        len(gazes[entity_type]["index"][get_ngram(tokens, i - 1, 2)])
                        + 1
                    ),
                    "|length:{}|pos:{}|idf".format(2, 1): math.log(
                        len(gazes[entity_type]["index"][get_ngram(tokens, i, 2)]) + 1
                    ),
                }

                for key, value in features.items():
                    feat_seq[i][feat_prefix + key] = value

                # these features are extracted on a window span around the current token
                window_features = [
                    {
                        "length": 1,
                        "position": 0,
                        "p_ngram": math.log(
                            sum(
                                [
                                    len(g["index"][get_ngram(tokens, i, 1)])
                                    for g in gazes.values()
                                ]
                            )
                            + 1
                        ),
                        "p_joint": math.log(
                            len(gazes[entity_type]["index"][get_ngram(tokens, i, 1)])
                            + 1
                        ),
                    },
                    {
                        "length": 2,
                        "position": -1,
                        "p_ngram": math.log(
                            sum(
                                [
                                    len(g["index"][get_ngram(tokens, i - 1, 2)])
                                    for g in gazes.values()
                                ]
                            )
                            + 1
                        ),
                        "p_joint": math.log(
                            len(
                                gazes[entity_type]["index"][get_ngram(tokens, i - 1, 2)]
                            )
                            + 1
                        ),
                    },
                    {
                        "length": 2,
                        "position": 1,
                        "p_ngram": math.log(
                            sum(
                                [
                                    len(g["index"][get_ngram(tokens, i, 2)])
                                    for g in gazes.values()
                                ]
                            )
                            + 1
                        ),
                        "p_joint": math.log(
                            len(gazes[entity_type]["index"][get_ngram(tokens, i, 2)])
                            + 1
                        ),
                    },
                    {
                        "length": 3,
                        "position": 0,
                        "p_ngram": math.log(
                            sum(
                                [
                                    len(g["index"][get_ngram(tokens, i - 1, 3)])
                                    for g in gazes.values()
                                ]
                            )
                            + 1
                        ),
                        "p_joint": math.log(
                            len(
                                gazes[entity_type]["index"][get_ngram(tokens, i - 1, 3)]
                            )
                            + 1
                        ),
                    },
                ]

                for window_feature in window_features:
                    features = {
                        "|length:{}|pos:{}|pmi".format(
                            window_feature["length"], window_feature["position"]
                        ): p_total
                        + window_feature["p_joint"]
                        - p_entity_type
                        - window_feature["p_ngram"],
                        "|length:{}|pos:{}|class_prob".format(
                            window_feature["length"], window_feature["position"]
                        ): p_total
                        + window_feature["p_joint"]
                        - window_feature["p_ngram"],
                        "|length:{}|pos:{}|output_prob".format(
                            window_feature["length"], window_feature["position"]
                        ): p_total
                        + window_feature["p_ngram"]
                        - p_entity_type,
                    }

                    for key, value in features.items():
                        feat_seq[i][feat_prefix + key] = value
            return feat_seq

        gazetteers = resources[GAZETTEER_RSC]
        tokens = query.normalized_tokens
        feat_seq = [{} for _ in tokens]

        for entity_type in gazetteers:
            feats = get_ngram_gaz_features(query, gazetteers, entity_type)
            update_features_sequence(feat_seq, feats)

        return feat_seq

    return _extractor


@register_query_feature(feature_name="bag-of-words-seq")
@requires(WORD_NGRAM_FREQ_RSC)
def extract_bag_of_words_features(
    ngram_lengths_to_start_positions, thresholds=(1,), **kwargs
):
    """Returns a bag-of-words feature extractor.

    Args:
        ngram_lengths_to_start_positions (dict)
        thresholds (int): Cut off value to include word in n-gram vocab

    Returns:
        (function) The feature extractor.
    """
    threshold_list = list(thresholds)
    word_thresholds = threshold_list + [1] * (
        len(ngram_lengths_to_start_positions.keys()) - len(threshold_list)
    )

    def remove_excess_out_of_bounds(n_gram: str) -> str:
        """
        In our frequency dictionaries, we only add a single OOB token to the start
        and end of a query. This function takes care to remove excess OOB tokens and
        retains only 1, so we can look up their counts and uphold the thresholding feature.
        Eg., '<$> <$> what' --> '<$> what'
             '<$> what' --> '<$> what'
             'help <$> <$>' --> 'help <$>'
             '<$> <$>' --> '<$>'
        """
        shortened_ngram = n_gram
        if OUT_OF_BOUNDS_TOKEN in n_gram:
            oob_last = n_gram.rfind(OUT_OF_BOUNDS_TOKEN)
            oob_first = n_gram.find(OUT_OF_BOUNDS_TOKEN)
            if oob_last + len(OUT_OF_BOUNDS_TOKEN) == len(n_gram):
                shortened_ngram = n_gram[:oob_first + len(OUT_OF_BOUNDS_TOKEN)]
            else:
                shortened_ngram = n_gram[oob_last:]
        return shortened_ngram

    def _extractor(query, resources):
        tokens = query.normalized_tokens
        tokens = [re.sub(r"\d", "0", t) for t in tokens]
        feat_seq = [{} for _ in tokens]

        if kwargs.get(ENABLE_STEMMING, False):
            stemmed_tokens = query.stemmed_tokens
            stemmed_tokens = [re.sub(r"\d", "0", t) for t in stemmed_tokens]

        for i in range(len(tokens)):
            threshold_index = 0
            for length, starts in ngram_lengths_to_start_positions.items():
                threshold = word_thresholds[threshold_index]
                for start in starts:
                    n_gram = get_ngram(tokens, i + int(start), int(length))
                    feat_name = "bag_of_words|length:{}|word_pos:{}".format(
                        length, start
                    )
                    short_ngram = remove_excess_out_of_bounds(n_gram)
                    if resources[WORD_NGRAM_FREQ_RSC].get(short_ngram, 0) >= threshold:
                        feat_seq[i][feat_name] = n_gram
                    else:
                        feat_seq[i][feat_name] = OUT_OF_VOCABULARY

                    if kwargs.get(ENABLE_STEMMING, False):
                        stemmed_n_gram = get_ngram(
                            stemmed_tokens, i + int(start), int(length)
                        )
                        short_stemmed_ngram = remove_excess_out_of_bounds(stemmed_n_gram)
                        stemmed_feat_name = (
                            "bag_of_words_stemmed|length:{}|word_pos:{}".format(
                                length, start
                            )
                        )
                        if (
                            resources[WORD_NGRAM_FREQ_RSC].get(short_stemmed_ngram, 0)
                            >= threshold
                        ):
                            feat_seq[i][stemmed_feat_name] = stemmed_n_gram
                        else:
                            feat_seq[i][stemmed_feat_name] = OUT_OF_VOCABULARY

                threshold_index += 1
        return feat_seq

    return _extractor


def char_ngrams(n, word, **kwargs):
    """This function extracts character ngrams for the given word

    Args:
        n (int): Max size of n-gram to extract
        word (str): The word to be extract n-grams from

    Returns:
        list: A list of character n-grams for the given word
    """
    del kwargs
    char_grams = []
    for i in range(len(word)):
        # if char ngram of length n doesn't exist, if no ngrams have been extracted for the token,
        # add token to the list and return. No need to compute for other windows.
        # Ex: token is "you", n=4, return ["you"], token is "doing", n=4 return ["doin","oing"]
        if len(word[i : i + n]) < n:
            if not char_grams:
                char_grams.append((word[i : i + n]))
            return char_grams
        char_grams.append((word[i : i + n]))
    return char_grams


@register_query_feature(feature_name="enable-stemming")
@requires(ENABLE_STEMMING)
def enabled_stemming(**kwargs):
    """Feature extractor for enabling stemming of the query"""
    del kwargs

    def _extractor(query, resources):
        # no op
        del query
        del resources

    return _extractor


@register_query_feature(feature_name="char-ngrams-seq")
@requires(CHAR_NGRAM_FREQ_RSC)
def extract_char_ngrams_features(
    ngram_lengths_to_start_positions, thresholds=(1,), **kwargs
):
    """Returns a character n-gram feature extractor.

    Args:
        ngram_lengths_to_start_positions (dict):
        The window of tokens to be considered relative to the
        current token while extracting char n-grams
        thresholds (int): Cut off value to include word in n-gram vocab

    Returns:
        (function) The feature extractor.
    """
    del kwargs
    threshold_list = list(thresholds)
    char_thresholds = threshold_list + [1] * (
        len(ngram_lengths_to_start_positions.keys()) - len(threshold_list)
    )

    def _extractor(query, resources):
        tokens = query.normalized_tokens
        # normalize digits
        tokens = [re.sub(r"\d", "0", t) for t in tokens]
        feat_seq = [{} for _ in tokens]

        for i in range(len(tokens)):
            threshold_index = 0
            for length, starts in ngram_lengths_to_start_positions.items():
                threshold = char_thresholds[threshold_index]
                for start in starts:
                    token_index = i + int(start)
                    if 0 <= token_index < len(tokens):
                        ngrams = char_ngrams(length, tokens[token_index])
                    else:
                        # if token index out of bounds, return OUT_OF_BOUNDS token
                        ngrams = [OUT_OF_BOUNDS_TOKEN]
                    for j, c_gram in enumerate(ngrams):
                        feat_name = (
                            "char_ngrams|length:{}|word_pos:{}|char_pos:{}".format(
                                length, start, j
                            )
                        )
                        if resources[CHAR_NGRAM_FREQ_RSC].get(c_gram, 0) < threshold:
                            c_gram = OUT_OF_VOCABULARY
                        feat_seq[i][feat_name] = c_gram
                threshold_index += 1
        return feat_seq

    return _extractor


@register_query_feature(feature_name="sys-candidates-seq")
@requires(SYS_TYPES_RSC)
def extract_sys_candidate_features(start_positions=(0,), **kwargs):
    """Return an extractor for features based on a heuristic guess of numeric
    candidates at/near the current token.

    Args:
        start_positions (tuple): positions relative to current token (=0)

    Returns:
        (function) The feature extractor.
    """
    del kwargs

    def _extractor(query, resources):
        feat_seq = [{} for _ in query.normalized_tokens]
        system_entities = query.get_system_entity_candidates(resources[SYS_TYPES_RSC])
        for entity in system_entities:
            for i in entity.normalized_token_span:
                for j in start_positions:
                    if 0 <= i - j < len(feat_seq):
                        feat_name = (
                            "sys_candidate|type:{}|granularity:{}|pos:{}".format(
                                entity.entity.type, entity.entity.value.get("grain"), j
                            )
                        )
                        feat_seq[i - j][feat_name] = feat_seq[i - j].get(feat_name, 0) + 1
                        feat_name = "sys_candidate|type:{}|granularity:{}|pos:{}|log_len".format(
                            entity.entity.type, entity.entity.value.get("grain"), j
                        )
                        feat_value = feat_seq[i - j][feat_name] = feat_seq[i - j].get(feat_name, [])
                        feat_value.append(len(entity.normalized_text))

        for token_features in feat_seq:
            for feature, value in token_features.items():
                if feature.endswith('log_len'):
                    token_features[feature] = math.log((float(sum(value)) / len(value)) + 1)
                else:
                    # Adjust value to be greater than 0
                    token_features[feature] = math.log(value + 1)
        return feat_seq

    return _extractor


def update_features_sequence(feat_seq, update_feat_seq, **kwargs):
    """Update a list of features with another parallel list of features.

    Args:
        feat_seq (list of dict): The original list of feature dicts which gets
            mutated.
        update_feat_seq (list of dict): The list of features to update with.
    """
    del kwargs
    for i, feat_seq_i in enumerate(feat_seq):
        feat_seq_i.update(update_feat_seq[i])


@register_query_feature(feature_name="char-ngrams")
@requires(CHAR_NGRAM_FREQ_RSC)
def extract_char_ngrams(lengths=(1,), thresholds=(1,), **kwargs):
    """Extract character ngrams of specified lengths.

    Args:
        lengths (list of int): The ngram length.
        thresholds (list of int): frequency cut off value to include ngram in vocab

    Returns:
        (function) An feature extraction function that takes a query and
            returns character ngrams of specified lengths.
    """
    del kwargs
    threshold_list = list(thresholds)
    char_thresholds = threshold_list + [1] * (len(lengths) - len(threshold_list))

    def _extractor(query, resources):

        query_text = re.sub(r"\d", "0", query.normalized_text)

        ngram_counter = Counter()
        for length, threshold in zip(lengths, char_thresholds):
            for i in range(len(query_text) - length + 1):
                char_ngram = []
                for token in query_text[i : i + length]:
                    char_ngram.append(token)

                joined_char_ngram = "".join(char_ngram)
                freq = resources[CHAR_NGRAM_FREQ_RSC].get(joined_char_ngram, 0)
                if freq < threshold:
                    joined_char_ngram = OUT_OF_VOCABULARY
                ngram_counter.update(
                    [
                        "char_ngram|length:{}|ngram:{}".format(
                            len(char_ngram), joined_char_ngram
                        )
                    ]
                )
        return ngram_counter

    return _extractor


@register_query_feature(feature_name="bag-of-words")
@requires(WORD_NGRAM_FREQ_RSC)
def extract_ngrams(lengths=(1,), thresholds=(1,), **kwargs):
    """
    Extract ngrams of some specified lengths.

    Args:
        lengths (list of int): The ngram length.
        thresholds (list of int): frequency cut off value to include ngram in vocab

    Returns:
        (function) An feature extraction function that takes a query and \
            returns ngrams of the specified lengths.
    """
    threshold_list = list(thresholds)
    word_thresholds = threshold_list + [1] * (len(lengths) - len(threshold_list))

    def _extractor(query, resources):
        tokens = query.normalized_tokens
        stemmed_tokens = query.stemmed_tokens
        ngram_counter = Counter()

        for length, threshold in zip(lengths, word_thresholds):
            for i in range(len(tokens) - length + 1):
                ngram = []
                stemmed_ngram = []

                for index in range(i, i + length):
                    # We never want to differentiate between number tokens.
                    # We may need to convert number words too, like "eighty".
                    token = tokens[index]
                    tok = re.sub('\d','0',token)
                    ngram.append(tok)

                    if kwargs.get(ENABLE_STEMMING, False):
                        tok_stemmed = re.sub('\d','0',stemmed_tokens[index])
                        stemmed_ngram.append(tok_stemmed)

                joined_ngram = " ".join(ngram)
                freq = resources[WORD_NGRAM_FREQ_RSC].get(joined_ngram, 0)
                if freq < threshold:
                    joined_ngram = OUT_OF_VOCABULARY
                ngram_counter.update(
                    [
                        "bag_of_words|length:{}|ngram:{}".format(len(ngram), joined_ngram)
                    ]
                )
                if kwargs.get(ENABLE_STEMMING, False):
                    joined_stemmed_ngram = " ".join(stemmed_ngram)
                    freq = resources[WORD_NGRAM_FREQ_RSC].get(joined_stemmed_ngram, 0)
                    if freq < threshold:
                        joined_stemmed_ngram = OUT_OF_VOCABULARY
                    ngram_counter.update(
                        [
                            "bag_of_words_stemmed|length:{}|ngram:{}".format(
                                len(stemmed_ngram), joined_stemmed_ngram)
                        ]
                    )

        return ngram_counter

    return _extractor


@register_query_feature(feature_name="sys-candidates")
def extract_sys_candidates(entities=None, **kwargs):
    """
    Return an extractor for features based on a heuristic guess of numeric \
        candidates in the current query.

    Returns:
            (function) The feature extractor.
     """
    del kwargs
    entities = entities or DEFAULT_SYS_ENTITIES

    def _extractor(query, resources):
        del resources
        system_entities = query.get_system_entity_candidates(list(entities))
        sys_ent_counter = Counter()
        for entity in system_entities:
            sys_ent_counter.update(["sys_candidate|type:{}".format(entity.entity.type)])
            sys_ent_counter.update(
                [
                    "sys_candidate|type:{}|granularity:{}".format(
                        entity.entity.type, entity.entity.value.get("grain")
                    )
                ]
            )
        return sys_ent_counter

    return _extractor


@register_query_feature(feature_name="word-shape")
def extract_word_shape(lengths=(1,), **kwargs):
    """
    Extracts word shape for ngrams of specified lengths.

    Args:
        lengths (list of int): The ngram length

    Returns:
        (function) An feature extraction function that takes a query and \
            returns ngrams of word shapes, for n of specified lengths.
    """
    del kwargs

    def word_shape_basic(token):
        # example: option --> xxxxx+, 123 ---> ddd, call --> xxxx
        shape = ["d" if character.isdigit() else "x" for character in token]
        if len(shape) > 5:
            if all(x == "d" for x in shape):
                return "ddddd+"
            elif all(x == "x" for x in shape):
                return "xxxxx+"
        return "".join(shape)

    def _extractor(query, resources):
        del resources
        tokens = query.normalized_tokens
        shape_counter = Counter()
        for length in lengths:
            for i in range(len(tokens) - length + 1):
                word_shapes = []
                for token in tokens[i : i + length]:
                    # We can incorporate different kinds of shapes in the future (capitalization)
                    tok = word_shape_basic(token)
                    word_shapes.append(tok)
                shape_counter.update(
                    [
                        "bag_of_words|length:{}|word_shape:{}".format(
                            len(word_shapes), " ".join(word_shapes)
                        )
                    ]
                )
        q_len = float(len(tokens))
        for entry in shape_counter:
            shape_counter[entry] = math.log(shape_counter[entry] + 1, 2) / q_len
        return shape_counter

    return _extractor


@register_query_feature(feature_name="edge-ngrams")
@requires(WORD_FREQ_RSC)
def extract_edge_ngrams(lengths=(1,), **kwargs):
    """
    Extract ngrams of some specified lengths.

    Args:
        lengths (list of int): The ngram length.

    Returns:
        (function) An feature extraction function that takes a query and \
            returns ngrams of the specified lengths at start and end of query.
    """
    del kwargs

    def _extractor(query, resources):
        tokens = query.normalized_tokens
        feats = {}

        for length in lengths:
            if length <= len(tokens):
                left_tokens = [mask_numerics(tok) for tok in tokens[:length]]
                left_tokens = [
                    tok if resources[WORD_FREQ_RSC].get(tok, 0) > 1 else OUT_OF_VOCABULARY
                    for tok in left_tokens
                ]
                right_tokens = [mask_numerics(tok) for tok in tokens[-length:]]
                right_tokens = [
                    tok if resources[WORD_FREQ_RSC].get(tok, 0) > 1 else OUT_OF_VOCABULARY
                    for tok in right_tokens
                ]
                feats.update(
                    {
                        "bag_of_words|edge:left|length:{}|ngram:{}".format(
                            length, " ".join(left_tokens)
                        ): 1
                    }
                )
                feats.update(
                    {
                        "bag_of_words|edge:right|length:{}|ngram:{}".format(
                            length, " ".join(right_tokens)
                        ): 1
                    }
                )
        return feats

    return _extractor


@register_query_feature(feature_name="freq")
@requires(WORD_FREQ_RSC)
def extract_freq(bins=5, **kwargs):
    """
    Extract frequency bin features.

    Args:
        bins (int): The number of frequency bins (besides OOV)

    Returns:
        (function): A feature extraction function that returns the log of the \
            count of query tokens within each frequency bin.

    """

    def _extractor(query, resources):
        tokens = query.normalized_tokens
        stemmed_tokens = query.stemmed_tokens

        freq_dict = resources[WORD_FREQ_RSC]
        max_freq = freq_dict.most_common(1)[0][1]
        freq_features = defaultdict(int)

        for idx, tok in enumerate(tokens):
            tok = mask_numerics(tok)

            if kwargs.get(ENABLE_STEMMING, False):
                stemmed_tok = stemmed_tokens[idx]
                stemmed_tok = mask_numerics(stemmed_tok)
                freq = freq_dict.get(tok, freq_dict.get(stemmed_tok, 0))
            else:
                freq = freq_dict.get(tok, 0)

            if freq < 2:
                freq_features["in_vocab:OOV"] += 1
            else:
                # Bin the frequency with break points at
                # half max, a quarter max, an eighth max, etc.
                freq_bin = int(math.log(max_freq, 2) - math.log(freq, 2))
                if freq_bin < bins:
                    freq_features["in_vocab:IV|freq_bin:{}".format(freq_bin)] += 1
                else:
                    freq_features["in_vocab:IV|freq_bin:{}".format(bins)] += 1

        q_len = float(len(tokens))
        for k in freq_features:
            # sublinear
            freq_features[k] = math.log(freq_features[k] + 1, 2)
            # ratio
            freq_features[k] /= q_len
        return freq_features

    return _extractor


@register_query_feature(feature_name="gaz-freq")
@requires(GAZETTEER_RSC)
@requires(WORD_FREQ_RSC)
def extract_gaz_freq(**kwargs):
    """
    Extract frequency bin features for each gazetteer

    Returns:
        (function): A feature extraction function that returns the log of the \
            count of query tokens within each gazetteer's frequency bins.
    """
    del kwargs

    def _extractor(query, resources):
        tokens = query.normalized_tokens
        freq_features = defaultdict(int)

        for tok in tokens:
            query_freq = OUT_OF_VOCABULARY if resources[WORD_FREQ_RSC].get(tok) is None \
                else IN_VOCABULARY
            for gaz_name, gaz in resources[GAZETTEER_RSC].items():
                freq = len(gaz["index"].get(tok, []))
                if freq > 0:
                    freq_bin = int(math.log(freq, 2) / 2)
                    freq_features[
                        "in_gaz|type:{}|gaz_freq_bin:{}".format(gaz_name, freq_bin)
                    ] += 1
                    freq_features[
                        "in_vocab:{}|in_gaz|type:{}|gaz_freq_bin:{}".format(
                            query_freq, gaz_name, freq_bin
                        )
                    ] += 1

        q_len = float(len(tokens))
        for k in freq_features:
            # sublinear
            freq_features[k] = math.log(freq_features[k] + 1, 2)
            # ratio
            freq_features[k] /= q_len
        return freq_features

    return _extractor


@register_query_feature(feature_name="in-gaz")
@requires(GAZETTEER_RSC)
def extract_in_gaz_feature(scaling=1, **kwargs):
    """Returns a feature extractor that generates a set of features indicating the presence
    of query n-grams in different entity gazetteers. Used by the domain and intent classifiers
    when the 'in-gaz' feature is specified in the config.

    Args:
        scaling (int): A multiplicative scale factor to the ``ratio_pop`` and ``ratio`` features of
        the in-gaz feature set.

    Returns:
        function: Returns an extractor function
    """
    del kwargs

    def _extractor(query, resources):
        in_gaz_features = defaultdict(float)

        norm_text = query.normalized_text
        tokens = query.normalized_tokens
        verbose_tokens = query.get_verbose_normalized_tokens()
        ngrams = get_ngrams_upto_n(tokens, len(tokens))
        for ngram, token_span in ngrams:
            for gaz_name, gaz in resources[GAZETTEER_RSC].items():
                if ngram in gaz["pop_dict"]:
                    popularity = gaz["pop_dict"].get(ngram, 0.0)
                    _, raw_ngram, _ = query.get_token_ngram_raw_ngram_span(
                        verbose_tokens, token_span[0], token_span[1])
                    ratio = len(raw_ngram) / len(norm_text) * scaling
                    ratio_pop = ratio * popularity
                    in_gaz_features[
                        "in_gaz|type:{}|ratio_pop".format(gaz_name)
                    ] += ratio_pop
                    in_gaz_features["in_gaz|type:{}|ratio".format(gaz_name)] += ratio
                    in_gaz_features["in_gaz|type:{}|pop".format(gaz_name)] += popularity
                    in_gaz_features["in_gaz|type:{}".format(gaz_name)] = 1

        return in_gaz_features

    return _extractor


@register_query_feature(feature_name="length")
def extract_length(**kwargs):
    """
    Extract length measures (tokens and chars; linear and log) on whole query.

    Returns:
        (function) A feature extraction function that takes a query and \
            returns number of tokens and characters on linear and log scales
    """
    del kwargs

    def _extractor(query, resources):
        del resources
        tokens = len(query.normalized_tokens)
        chars = len(query.normalized_text)
        return {
            "tokens": tokens,
            "chars": chars,
            "tokens_log": math.log(tokens + 1),
            "chars_log": math.log(chars + 1),
        }

    return _extractor


@register_query_feature(feature_name="exact")
@requires(QUERY_FREQ_RSC)
def extract_query_string(scaling=1000, **kwargs):
    """
    Extract whole query string as a feature.

    Returns:
        (function) A feature extraction function that takes a query and \
            returns the whole query string for exact matching

    """

    def _extractor(query, resources):
        query_key = "<{}>".format(query.normalized_text)
        if query_key in resources[QUERY_FREQ_RSC]:
            return {"exact|query:{}".format(query_key): scaling}

        if kwargs.get(ENABLE_STEMMING, False):
            stemmed_query_key = "<{}>".format(query.stemmed_text)
            if stemmed_query_key in resources[QUERY_FREQ_RSC]:
                return {"exact|query:{}".format(stemmed_query_key): scaling}

        return {"exact|query:{}".format("<OOV>"): scaling}

    return _extractor


@register_query_feature(feature_name="sentiment")
@requires(SENTIMENT_ANALYZER)
def extract_sentiment(analyzer="composite", **kwargs):
    """Generates sentiment intensity scores for each query

    Returns:
        (function) A feature extraction function that takes in a query and \
            returns sentiment values across positive, negative and neutral

    """
    del kwargs

    def _extractor(query, resources):
        text = query.text
        sentiment_scores = resources[SENTIMENT_ANALYZER].polarity_scores(text)
        if analyzer == "composite":
            return {"sentiment|composite": sentiment_scores["compound"]}
        else:
            return {
                "sentiment|positive": sentiment_scores["pos"],
                "sentiment|negative": sentiment_scores["neg"],
                "sentiment|neutral": sentiment_scores["neu"],
            }

    return _extractor


def find_ngrams(input_list, n, **kwargs):
    """Generates all n-gram combinations from a list of strings

    Args:
        input_list (list): List of string to n-gramize
        n (int): The size of the n-gram

    Returns:
        list: A list of ngrams across all the strings in the \
            input list
    """
    del kwargs
    result = []
    for ngram in zip(*[input_list[i:] for i in range(n)]):
        result.append(" ".join(ngram))
    return result
