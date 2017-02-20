import math
import re

import util

from mmworkbench import util as mmutil


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
        token = (util.OUT_OF_BOUNDS_TOKEN if index < 0 or index >= len(tokens)
                 else tokens[index])
        ngram_tokens.append(token)
    return ' '.join(ngram_tokens)


def extract_in_gaz_features():
    """Returns a feature extractor for properties of spans in gazetteers
    Deprecated, but needed to maintain current behavior in music-assistant
    """
    def extractor(query, resources):
        def get_span_features(query, start, end, ftype, entity):
            tokens = query.get_normalized_tokens()
            feat_seq = [{} for _ in tokens]
            for i in range(start, end):
                # Used to distinguish features for B tags vs I tags
                pos_attr = 'start' if i == start else 'continue'

                feat_name_prefix = 'in-gaz|type:{}|pos:{}'.format(
                    ftype, pos_attr)

                # Basic existence features
                feat_seq[i][feat_name_prefix] = 1
                # Features for ngram identity after the span
                feat_name = feat_name_prefix + '|ngram-after|length:{}'.format(
                    1)
                ('in-gaz|ngram-after|length:{}|type:{}|pos:{}'.format(
                    1, ftype, pos_attr))
                feat_seq[i][feat_name] = get_ngram(tokens, end, 1)
                feat_name = feat_name_prefix + '|ngram-after|length:{}'.format(
                    2)
                feat_seq[i][feat_name] = get_ngram(tokens, end, 2)
                # Popularity features
                pop = domain_gazes[ftype]['edict'][entity]
                feat_name = feat_name_prefix + '|pop'
                feat_seq[i][feat_name] = pop
                feat_name = feat_name_prefix + '|exp-pop'
                feat_seq[i][feat_name] = math.exp(pop)
                # Inverse document frequency
                if gaz['total_entities'] > 1:
                    feat_name = feat_name_prefix + '|idf'
                    feat_seq[i][feat_name] = (math.log(gaz['total_entities'] /
                                              (len(gaz['index'][ftype]) + 1)))
                # Character length features
                feat_name = feat_name_prefix + '|log-char-len'
                feat_seq[i][feat_name] = math.log(len(entity))
                feat_name = feat_name_prefix + '|pct-char-len'
                feat_seq[i][feat_name] = (float(len(entity)) /
                                          len(' '.join(tokens)))
            # End of span feature
            if end < len(tokens):
                feat_name = 'in-gaz|end|type:{}'.format(ftype)
                feat_seq[end][feat_name] = 1

            return feat_seq

        def get_exact_span_conflict_features(query, start, end, ftype_1,
                                             ftype_2, entity):
            feat_seq = [{} for _ in query.get_normalized_tokens()]
            for i in range(start, end):
                # Used to distinguish features for B tags vs I tags
                pos_attr = 'start' if i == 0 else 'continue'

                feat_name_prefix = (
                    'in-gaz|conflict|exact|type1:{}|type2:{}|pos:{}'
                    .format(ftype_1, ftype_2, pos_attr))

                feat_seq[i][feat_name_prefix] = 1

                pop_1 = domain_gazes[ftype_1]['edict'][entity]
                pop_2 = domain_gazes[ftype_2]['edict'][entity]
                feat_name = feat_name_prefix + 'diff-pop'
                feat_seq[i][feat_name] = pop_1 - pop_2
                feat_name = feat_name_prefix + 'exp-1-pop'
                feat_seq[i][feat_name] = math.exp(pop_1) - pop_2
                feat_name = feat_name_prefix + 'exp-2-pop'
                feat_seq[i][feat_name] = pop_1 - math.exp(pop_2)

            return feat_seq

        def get_same_start_span_conflict_features(query, start, end_1, end_2, ftype_1, ftype_2,
                                                  entity_1, entity_2):
            feat_seq = [{} for _ in query.get_normalized_tokens()]
            for i in range(start, min(end_1, end_2)):
                feat_name_prefix = 'in-gaz|conflict|same-start|'
                # Used to distinguish features for B tags vs I tags
                pos_attr = 'start' if i == 0 else 'continue'
                feat_name = (feat_name_prefix + 'type1:{}|type2:{}|pos:{}'
                             .format(ftype_1, ftype_2, pos_attr))
                feat_seq[i][feat_name] = 1

                pop_1 = domain_gazes[ftype_1]['edict'][entity_1]
                pop_2 = domain_gazes[ftype_2]['edict'][entity_2]
                feat_name = (feat_name_prefix +
                             'diff-pop|type1:{}|type2:{}|pos:{}'
                             .format(ftype_1, ftype_2, pos_attr))
                feat_seq[i][feat_name] = pop_1 - pop_2
                feat_name = (feat_name_prefix +
                             'exp-1-pop|type1:{}|type2:{}|pos:{}'
                             .format(ftype_1, ftype_2, pos_attr))
                feat_seq[i][feat_name] = math.exp(pop_1) - pop_2
                feat_name = (feat_name_prefix +
                             'exp-2-pop|type1:{}|type2:{}|pos:{}'
                             .format(ftype_1, ftype_2, pos_attr))
                feat_seq[i][feat_name] = pop_1 - math.exp(pop_2)
                feat_name = (feat_name_prefix +
                             'diff-len|type1:{}|type2:{}|pos:{}'
                             .format(ftype_1, ftype_2, pos_attr))
                feat_seq[i][feat_name] = len(entity_2) - len(entity_1)
                feat_name = (feat_name_prefix +
                             'diff-pct-len|type1:{}|type2:{}|pos:{}'
                             .format(ftype_1, ftype_2, pos_attr))
                feat_seq[i][feat_name] = ((len(entity_2) - len(entity_1)) /
                                          float(len(' '.join(query.get_normalized_tokens()))))

            return feat_seq

        domain_gazes = resources['gazetteers']
        tokens = query.get_normalized_tokens()
        feat_seq = [{} for _ in tokens]

        # Tuple of (start index, end index, ngram, facet type)
        in_gaz_spans = []

        for start in range(len(tokens)):
            for end in range(start+1, len(tokens)+1):
                for gaz_name, gaz in domain_gazes.items():
                    ngram = ' '.join(tokens[start:end])
                    if ngram in gaz['edict']:
                        in_gaz_spans.append((start, end, gaz_name, ngram))

        num_facets = query.get_candidate_numeric_facets(resources['num_types'])

        # Check ngrams with flattened numerics against the gazetteer
        # This algorithm iterates through each numeric facet and through
        # every ngram that includes the entire facet span
        for num_facet in num_facets:
            for gaz_name, gaz in domain_gazes.items():
                flattened_token = '@' + num_facet['type'] + '@'
                if flattened_token not in gaz['index']:
                    continue
                for start in range(num_facet['start']+1):
                    for end in range(num_facet['end']+1, len(tokens)+1):
                        ngram = ' '.join(
                            tokens[start:num_facet['start']] +
                            [flattened_token] +
                            tokens[num_facet['end']+1:end])
                        if ngram in gaz['edict']:
                            in_gaz_spans.append((start, end, gaz_name, ngram))
        # Sort the spans by their indices. The algorithm below assumes this
        # sort order.
        in_gaz_spans.sort()
        while in_gaz_spans:
            span = in_gaz_spans.pop(0)

            span_feat_seq = get_span_features(query, *span)
            util.update_features_sequence(feat_seq, span_feat_seq)

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
                            query, span[0], span[1], span[2], other_span[2],
                            span[3])
                        util.update_features_sequence(feat_seq, cmp_span_features)
                    # else:
                    #     cmp_span_features = get_same_start_span_conflict_features(
                    #         query, span[0], span[1], other_span[1], span[2],
                    #         other_span[2], span[3], other_span[3])
                    #     util.update_features_sequence(feat_seq, cmp_span_features)

        return feat_seq

    return extractor


def extract_in_gaz_span_features():
    """Returns a feature extractor for properties of spans in gazetteers
    """
    def extractor(query, resources):
        def get_span_features(query, gazes, start, end, ftype, entity):
            tokens = query.get_normalized_tokens()
            tokens = [re.sub('\d', '0', t) for t in tokens]
            feat_seq = [{} for _ in tokens]

            pop = gazes[ftype]['edict'][entity]
            p_total = math.log(sum([g['total_entities']
                                    for g in gazes.values()]) + 1) / 2

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
                feat_seq[i][feat_name] = (float(len(entity)) /
                                          len(' '.join(tokens)))

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
                feat_seq[i][feat_name] = (float(len(entity)) /
                                          len(' '.join(tokens)))

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
                feat_seq[end][feat_name] = float(len(entity)) / len(' '.join(tokens))
                feat_name = feat_prefix + '|pmi'
                feat_seq[end][feat_name] = p_total + p_joint - p_ftype - p_entity
                feat_name = feat_prefix + '|p_fe'
                feat_seq[end][feat_name] = p_total + p_joint - p_entity
                feat_name = feat_prefix + '|p_ef'
                feat_seq[end][feat_name] = p_total + p_joint - p_ftype

            return feat_seq

        def get_exact_span_conflict_features(query, gazes, start, end, ftype_1,
                                             ftype_2, entity):
            feat_seq = [{} for _ in query.get_normalized_tokens()]
            for i in range(start, end):

                feat_prefix = (
                    'in-gaz|conflict|exact|type1:{}|type2:{}'
                    .format(ftype_1, ftype_2))

                p_ftype_1 = math.log(gazes[ftype_1]['total_entities'] + 1)
                p_ftype_2 = math.log(gazes[ftype_2]['total_entities'] + 1)
                p_joint_1 = math.log(len(gazes[ftype_1]['index'][entity]) + 1)
                p_joint_2 = math.log(len(gazes[ftype_2]['index'][entity]) + 1)

                pop_1 = gazes[ftype_1]['edict'][entity]
                pop_2 = gazes[ftype_2]['edict'][entity]

                # Generic non-positional features
                feat_seq[i][feat_prefix] = 1
                feat_name = feat_prefix + '|diff-pop'
                feat_seq[i][feat_name] = pop_1 - pop_2
                feat_name = feat_prefix + '|diff-pmi'
                feat_seq[i][feat_name] = p_ftype_2 - p_ftype_1 - p_joint_2 + p_joint_1
                feat_name = feat_prefix + '|diff-p_fe'
                feat_seq[i][feat_name] = p_joint_1 - p_joint_2

            return feat_seq

        def get_gaz_spans(query, domain_gazes, num_types):
            """Collect tuples of (start index, end index, ngram, facet type)
            tracking ngrams that match with the entity gazetteer data
            """
            in_gaz_spans = []
            tokens = query.get_normalized_tokens()

            # Collect ngrams of plain normalized ngrams
            for start in range(len(tokens)):
                for end in range(start+1, len(tokens)+1):
                    for gaz_name, gaz in domain_gazes.items():
                        ngram = ' '.join(tokens[start:end])
                        if ngram in gaz['edict']:
                            in_gaz_spans.append((start, end, gaz_name, ngram))

            # Check ngrams with flattened numerics against the gazetteer
            # This algorithm iterates through each pair of numeric facets
            # and through every ngram that includes the entire facet span.
            # This limits regular facets to contain at most two numeric facets
            num_facets = query.get_candidate_numeric_facets(num_types)

            for gaz_name, gaz in domain_gazes.items():
                for i, num_facet_i in enumerate(num_facets):
                    if num_facet_i['type'] not in gaz['numtypes']:
                        continue
                    # logging.debug('Looking for [{}|num:{}] in {} gazetteer '
                    #               'with known numeric types {}'
                    #               .format(num_facet_i['entity'],
                    #                       num_facet_i['type'],
                    #                       gaz_name, list(gaz['numtypes'])))

                    # Collect ngrams that include all of num_facet_i
                    for start in range(num_facet_i['start']+1):
                        for end in range(num_facet_i['end']+1, len(tokens)+1):
                            ngram, ntoks = get_flattened_ngram(tokens, start, end, num_facet_i, 0)
                            if ngram in gaz['edict']:
                                in_gaz_spans.append((start, end, gaz_name, ngram))

                            # Check if we can fit any other num_facet_j between
                            # num_facet_i and the edge of the ngram
                            for j, num_facet_j in enumerate(num_facets[i+1:]):
                                if (num_facet_j['type'] in gaz['numtypes']
                                    and (start <= num_facet_j['start'])
                                    and (num_facet_j['end'] < end)
                                    and (num_facet_j['end'] < num_facet_i['start']
                                         or num_facet_i['end'] < num_facet_j['start'])):
                                    ngram, ntoks2 = get_flattened_ngram(
                                        ntoks, start, end, num_facet_j, start)
                                    if ngram in gaz['edict']:
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

        domain_gazes = resources['gazetteers']
        feat_seq = [{} for _ in query.get_normalized_tokens()]
        num_types = []
        for emap in resources['entity_maps']['entities']:
            if emap.get('numeric'):
                num_types.append(emap.get('numeric'))

        in_gaz_spans = get_gaz_spans(query, domain_gazes, num_types)

        # Sort the spans by their indices. The algorithm below assumes this
        # sort order.
        in_gaz_spans.sort()
        while in_gaz_spans:
            span = in_gaz_spans.pop(0)
            span_feat_seq = get_span_features(query, domain_gazes, *span)
            util.update_features_sequence(feat_seq, span_feat_seq)
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
                        util.update_features_sequence(feat_seq, cmp_span_features)

        return feat_seq

    return extractor


def extract_in_gaz_ngram_features():
    """Returns a feature extractor for surrounding ngrams in gazetteers
    """
    def extractor(query, resources):

        def get_ngram_gaz_features(query, gazes, ftype):
            tokens = query.get_normalized_tokens()
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
                p_total = math.log(sum([g['total_entities']
                                        for g in gazes.values()]) + 1) / 2
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
        tokens = query.get_normalized_tokens()
        feat_seq = [{} for _ in tokens]

        for ftype in domain_gazes:
            feats = get_ngram_gaz_features(query, domain_gazes, ftype)
            util.update_features_sequence(feat_seq, feats)

        return feat_seq

    return extractor


def extract_bag_of_words_features(ngram_lengths_to_start_positions):
    """Returns a bag-of-words feature extractor.
    Args:
        ngram_lengths_to_start_positions (dict):
    Returns:
        (function) The feature extractor.
    """
    def extractor(query, resources):
        tokens = query.get_normalized_tokens()
        tokens = [re.sub('\d', '0', t) for t in tokens]
        feat_seq = [{} for _ in tokens]
        for i in range(len(tokens)):
            for length, starts in ngram_lengths_to_start_positions.items():
                for start in starts:
                    feat_name = 'bag-of-words|length:{}|pos:{}'.format(
                        length, start)
                    feat_seq[i][feat_name] = get_ngram(tokens, i+int(start), int(length))
        return feat_seq

    return extractor


def extract_numeric_candidate_features(start_positions=(0,)):
    """Return an extractor for features based on a heuristic guess of numeric
    candidates at/near the current token.
    Args:
        start_positions (tuple): positions relative to current token (=0)
    Returns:
        (function) The feature extractor.
    """
    def extractor(query, resources):
        feat_seq = [{} for _ in query.get_normalized_tokens()]
        num_facets = query.get_candidate_numeric_facets(resources['num_types'])
        mmutil.resolve_conflicts([num_facets])
        for f in num_facets:
            for i in range(f['start'], f['end']+1):
                for j in start_positions:
                    if 0 < i-j < len(feat_seq):
                        feat_name = 'num-candidate|type:{}:{}|pos:{}'.format(
                            f['type'], f.get('grain'), j)
                        feat_seq[i-j][feat_name] = 1
                        feat_name = 'num-candidate|type:{}:{}|pos:{}|log-len'.format(
                            f['type'], f.get('grain'), j)
                        feat_seq[i-j][feat_name] = math.log(len(f['entity']))
        return feat_seq

    return extractor


FEATURE_NAME_MAP = {
    'bag-of-words': extract_bag_of_words_features,
    'in-gaz': extract_in_gaz_features,
    'in-gaz-span': extract_in_gaz_span_features,
    'in-gaz-ngram': extract_in_gaz_ngram_features,
    'num-candidates': extract_numeric_candidate_features
}
