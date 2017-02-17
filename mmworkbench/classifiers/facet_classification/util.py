import itertools
import logging

from mmserver import mallard

START_TAG = 'START'
B_TAG = 'B'
I_TAG = 'I'
O_TAG = 'O'
E_TAG = 'E'
S_TAG = 'S'
OUT_OF_BOUNDS_TOKEN = '<$>'


def get_tags_from_facets(query, scheme='IOB'):
    """Get joint non-numeric and numeric IOB tags from a query's facets.

    Args:
        query (Query): An annotated query instance.

    Returns:
        (list of str): The tags for each token in the query. A tag has four
            parts separated by '|'. The first two are the IOB status for
            non-numeric facets followed by the type if non-numeric facet or
            '' if the IOB status is 'O'. The last two are like the first two,
            but for numeric facets.
    """
    iobs = [O_TAG for _ in query.get_normalized_tokens()]
    ftypes = ['' for _ in query.get_normalized_tokens()]
    num_iobs = [O_TAG for _ in query.get_normalized_tokens()]
    ntypes = ['' for _ in query.get_normalized_tokens()]

    # Regular facets #

    gold_facets = query.get_gold_facets()

    # tag I and type for all tag schemes
    for facet in gold_facets:
        for i in range(facet['start'], facet['end']+1):
            iobs[i] = I_TAG
            ftypes[i] = facet['type']

    # Replace I with B/E/S when appropriate
    if scheme in ('IOB', 'IOBES'):
        for facet in gold_facets:
            iobs[facet['start']] = B_TAG
    if scheme == 'IOBES':
        for facet in gold_facets:
            if facet['start'] == facet['end']:
                iobs[facet['end']] = S_TAG
            else:
                iobs[facet['end']] = E_TAG

    # Numerical Facets #
    # This algorithm assumes that the query numeric facets are well-formed and
    # only occur as standalone or fully inside a regular facet.

    gold_num_facets = query.get_gold_numeric_facets()

    # tag I and type for all tag schemes
    for facet in gold_num_facets:
        for i in range(facet['start'], facet['end']+1):
            num_iobs[i] = I_TAG
            ntypes[i] = facet['type']

    # Replace I with B/E/S when appropriate
    if scheme in ('IOB', 'IOBES'):
        for facet in gold_num_facets:
            num_iobs[facet['start']] = B_TAG
    if scheme == 'IOBES':
        for facet in gold_num_facets:
            if facet['start'] == facet['end']:
                num_iobs[facet['end']] = S_TAG
            else:
                num_iobs[facet['end']] = E_TAG

    tags = ['|'.join(args) for args in
            itertools.izip(iobs, ftypes, num_iobs, ntypes)]

    return tags


def get_facets_from_tags(query, tags):
    """From a set of joint IOB tags, parse the numeric and non-numeric facets.

    This performs the reverse operation of get_tags_from_facets.

    Args:
        query (Query): Any query instance.
        tags (list of str): Joint numeric and non-numeric tags, like those
            created by get_tags_from_facets.

    Returns:
        (list of dict, list of dict) The tuple containing the list of non-numeric
            facets and the list of numeric facets.
    """

    facets = []
    num_facets = []

    ntypes = set([tag.split('|')[3] for tag in tags])
    num_candidates = query.get_candidate_numeric_facets(set(ntypes))

    entity_tokens = []
    flat_entity_tokens = []
    facet_start = None
    prev_ftype = ''
    num_facet_start = None
    prev_ntype = ''

    def append_facet(start, end, ftype, entity_tokens, flat_entity_tokens):
        facet = {
            'start': start,
            'end': end - 1,
            'type': ftype,
            'chstart': query.get_chstart(start),
            'chend': query.get_chend(end),
            'tstart': start,
            'tend': end,
            'entity': ' '.join(entity_tokens),
            'flat-entity': ' '.join(flat_entity_tokens)
            }
        facets.append(facet)
        logging.debug("Appended {}".format(facet))

    def append_num_facet(start, end, ntype):
        logging.debug("Looking for '{}' between {} and {}".format(ntype, start, end))
        for num_candidate in num_candidates:
            if (num_candidate['start'] == start and
                    num_candidate['end'] == end - 1 and
                    num_candidate['type'] == ntype):
                num_facets.append(num_candidate)
                logging.debug("Appended numeric {}".format(num_candidate))
                return
        # If no corresponding numerical candidate was found, try calling Mallard
        # again.
        entity = ' '.join(query.get_normalized_tokens()[start:end])
        for raw_num_candidate in mallard.parse_numerics(entity)['data']:
            num_candidate = mallard.item_to_facet(raw_num_candidate,
                query.get_normalized_marked_down_query())
            # If there is numeric candidate matches the entire entity, then
            # fiddle with its indices.
            if (num_candidate['start'] == 0 and
                num_candidate['end'] == end - start - 1 and
                num_candidate['type'] == ntype):

                num_candidate['start'] = start
                num_candidate['end'] = end
                num_candidate['chstart'] = query.get_chstart(start)
                num_candidate['chend'] = query.get_chend(end)
                num_facets.append(num_candidate)
                return

            logging.debug("Did not append numeric {}".format(num_candidate))

    for tag_idx, tag in enumerate(tags):
        iob, ftype, num_iob, ntype = tag.split('|')

        # Close numeric facet and reset if the tag indicates a new facet
        if (num_facet_start is not None and
                (num_iob in (O_TAG, B_TAG, S_TAG) or ntype != prev_ntype)):
            logging.debug("Num facet closed at prev")
            append_num_facet(num_facet_start, tag_idx, prev_ntype)
            num_facet_start = None
            prev_ntype = ''

        # Close regular facet and reset if the tag indicates a new facet
        if (facet_start is not None and
                (iob in (O_TAG, B_TAG, S_TAG) or ftype != prev_ftype)):
            logging.debug("Facet closed at prev")
            append_facet(facet_start, tag_idx, prev_ftype,
                         entity_tokens, flat_entity_tokens)
            facet_start = None
            prev_ftype = ''
            entity_tokens = []
            flat_entity_tokens = []

            # Cut short any numeric facet that might continue beyond the facet
            if num_facet_start is not None:
                append_num_facet(num_facet_start, tag_idx, prev_ntype)
            num_facet_start = None
            prev_ntype = ''

        # Check if a regular facet has started
        if iob in (B_TAG, S_TAG) or ftype not in ('', prev_ftype):
            facet_start = tag_idx
        # Check if a numeric facet has started
        if num_iob in (B_TAG, S_TAG) or ntype not in ('', prev_ntype):
            num_facet_start = tag_idx

        # Append the current token to the current entity, if applicable.
        if iob != O_TAG and facet_start is not None:
            entity_tokens.append(query.get_normalized_tokens()[tag_idx])
            # Append any flattened numerics to the flat entity tokens.
            if num_iob in (B_TAG, S_TAG) or ntype not in ('', prev_ntype):
                flat_entity_tokens.append('@' + ntype + '@')
            elif num_iob == O_TAG:
                flat_entity_tokens.append(query.get_normalized_tokens()[tag_idx])
            else:
                logging.debug("Dropping this from flat-entity: {} {}"
                              .format(query.get_tokens()[tag_idx], tag))

        # Close the numeric facet if the tag indicates it closed
        if (num_facet_start is not None and
                num_iob in (E_TAG, S_TAG)):
            logging.debug("Num facet closed here")
            append_num_facet(num_facet_start, tag_idx+1, ntype)
            num_facet_start = None
            ntype = ''

        # Close the regular facet if the tag indicates it closed
        if (facet_start is not None and
                iob in (E_TAG, S_TAG)):
            logging.debug("Facet closed here")
            append_facet(facet_start, tag_idx+1, ftype,
                         entity_tokens, flat_entity_tokens)
            facet_start = None
            ftype = ''
            entity_tokens = []
            flat_entity_tokens = []

            # Cut short any numeric facet that might continue beyond the facet
            if num_facet_start is not None:
                append_num_facet(num_facet_start, tag_idx+1, ntype)
            num_facet_start = None
            ntype = ''

        prev_ftype = ftype
        prev_ntype = ntype

    # Handle facets that end with the end of the query
    if facet_start is not None:
        logging.debug("Facet closed at end: {}".format(entity_tokens))
        append_facet(facet_start, len(tags), prev_ftype,
                     entity_tokens, flat_entity_tokens)
    else:
        logging.debug("Facet did not end: {}".format(facet_start))
    if num_facet_start is not None:
        append_num_facet(num_facet_start, len(tags), prev_ntype)

    return facets, num_facets


def get_numeric_types(gazetteers):
    """From the gazetteers, get all the numeric facet types that exist.

    Args:
        (dict of Gazetteers): dictionary of gazetteers with numtypes property

    Returns:
        (set of str): A set of the numeric facet types in the input query list.
    """
    num_types = set()
    for g in gazetteers:
        num_types.update(gazetteers[g]['numtypes'])
    return num_types


def update_features_sequence(feat_seq, update_feat_seq):
    """Update a list of features with another parallel list of features.

    Args:
        feat_seq (list of dict): The original list of feature dicts which gets
            mutated.
        update_feat_seq (list of dict): The list of features to update with.
    """
    for i in range(len(feat_seq)): feat_seq[i].update(update_feat_seq[i])


