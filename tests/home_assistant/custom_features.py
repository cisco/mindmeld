from mmworkbench.models.helpers import register_query_feature


@register_query_feature(feature_name='average-token-length')
def extract_average_token_length(**args):
    """
    Example query feature that gets the average length of normalized tokens in the query„

    Returns:
        (function) A feature extraction function that takes a query and
            returns the average normalized token length
    """
    # pylint: disable=locally-disabled,unused-argument
    def _extractor(query, resources):
        tokens = query.normalized_tokens
        average_token_length = sum([len(t) for t in tokens]) / len(tokens)
        return {'average_token_length': average_token_length}

    return _extractor
