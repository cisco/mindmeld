from mindmeld.models.helpers import register_query_feature, register_entity_feature


@register_query_feature(feature_name='average-token-length')
def extract_average_token_length(**args):
    """
    Example query feature that gets the average length of normalized tokens in the query

    Returns:
        (function) A feature extraction function that takes a query and
            returns the average normalized token length
    """
    def _extractor(query, resources):
        tokens = query.normalized_tokens
        average_token_length = sum([len(t) for t in tokens]) / len(tokens)
        return {'average_token_length': average_token_length}

    return _extractor


@register_entity_feature(feature_name='entity-span-start')
def extract_entity_span_start(**args):
    """
    Example entity feature that gets the start span for each entity

    Returns:
        (function) A feature extraction function that returns the start span of the entity
    """
    def _extractor(example, resources):
        query, entities, entity_index = example
        features = {}

        current_entity = entities[entity_index]
        current_entity_token_start = current_entity.token_span.start

        features['entity_span_start'] = current_entity_token_start
        return features

    return _extractor
