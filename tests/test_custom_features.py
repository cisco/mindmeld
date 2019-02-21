from mmworkbench.components.nlp import NaturalLanguageProcessor


def test_custom_features_loaded(home_assistant_app_path):
    nlp = NaturalLanguageProcessor(app_path=home_assistant_app_path)
    dc = nlp.domain_classifier
    dc.fit()
    assert 'average-token-length' in dc.config.features
    example = nlp.create_query('set the temperature')
    feature_set = dc._model._extract_features(example)
    assert 'average_token_length' in feature_set
