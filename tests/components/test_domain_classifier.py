from mmworkbench.components.nlp import NaturalLanguageProcessor
from unittest.mock import patch


def test_domain_classifier_random_forest(home_assistant_app_path):
    nlp = NaturalLanguageProcessor(app_path=home_assistant_app_path)
    dc = nlp.domain_classifier
    params = {'C': 10}
    with patch('logging.Logger.warning') as mock:
        dc.fit(model_settings={'classifier_type': 'rforest'}, params=params)
        mock.assert_any_call('Unexpected param `C`, dropping it from model config.')
