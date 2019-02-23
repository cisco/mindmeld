from mmworkbench.components.nlp import NaturalLanguageProcessor
from unittest.mock import patch


def test_memm_model(kwik_e_mart_app_path):
    nlp = NaturalLanguageProcessor(app_path=kwik_e_mart_app_path)
    er = nlp.domains['store_info'].intents['get_store_hours'].entity_recognizer
    params = {
        'type': 'k-fold',
        'k': 5,
        'scoring': 'accuracy',
        'grid': {
            'penalty': ['l1', 'l2'],
            'C': [0.01, 1, 100, 10000, 1000000, 100000000]
        },
    }
    with patch('logging.Logger.warning') as mock:
        er.fit(param_selection=params)
        assert 'C' in er.config.param_selection['grid']
        assert 'penalty' in er.config.param_selection['grid']
        mock.assert_not_called()
