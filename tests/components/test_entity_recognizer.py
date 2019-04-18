from mindmeld.components.nlp import NaturalLanguageProcessor
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


def test_tagger_get_stats(kwik_e_mart_app_path, capsys):
    nlp = NaturalLanguageProcessor(app_path=kwik_e_mart_app_path)
    er = nlp.domains['store_info'].intents['get_store_hours'].entity_recognizer
    er.fit()
    eval = er.evaluate()
    eval.print_stats()
    captured = capsys.readouterr()
    all_elems = set([k for k in captured.out.replace('\n', '').split(' ') if k != ''])
    assert 'Overall' in all_elems
    assert 'statistics:' in all_elems
    assert 'accuracy' in all_elems
    assert 'f1_weighted' in all_elems
    assert 'tp' in all_elems
    assert 'fp' in all_elems
    assert 'fn' in all_elems
