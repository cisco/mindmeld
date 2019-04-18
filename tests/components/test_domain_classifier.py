from mindmeld.components.nlp import NaturalLanguageProcessor
from unittest.mock import patch


def test_domain_classifier_random_forest(home_assistant_app_path):
    nlp = NaturalLanguageProcessor(app_path=home_assistant_app_path)
    dc = nlp.domain_classifier
    params = {'C': 10}
    with patch('logging.Logger.warning') as mock:
        dc.fit(model_settings={'classifier_type': 'rforest'}, params=params)
        mock.assert_any_call('Unexpected param `C`, dropping it from model config.')


def test_domain_classifier_get_stats(home_assistant_app_path, capsys):
    nlp = NaturalLanguageProcessor(app_path=home_assistant_app_path)
    dc = nlp.domain_classifier
    dc.fit()
    eval = dc.evaluate()
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
