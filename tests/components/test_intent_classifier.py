from mindmeld.components.nlp import NaturalLanguageProcessor
from unittest.mock import patch


def test_intent_classifier_svm(kwik_e_mart_app_path):
    nlp = NaturalLanguageProcessor(app_path=kwik_e_mart_app_path)
    search_grid = {
       'C': [0.1, 0.5, 1, 5, 10, 50, 100, 1000, 5000],
       'kernel': ['linear', 'rbf', 'poly'],
    }

    param_selection_settings = {
        'grid': search_grid,
        'type': 'k-fold',
        'k': 10
    }
    ic = nlp.domains['store_info'].intent_classifier
    ic.fit(model_settings={'classifier_type': 'svm'}, param_selection=param_selection_settings)


def test_intent_classifier_logreg(kwik_e_mart_app_path):
    nlp = NaturalLanguageProcessor(app_path=kwik_e_mart_app_path)
    features = {
        'bag-of-words': {'lengths': [1]},
        'freq': {'bins': 5},
        'in-gaz': {},
        'length': {}
    }
    ic = nlp.domains['store_info'].intent_classifier
    ic.fit(model_settings={'classifier_type': 'logreg'},
           features=features)
    features = {
        'bag-of-words': {'lengths': [1, 2]},
        'freq': {'bins': 5},
        'in-gaz': {},
        'length': {}
    }
    ic.fit(model_settings={'classifier_type': 'logreg'},
           features=features)


def test_intent_classifier_random_forest(kwik_e_mart_app_path, caplog):
    nlp = NaturalLanguageProcessor(app_path=kwik_e_mart_app_path)
    search_grid = {
        'n_estimators': [5, 10, 15, 20],
        'criterion': ['gini', 'entropy'],
        'warm_start': [True, False],
    }

    param_selection_settings = {
        'grid': search_grid,
        'type': 'k-fold',
        'k': 10
    }

    ic = nlp.domains['store_info'].intent_classifier
    ic.fit(model_settings={'classifier_type': 'rforest'}, param_selection=param_selection_settings)

    ic.fit(model_settings={'classifier_type': 'rforest'},
           param_selection={'type': 'k-fold', 'k': 10, 'grid': {'class_bias': [0.7, 0.3, 0]}})

    features = {
        'bag-of-words': {'lengths': [1, 2]},
        'freq': {'bins': 5},
        'in-gaz': {},
        'length': {}
    }
    with patch('logging.Logger.warning') as mock:
        ic.fit(model_settings={'classifier_type': 'rforest'},
               features=features)
        mock.assert_any_call('Unexpected param `C`, dropping it from model config.')
        mock.assert_any_call('Unexpected param `fit_intercept`, dropping it from model config.')
