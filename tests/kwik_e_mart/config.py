APP_NAME = 'kwik_e_mart'

INTENT_MODEL_CONFIG = {
    'model_type': 'text',
    'model_settings': {
        'classifier_type': 'logreg'
    },
    'param_selection': {
        'type': 'k-fold',
        'k': 5,
        'grid': {
            'fit_intercept': [True, False],
            'C': [1, 20, 300],
            'class_bias': [1, 0]
        }
    },
    'features': {
        'bag-of-words': {
            'lengths': [1]
        },
        'in-gaz': {},
        'freq': {'bins': 5},
        'length': {}
    }
}