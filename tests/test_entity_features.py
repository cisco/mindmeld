import pytest


@pytest.mark.parametrize(
    "query, feature_keys, expected_feature_values",
    [
        # Test for extract_bag_of_words_after_features
        ("change alarm from 8am to 9am",
         ['bag_of_words|ngram_after|length:1|pos:0',
          'bag_of_words|ngram_after|length:1|pos:1',
          'bag_of_words|ngram_after|length:2|pos:0',
          'bag_of_words|ngram_after|length:2|pos:1'],
         ['8am',
          'to',
          '8am to',
          'to 9am']),

        # Test for extract_bag_of_words_before_features
        ("change alarm from 8am to 9am",
         ['bag_of_words|ngram_before|length:1|pos:-1',
          'bag_of_words|ngram_before|length:1|pos:-2',
          'bag_of_words|ngram_before|length:2|pos:-1',
          'bag_of_words|ngram_before|length:2|pos:-2'],
         ['from',
          'alarm',
          'from 8am',
          'alarm from']),

        # Test for extract_other_entities_features
        ("change alarm from 8am to 9am",
         ['other_entities|type:sys_time'],
         [1]),

        # Test for extract_numeric_candidate_features
        ("change alarm from 8am to 9am",
         ['sys_candidate|type:sys_time|pos:3',
          'sys_candidate|type:sys_time|pos:5'],
         [1,
          1])
    ]
)
def test_entity_features(home_assistant_nlp, query, feature_keys, expected_feature_values):
    role_classifier_config_all_features = {
        'model_type': 'text',
        'model_settings': {
            'classifier_type': 'logreg'
        },
        'params': {
            'C': 100,
            'penalty': 'l1'
        },
        'features': {
            'in-gaz': {},
            'bag-of-words-before': {
                'ngram_lengths_to_start_positions': {
                    1: [-2, -1],
                    2: [-2, -1]
                }
            },
            'bag-of-words-after': {
                'ngram_lengths_to_start_positions': {
                    1: [0, 1],
                    2: [0, 1]
                }
            },
            'other-entities': {},
            'numeric': {},
        }
    }

    change_alarm_intent = home_assistant_nlp.domains['times_and_dates'].intents['change_alarm']

    entity_recognizer = change_alarm_intent.entity_recognizer
    role_classifier = change_alarm_intent.entities['sys_time'].role_classifier

    role_classifier.fit(**role_classifier_config_all_features)
    entities = entity_recognizer.predict(query)

    extracted_features = role_classifier.view_extracted_features(query, entities, 0)

    for feature_key, expected_value in zip(feature_keys, expected_feature_values):
        assert expected_value == extracted_features[feature_key]
