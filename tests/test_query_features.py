import pytest
import math

EXACT_QUERY_MATCH_SCALING_FACTOR = 10
EPSILON = math.pow(10, -5)


@pytest.mark.parametrize(
    "query, feature_keys, expected_feature_values, dynamic_resource",
    [
        # Test for extract_in_gaz_feature
        ("set alarm potato",
         ['in_gaz|type:duration'],
         [1],
         {'gazetteers': {'duration': {"potato": 1000.0}}}),

        # Test for extract_gaz_freq
        ("set alarm potato",
         ['in_gaz|type:duration|gaz_freq_bin:0'],
         [math.log(2, 2) / 3],
         {'gazetteers': {'duration': {"potato": 1000.0}}}),

        # Test for extract_freq
        ("set temperature potato",
         ['in_vocab:IV|freq_bin:1', 'in_vocab:OOV'],
         [math.log(3, 2) / 3, math.log(2, 2) / 3],
         None),

        # Test for extract_edge_ngrams
        ("set temperature to 60",
         ['bag_of_words|edge:left|length:1|ngram:set',
          'bag_of_words|edge:left|length:2|ngram:set temperature',
          'bag_of_words|edge:right|length:1|ngram:#NUM',
          'bag_of_words|edge:right|length:2|ngram:to #NUM'],
         [1, 1, 1, 1],
         None),

        # Test for extract_sys_candidates
        ("set temperature to 60",
         ['sys_candidate|type:sys_temperature', 'sys_candidate|type:sys_number'],
         [2, 1],
         None),

        # Test for extract_char_ngrams
        ("set the",
         ['char_ngram|length:1|ngram:e', 'char_ngram|length:1|ngram:t',
          'char_ngram|length:2|ngram:s e', 'char_ngram|length:2|ngram:e t'],
         [2, 2, 1, 1],
         None),

        # Test for extract_word_shape
        ("set the thermostat",
         ['bag_of_words|length:1|word_shape:xxx', 'bag_of_words|length:1|word_shape:xxxxx+'],
         [math.log(3, 2) / 3, math.log(2, 2) / 3],
         None),

        # Test for extract_ngrams
        ("set thermostat",
         ['bag_of_words|length:1|ngram:set', 'bag_of_words|length:1|ngram:thermostat',
          'bag_of_words|length:2|ngram:set thermostat'],
         [1, 1, 1],
         None),

        # Test for extract_length
        ("set the thermostat",
         ["tokens", "chars"],
         [3, 18],
         None),

        # Test for extract_query_string
        # This query occurs multiple times in training set so the value is the scaling factor
        ("turn the thermostat up to 70",
         ["exact|query:<turn the thermostat up to 70>"],
         [EXACT_QUERY_MATCH_SCALING_FACTOR],
         None),

        # Test for enabled_stemming
        # Make sure the stemmed words appear in bag of words
        ("setting the thermostat",
         ["bag_of_words_stemmed|length:1|ngram:set"],
         [1],
         None),
    ]
)
def test_domain_query_features(home_assistant_nlp, query, feature_keys, expected_feature_values,
                               dynamic_resource):
    """
    Test to make sure query level text model features work as expected
    Args:
        home_assistant_nlp: nlp object for the home_assistant blueprint
        query: text query to use
        feature_keys: list of keys in the feature dictionary
        expected_feature_values: the expected values corresponding to each key
        dynamic_resource: gazetteer given to nlp object
    """
    assert len(feature_keys) == len(expected_feature_values)
    domain_classifier_config_all_features = {
        'model_type': 'text',
        'model_settings': {
            'classifier_type': 'logreg'
        },
        'params': {
            'C': 10,
        },
        'features': {
            "char-ngrams": {
                "lengths": [1, 2],
                "thresholds": [0]
            },
            "bag-of-words": {
                "lengths": [1, 2]
            },
            "sys-candidates": {},
            "word-shape": {},
            "edge-ngrams": {"lengths": [1, 2]},
            "freq": {"bins": 5},
            "gaz-freq": {},
            "in-gaz": {},
            "length": {},
            "exact": {"scaling": EXACT_QUERY_MATCH_SCALING_FACTOR},
            "enable-stemming": True,
        }
    }

    domain_classifier = home_assistant_nlp.domain_classifier
    domain_classifier.fit(**domain_classifier_config_all_features)

    extracted_features = domain_classifier.view_extracted_features(
        query, dynamic_resource=dynamic_resource)
    for feature_key, expected_value in zip(feature_keys, expected_feature_values):

        if isinstance(expected_value, float):
            assert abs(expected_value - extracted_features[feature_key]) < EPSILON
        else:
            assert expected_value == extracted_features[feature_key]


@pytest.mark.parametrize(
    "query, feature_keys, expected_feature_values, index",
    [
        # Test for extract_in_gaz_ngram_features
        ("change alarm from 8am to 9am",
         ['in_gaz|type:duration|ngram|length:1|pos:0|class_prob',
          'in_gaz|type:duration|ngram|length:1|pos:0|idf',
          'in_gaz|type:duration|ngram|length:1|pos:0|output_prob',
          'in_gaz|type:duration|ngram|length:1|pos:0|pmi'],
         [math.log(2 + 1) / 2,
          0,
          math.log(2 + 1) / 2 - math.log(2),
          math.log(2 + 1) / 2 - math.log(2)],
         0),

        # Test for extract_bag_of_words_features
        ("change alarm from 8am to 9am",
         ['bag_of_words|length:1|word_pos:0',
          'bag_of_words|length:1|word_pos:1',
          'bag_of_words|length:2|word_pos:0'],
         ['change',
          'alarm',
          'change alarm'],
         0),

        # Test for extract_char_ngrams_features
        ("change alarm from 8am to 9am",
         ['char_ngrams|length:1|word_pos:0|char_pos:0',
          'char_ngrams|length:2|word_pos:0|char_pos:0'],
         ['c',
          'ch'],
         0),

        # Test for extract_sys_candidates
        ("change alarm from 8am to 9am",
         ['sys_candidate|type:sys_interval|granularity:hour|pos:0|log_len',
          'sys_candidate|type:sys_time|granularity:hour|pos:0|log_len'],
         [math.log(10),
          math.log(3)],
         -1),
    ]
)
def test_entity_query_features(home_assistant_nlp, query, feature_keys, expected_feature_values,
                               index):
    """
    Test to make sure tagger model features work as expected
    Args:
        home_assistant_nlp: nlp object for the home_assistant blueprint
        query: text query to use
        feature_keys: list of keys in the feature dictionary
        expected_feature_values: the expected values corresponding to each key
        index: the index of the token to examine
    """
    assert len(feature_keys) == len(expected_feature_values)
    entity_recognizer_config_all_features = {
        'model_type': 'tagger',
        'model_settings': {
            'classifier_type': 'memm', 'tag_scheme': 'IOB', 'feature_scaler': 'max-abs'
        },
        'params': {
            'C': 10,
            'penalty': 'l2'
        },
        'features': {
            "in-gaz-span-seq": {},
            "in-gaz-ngram-seq": {},
            "bag-of-words-seq": {
                'ngram_lengths_to_start_positions': {1: [-1, 0, 1], 2: [-1, 0, 1]},
                "thresholds": [0]
            },
            "char-ngrams-seq": {
                'ngram_lengths_to_start_positions': {1: [-1, 0, 1], 2: [-1, 0, 1]},
                "thresholds": [0]
            },
            "sys-candidates-seq": {
                'start_positions': [0]
            },
        }
    }

    entity_recognizer = \
        home_assistant_nlp.domains['times_and_dates'].intents['change_alarm'].entity_recognizer
    entity_recognizer.fit(**entity_recognizer_config_all_features)

    extracted_features = entity_recognizer.view_extracted_features(query)[index]

    for feature_key, expected_value in zip(feature_keys, expected_feature_values):

        if isinstance(expected_value, float):
            assert abs(expected_value - extracted_features[feature_key]) < EPSILON
        else:
            assert expected_value == extracted_features[feature_key]


@pytest.mark.parametrize(
    "query, feature_keys, expected_feature_values, index",
    [
        ("When will one on 23 Elm Street open?",
         ['in_gaz|type:store_name',
          'in_gaz|type:store_name|ngram_before|length:1',
          'in_gaz|type:store_name|ngram_after|length:1',
          'in_gaz|type:store_name|ngram_first|length:1',
          'in_gaz|type:store_name|ngram_last|length:1',
          'in_gaz|type:store_name|pop',
          'in_gaz|type:store_name|log_char_len',
          'in_gaz|type:store_name|pct_char_len',
          'in_gaz|type:store_name|pmi',
          'in_gaz|type:store_name|class_prob',
          'in_gaz|type:store_name|output_prob',
          'in_gaz|type:store_name|segment:start',
          'in_gaz|type:store_name|segment:start|ngram_before|length:1',
          'in_gaz|type:store_name|segment:start|ngram_after|length:1',
          'in_gaz|type:store_name|segment:start|ngram_first|length:1',
          'in_gaz|type:store_name|segment:start|ngram_last|length:1',
          'in_gaz|type:store_name|segment:start|pop',
          'in_gaz|type:store_name|segment:start|log_char_len',
          'in_gaz|type:store_name|segment:start|pct_char_len',
          'in_gaz|type:store_name|segment:start|pmi',
          'in_gaz|type:store_name|segment:start|class_prob',
          'in_gaz|type:store_name|segment:start|output_prob',
          'in_gaz|type:store_name|ngram|length:1|pos:0|idf',
          'in_gaz|type:store_name|ngram|length:2|pos:-1|idf',
          'in_gaz|type:store_name|ngram|length:2|pos:1|idf',
          'in_gaz|type:store_name|ngram|length:1|pos:0|pmi',
          'in_gaz|type:store_name|ngram|length:1|pos:0|class_prob',
          'in_gaz|type:store_name|ngram|length:1|pos:0|output_prob',
          'in_gaz|type:store_name|ngram|length:2|pos:-1|pmi',
          'in_gaz|type:store_name|ngram|length:2|pos:-1|class_prob',
          'in_gaz|type:store_name|ngram|length:2|pos:-1|output_prob',
          'in_gaz|type:store_name|ngram|length:2|pos:1|pmi',
          'in_gaz|type:store_name|ngram|length:2|pos:1|class_prob',
          'in_gaz|type:store_name|ngram|length:2|pos:1|output_prob',
          'in_gaz|type:store_name|ngram|length:3|pos:0|pmi',
          'in_gaz|type:store_name|ngram|length:3|pos:0|class_prob',
          'in_gaz|type:store_name|ngram|length:3|pos:0|output_prob',
          'bag_of_words|length:1|word_pos:-1',
          'bag_of_words|length:1|word_pos:0',
          'bag_of_words|length:1|word_pos:1',
          'bag_of_words|length:2|word_pos:-1',
          'bag_of_words|length:2|word_pos:0',
          'bag_of_words|length:2|word_pos:1',
          'char_ngrams|length:1|word_pos:-1|char_pos:0',
          'char_ngrams|length:1|word_pos:-1|char_pos:1',
          'char_ngrams|length:1|word_pos:0|char_pos:0',
          'char_ngrams|length:1|word_pos:0|char_pos:1',
          'char_ngrams|length:1|word_pos:1|char_pos:0',
          'char_ngrams|length:1|word_pos:1|char_pos:1',
          'char_ngrams|length:1|word_pos:1|char_pos:2',
          'char_ngrams|length:2|word_pos:-1|char_pos:0',
          'char_ngrams|length:2|word_pos:0|char_pos:0',
          'char_ngrams|length:2|word_pos:1|char_pos:0',
          'char_ngrams|length:2|word_pos:1|char_pos:1',
          'sys_candidate|type:sys_time|granularity:hour|pos:0',
          'sys_candidate|type:sys_time|granularity:hour|pos:0|log_len'],
         [1,
          'on',
          'open',
          '00',
          'street',
          1.0,
          2.5649493574615367,
          0.37142857142857144,
          -2.5852419975190757,
          2.5852419975190757,
          -2.5852419975190757,
          1,
          'on',
          'open',
          '00',
          'street',
          1.0,
          2.5649493574615367,
          0.37142857142857144,
          -2.5852419975190757,
          2.5852419975190757,
          -2.5852419975190757,
          1.3862943611198906,
          0.0,
          0.0,
          -2.585241997519076,
          2.5852419975190752,
          -1.1989476363991853,
          -2.5852419975190757,
          2.5852419975190757,
          -2.5852419975190757,
          -2.5852419975190757,
          2.5852419975190757,
          -2.5852419975190757,
          -2.5852419975190757,
          2.5852419975190757,
          -2.5852419975190757,
          'on',
          '00',
          'elm',
          'on 00',
          '00 elm',
          'elm street',
          'o',
          'n',
          '0',
          '0',
          'e',
          'l',
          'm',
          'on',
          '00',
          'el',
          'lm',
          1,
          0.6931471805599453],
         4),
    ]
)
def test_entity_gaz_query_features(kwik_e_mart_nlp, query, feature_keys, expected_feature_values,
                                   index):
    """
    Test to make sure tagger model gazetteer features work as expected
    Args:
        kwik_e_mart_nlp: nlp object for the kwik e mart blueprint
        query: text query to use
        feature_keys: list of keys in the feature dictionary
        expected_feature_values: the expected values corresponding to each key
        index: the index of the token to examine
    """
    entity_recognizer_config_all_features = {
        'model_type': 'tagger',
        'model_settings': {
            'classifier_type': 'memm', 'tag_scheme': 'IOB', 'feature_scaler': 'max-abs'
        },
        'params': {
            'C': 10,
            'penalty': 'l2'
        },
        'features': {
            "in-gaz-span-seq": {},
            "in-gaz-ngram-seq": {},
            "bag-of-words-seq": {
                'ngram_lengths_to_start_positions': {1: [-1, 0, 1], 2: [-1, 0, 1]},
                "thresholds": [0]
            },
            "char-ngrams-seq": {
                'ngram_lengths_to_start_positions': {1: [-1, 0, 1], 2: [-1, 0, 1]},
                "thresholds": [0]
            },
            "sys-candidates-seq": {
                'start_positions': [0]
            },
        }
    }

    entity_recognizer = \
        kwik_e_mart_nlp.domains['store_info'].intents['get_store_hours'].entity_recognizer
    entity_recognizer.fit(**entity_recognizer_config_all_features)
    extracted_features = entity_recognizer.view_extracted_features(query)[index]
    for feature_key, expected_value in zip(feature_keys, expected_feature_values):
        if isinstance(expected_value, float):
            assert abs(expected_value - extracted_features[feature_key]) < EPSILON
        else:
            assert expected_value == extracted_features[feature_key]
    entity_recognizer.fit()
