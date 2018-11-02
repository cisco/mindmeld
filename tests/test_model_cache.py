#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_model_cache
----------------------------------

Tests the generated model cache has to correct key value types.

"""
# pylint: disable=locally-disabled,redefined-outer-name
import shutil
import os

from mmworkbench.path import MODEL_CACHE_PATH
from mmworkbench.exceptions import FileNotFoundError
from mmworkbench.components import NaturalLanguageProcessor


def test_model_accuracies_are_similar_before_and_after_caching(kwik_e_mart_app_path):
    # clear model cache
    model_cache_path = MODEL_CACHE_PATH.format(app_path=kwik_e_mart_app_path)
    try:
        shutil.rmtree(MODEL_CACHE_PATH.format(app_path=kwik_e_mart_app_path))
    except FileNotFoundError:
        pass

    # Make sure no cache exists
    assert os.path.exists(model_cache_path) is False
    nlp = NaturalLanguageProcessor(kwik_e_mart_app_path)
    nlp.build(incremental=True)
    nlp.dump()

    intent_eval = nlp.domains['store_info'].intent_classifier.evaluate()
    entity_eval = nlp.domains['store_info'].intents['get_store_hours'].entity_recognizer.evaluate()
    intent_accuracy_no_cache = intent_eval.get_accuracy()
    entity_accuracy_no_cache = entity_eval.get_accuracy()

    example_cache = os.listdir(MODEL_CACHE_PATH.format(app_path=kwik_e_mart_app_path))[0]
    nlp = NaturalLanguageProcessor(kwik_e_mart_app_path)
    nlp.load(example_cache)

    # make sure cache exists
    assert os.path.exists(model_cache_path) is True

    intent_eval = nlp.domains['store_info'].intent_classifier.evaluate()
    entity_eval = nlp.domains['store_info'].intents['get_store_hours'].entity_recognizer.evaluate()
    intent_accuracy_cached = intent_eval.get_accuracy()
    entity_accuracy_cached = entity_eval.get_accuracy()

    assert intent_accuracy_no_cache == intent_accuracy_cached
    assert entity_accuracy_no_cache == entity_accuracy_cached
