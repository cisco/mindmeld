#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test MindMeldALClassifier
----------------------------------

Tests for MindMeldALClassifier in the `active_learning.classifiers` module.
"""

import pytest
from mindmeld.active_learning.classifiers import MindMeldALClassifier
from mindmeld.active_learning.data_loading import DataBucketFactory
from mindmeld.active_learning.heuristics import (
    LeastConfidenceSampling,
    KLDivergenceSampling,
)
from mindmeld.constants import (
    DEFAULT_TRAIN_SET_REGEX,
    DEFAULT_TEST_SET_REGEX,
    TuneLevel,
)


@pytest.fixture(scope="module")
def mindmeld_al_classifier(kwik_e_mart_app_path):
    return MindMeldALClassifier(
        app_path=kwik_e_mart_app_path,
        tuning_level=[TuneLevel.INTENT.value],
        n_classifiers=3,
    )


@pytest.fixture(scope="module")
def mindmeld_al_classifier_domain(kwik_e_mart_app_path):
    return MindMeldALClassifier(
        app_path=kwik_e_mart_app_path,
        tuning_level=[TuneLevel.DOMAIN.value],
        n_classifiers=3,
    )


@pytest.fixture(scope="module")
def tuning_data_bucket(kwik_e_mart_app_path):
    return DataBucketFactory.get_data_bucket_for_strategy_tuning(
        app_path=kwik_e_mart_app_path,
        tuning_level=[TuneLevel.INTENT.value],
        train_pattern=DEFAULT_TRAIN_SET_REGEX,
        test_pattern=DEFAULT_TEST_SET_REGEX,
        train_seed_pct=0.2,
    )


@pytest.fixture(scope="module")
def tuning_data_bucket_domain(kwik_e_mart_app_path):
    return DataBucketFactory.get_data_bucket_for_strategy_tuning(
        app_path=kwik_e_mart_app_path,
        tuning_level=[TuneLevel.DOMAIN.value],
        train_pattern=DEFAULT_TRAIN_SET_REGEX,
        test_pattern=DEFAULT_TEST_SET_REGEX,
        train_seed_pct=0.2,
    )


def test_mindmeld_al_classifier_mappings(kwik_e_mart_nlp, mindmeld_al_classifier):
    intent2idx, idx2intent, domain_indices = mindmeld_al_classifier._get_mappings()
    # Test intent2idx and idx2intent
    for intent, idx in intent2idx.items():
        assert idx2intent[idx] == intent
    # Check length of domain_indices
    assert len(domain_indices.keys()) == len(list(kwik_e_mart_nlp.domains.keys()))


def test_validate_aggregate_statistic(mindmeld_al_classifier):
    # Intentionally incorrect statistic selected. Expected error.
    statistic = "mindmeld"
    with pytest.raises(ValueError):
        mindmeld_al_classifier._validate_aggregate_statistic(statistic)


def test_validate_class_level_statistic(mindmeld_al_classifier):
    # Intentionally incorrect statistic selected. Expected error.
    statistic = "mindmeld"
    with pytest.raises(ValueError):
        mindmeld_al_classifier._validate_class_level_statistic(statistic)


# Test single model classification, for example: LeastConfidenceSampling.
def test_single_model_classification(
    mindmeld_al_classifier_domain, tuning_data_bucket_domain
):

    (
        _,
        confidences_2d,
        confidences_3d,
        domain_indices,
    ) = mindmeld_al_classifier_domain.train(
        tuning_data_bucket_domain, LeastConfidenceSampling()
    )
    assert len(confidences_2d) == len(tuning_data_bucket_domain.unsampled_queries)
    assert confidences_3d is None
    assert domain_indices is None


# Test multi model classification, for example: KLDivergenceSampling.
def test_multi_model_classification(
    mindmeld_al_classifier_domain, tuning_data_bucket_domain
):

    (
        _,
        confidences_2d,
        confidences_3d,
        domain_indices,
    ) = mindmeld_al_classifier_domain.train(
        tuning_data_bucket_domain, KLDivergenceSampling()
    )
    assert len(confidences_2d) == len(tuning_data_bucket_domain.unsampled_queries)
    assert len(confidences_3d[0]) == len(tuning_data_bucket_domain.unsampled_queries)
    assert domain_indices is not None


# Intentional fail test, only single intent in domain.
def test_single_model_classification_intent(mindmeld_al_classifier, tuning_data_bucket):
    with pytest.raises(ValueError):
        (
            _,
            confidences_2d,
            confidences_3d,
            domain_indices,
        ) = mindmeld_al_classifier.train(tuning_data_bucket, LeastConfidenceSampling())
