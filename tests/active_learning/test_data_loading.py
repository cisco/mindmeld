#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test Data_Loading
----------------------------------

Tests for Data_Loading in the `active_learning.data_loading` module.
"""
import pytest
from mindmeld.active_learning.data_loading import (
    LabelMap,
    LogQueriesLoader,
    DataBucket,
    DataBucketFactory,
)
from mindmeld.constants import (
    DEFAULT_TRAIN_SET_REGEX,
    DEFAULT_TEST_SET_REGEX,
    TuneLevel,
)
from mindmeld.core import ProcessedQuery


@pytest.fixture(scope="module")
def all_train_queries(kwik_e_mart_nlp):
    return kwik_e_mart_nlp.resource_loader.get_flattened_label_set(
        label_set=DEFAULT_TRAIN_SET_REGEX
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


# Test Label Map domain to id bidirectional mapping
def test_label_map_domains(kwik_e_mart_nlp):
    kwik_e_mart_query_tree = kwik_e_mart_nlp.resource_loader.get_labeled_queries()
    label_map = LabelMap(kwik_e_mart_query_tree)
    for domain, idx in label_map.domain2id.items():
        assert label_map.id2domain[idx] == domain


# Test Label Map intent to id bidirectional mapping
def test_label_map_intents(kwik_e_mart_nlp):
    kwik_e_mart_query_tree = kwik_e_mart_nlp.resource_loader.get_labeled_queries()
    label_map = LabelMap(kwik_e_mart_query_tree)
    for domain in label_map.domain_to_intent2id:
        intent2id = label_map.domain_to_intent2id[domain]
        for intent, idx in intent2id.items():
            assert label_map.id2intent[domain][idx] == intent


# Test the Domain class label for all queries
def test_get_class_labels_domains(kwik_e_mart_nlp, all_train_queries):
    app_domains = list(kwik_e_mart_nlp.domains.keys())
    unique_domain_labels = list(
        set(LabelMap.get_class_labels(TuneLevel.DOMAIN.value, all_train_queries))
    )
    assert all(domain in app_domains for domain in unique_domain_labels)


# Test the Domain-Intent class label for all queries
def test_get_class_labels_domains_intents(kwik_e_mart_nlp, all_train_queries):
    nlp_domain_intent_labels = []
    for domain in kwik_e_mart_nlp.domains:
        for intent in kwik_e_mart_nlp.domains[domain].intents:
            label = f"{domain}.{intent}"
            nlp_domain_intent_labels.append(label)
    unique_domain_intent_labels = list(
        set(LabelMap.get_class_labels(TuneLevel.INTENT.value, all_train_queries))
    )
    assert all(
        label in nlp_domain_intent_labels for label in unique_domain_intent_labels
    )


# Test the LogQueriesLoader class
def test_log_queries_loader(kwik_e_mart_nlp, kwik_e_mart_app_path):
    first_ten_queries = [
        kwik_e_mart_nlp.resource_loader.query_cache.get(i) for i in range(1, 10)
    ]
    first_ten_queries_raw = [q.query.text for q in first_ten_queries]
    log_queries_loader = LogQueriesLoader(
        app_path=kwik_e_mart_app_path,
        tuning_level=TuneLevel.INTENT.value,
        log_file_path="",
    )
    text_to_processed_queries = log_queries_loader.convert_text_queries_to_processed(
        first_ten_queries_raw
    )
    for q in text_to_processed_queries:
        assert isinstance(q, ProcessedQuery)
        assert q.domain is not None
        assert q.intent is not None


# Test the DataBucketFactory and DataBucket
def test_data_bucket_factory(kwik_e_mart_app_path, tuning_data_bucket):
    len_sampled_queries = len(tuning_data_bucket.sampled_queries)
    len_unsampled_queries = len(tuning_data_bucket.unsampled_queries)

    # Simulating sampling by simply selecting the first 10 queries
    newly_sampled_queries_ids = tuning_data_bucket.unsampled_queries.elements[:10]
    remaining_queries = tuning_data_bucket.unsampled_queries.elements[10:]

    # Update Sampled Queries
    tuning_data_bucket.update_sampled_queries(newly_sampled_queries_ids)
    assert len(tuning_data_bucket.sampled_queries) == len_sampled_queries + len(
        newly_sampled_queries_ids
    )

    # Update Unsampled Queries
    tuning_data_bucket.update_unsampled_queries(remaining_queries)
    assert len(tuning_data_bucket.unsampled_queries) == (
        len_unsampled_queries - len(newly_sampled_queries_ids)
    )


# Test Filter Queries
def test_filter_queries_by_nlp_component(all_train_queries):
    domain_to_filter_by = "banking"
    filtered_ids, filtered_queries = DataBucket.filter_queries_by_nlp_component(
        query_list=all_train_queries, component_type="domain", component_name=domain_to_filter_by
    )
    assert len(filtered_ids) == len(filtered_queries)
    assert all(q.domain == domain_to_filter_by for q in filtered_queries)
