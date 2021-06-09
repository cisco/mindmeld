# -*- coding: utf-8 -*-
#
# Copyright (c) 2015 Cisco Systems, Inc. and others.  All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This module contains classes used to load queries for the Active Learning Pipeline.
"""

from typing import Dict, List
import logging

from .heuristics import Heuristic, stratified_random_sample

from ..auto_annotator import BootstrapAnnotator
from ..components._config import DEFAULT_AUTO_ANNOTATOR_CONFIG
from ..constants import TUNE_LEVEL_DOMAIN, TUNE_LEVEL_INTENT, AL_MAX_LOG_USAGE_PCT
from ..core import ProcessedQuery
from ..markup import read_query_file
from ..resource_loader import ResourceLoader, ProcessedQueryList

logger = logging.getLogger(__name__)


class LabelMap:
    """Class that handles label encoding and mapping."""

    def __init__(self, query_tree: Dict):
        """
        Args:
            query_tree (dict): Nested Dictionary containing queries.
                Has the format: {"domain":{"intent":[Query List]}}.
        """
        domain_to_intents = LabelMap.get_domain_to_intents(query_tree)

        self.domain2id = LabelMap._get_domain_mappings(domain_to_intents)
        self.id2domain = LabelMap._reverse_dict(self.domain2id)
        self.domain_to_intent2id = LabelMap._get_intent_mappings(domain_to_intents)
        self.id2intent = LabelMap._reverse_nested_dict(self.domain_to_intent2id)

    @staticmethod
    def get_domain_to_intents(query_tree: Dict) -> Dict:
        """
        Args:
            query_tree (dict): Nested Dictionary containing queries.
                Has the format: {"domain":{"intent":[Query List]}}

        Returns:
            domain_to_intents (dict): Dict mapping domains to a list of intents.
        """
        domain_to_intents = {}
        for domain in query_tree:
            domain_to_intents[domain] = list(query_tree[domain])
        return domain_to_intents

    @staticmethod
    def _get_domain_mappings(domain_to_intents: Dict) -> Dict:
        """Creates a dictionary that maps domains to encoded ids.

        Args:
            domain_to_intents (dict): Dict mapping domains to a list of intents.

        Returns:
            domain2id (dict): dict with domain to id mappings.
        """
        domain2id = {}
        domains = list(domain_to_intents)
        for index, domain in enumerate(domains):
            domain2id[domain] = index
        return domain2id

    @staticmethod
    def _get_intent_mappings(domain_to_intents: Dict) -> Dict:
        """Creates a dictionary that maps intents to encoded ids.

        Args:
            domain_to_intents (dict): Dict mapping domains to a list of intents.

        Returns:
            domain_to_intent2id (dict): dict with intent to id mappings.
        """
        domain_to_intent2id = {}
        for domain in domain_to_intents:
            intent_labels = {}
            for index, intent in enumerate(domain_to_intents[domain]):
                intent_labels[intent] = index
            domain_to_intent2id[domain] = intent_labels
        return domain_to_intent2id

    @staticmethod
    def _reverse_dict(dictionary: Dict[str, int]):
        """
        Returns:
            reversed_dict (dict): Reversed dictionary.
        """
        reversed_dict = {v: k for k, v in dictionary.items()}
        return reversed_dict

    @staticmethod
    def _reverse_nested_dict(dictionary: Dict[str, Dict[str, int]]):
        """
        Returns:
            reversed_dict (dict): Reversed dictionary.
        """
        reversed_dict = {}

        for parent_key, parent_value in dictionary.items():
            reversed_dict[parent_key] = LabelMap._reverse_dict(parent_value)
        return reversed_dict

    @staticmethod
    def get_class_labels(
        tuning_level: str, query_list: ProcessedQueryList
    ) -> List[str]:
        """Creates a class label for a set of queries. These labels are used to split
            queries by type. Labels follow the format of "domain" or "domain|intent".
            For example, "date|get_date".

        Args:
            tuning_level (str): The hierarchy level to tune ("domain" or "intent")
            query_list (ProcessedQueryList): Data structure containing a list of processed queries.
        Returns:
            class_labels (List[str]): list of labels for classification task.
        """
        if tuning_level == TUNE_LEVEL_DOMAIN:
            return [f"{d}" for d in query_list.domains()]
        elif tuning_level == TUNE_LEVEL_INTENT:
            return [
                f"{d}.{i}" for d, i in zip(query_list.domains(), query_list.intents())
            ]
        else:
            raise ValueError(
                f"Invalid label_type {tuning_level}. Must be '{TUNE_LEVEL_DOMAIN}'"
                f" or '{TUNE_LEVEL_INTENT}'"
            )

    @staticmethod
    def create_label_map(app_path, file_pattern):
        """Creates a label map.

        Args:
            app_path (str): Path to MindMeld application
            file_pattern (str): Regex pattern to match text files. (".*train.*.txt")

        Returns:
            label_map (LabelMap): A label map.
        """
        resource_loader = ResourceLoader.create_resource_loader(app_path)
        query_tree = resource_loader.get_labeled_queries(label_set=file_pattern)
        return LabelMap(query_tree)


class LogQueriesLoader:
    def __init__(self, app_path: str, tuning_level: str, log_file_path: str):
        """This class loads data as processed queries from a specified log file.
        Args:
            app_path (str): Path to the MindMeld application.
            tuning_level (str): The hierarchy level to tune ("domain" or "intent")
            log_file_path (str): Path to the log file with log queries.
        """
        self.app_path = app_path
        self.tuning_level = tuning_level
        self.log_file_path = log_file_path

    @staticmethod
    def deduplicate_raw_text_queries(log_queries_iter) -> List[str]:
        """Removes duplicates in the text queries.

        Args:
            log_queries_iter (generator): Log queries generator.
        Returns:
            filtered_text_queries (List[str]): a List of filtered text queries.
        """
        return list(set(q for q in log_queries_iter))

    def convert_text_queries_to_processed(
        self, text_queries: List[str]
    ) -> List[ProcessedQuery]:
        """Converts text queries to processed queries using an annotator.

        Args:
            text_queries (List[str]): a List of text queries.
        Returns:
            queries (List[ProcessedQuery]): List of processed queries.
        """
        logger.info("Loading a Bootstrap Annotator to process log queries.")
        annotator_params = DEFAULT_AUTO_ANNOTATOR_CONFIG
        annotator_params["app_path"] = self.app_path
        bootstrap_annotator = BootstrapAnnotator(**annotator_params)
        return bootstrap_annotator.text_queries_to_processed_queries(
            text_queries=text_queries
        )

    @property
    def queries(self):
        log_queries_iter = read_query_file(self.log_file_path)
        filtered_text_queries = LogQueriesLoader.deduplicate_raw_text_queries(
            log_queries_iter
        )
        return self.convert_text_queries_to_processed(filtered_text_queries)


class DataBucket:
    """Class to hold data throughout the Active Learning training pipeline.
    Responsible for data conversion, filtration, and storage.
    """

    def __init__(
        self,
        label_map,
        resource_loader,
        test_queries: ProcessedQueryList,
        unsampled_queries: ProcessedQueryList,
        sampled_queries: ProcessedQueryList,
    ):
        """
        Args:
            app_path (str): Path to MindMeld application
            test_queries (ProcessedQueryList): Queries to use for evaluation.
            unsampled_queries (ProcessedQueryList): Queries to sample from iteratively.
            sampled_queries (ProcessedQueryList): Queries currently included in the sample set.
        """
        self.label_map = label_map
        self.resource_loader = resource_loader
        self.test_queries = test_queries
        self.unsampled_queries = unsampled_queries
        self.sampled_queries = sampled_queries

    def get_queries(self, query_ids):
        """Method to get multiple queries from the QueryCache given a list of query ids.

        Args:
            query_ids (List[int]): List of ids corresponding to queries in the QueryCache.
        Returns:
            queries (List[ProcessedQuery]): List of processed queries from the cache.
        """
        return [
            self.resource_loader.query_cache.get(query_id) for query_id in query_ids
        ]

    def update_sampled_queries(self, newly_sampled_queries_ids):
        """Update the current set of sampled queries by adding the set of newly sampled
        queries. A new PrcoessedQueryList object is created with the updated set of query ids.

        Args:
            newly_sampled_queries_ids (List[int]): List of ids corresponding the newly sampled
                queries in the QueryCache.
        """
        sampled_queries_ids = self.sampled_queries.elements + newly_sampled_queries_ids
        self.sampled_queries = ProcessedQueryList(
            cache=self.resource_loader.query_cache, elements=sampled_queries_ids
        )

    def update_unsampled_queries(self, remaining_indices):
        """Update the current set of unsampled queries by removing the set of newly sampled
        queries. A new PrcoessedQueryList object is created with the updated set of query ids.

        Args:
            remaining_indices (List[int]): List of ids corresponding the reamining queries
                queries in self.unsampled_queries.
        """
        remaining_queries_ids = [
            self.unsampled_queries.elements[i] for i in remaining_indices
        ]
        self.unsampled_queries = ProcessedQueryList(
            cache=self.resource_loader.query_cache, elements=remaining_queries_ids
        )

    def sample_and_update(
        self,
        sampling_size: int,
        confidences_2d: List[List[float]],
        confidences_3d: List[List[List[float]]],
        heuristic: Heuristic,
        confidence_segments: Dict = None,
    ):
        """Method to sample a DataBucket's unsampled_queries and update its sampled_queries
        and newly_sampled_queries.
        Args:
            sampling_size (int): Number of elements to sample in the next iteration.
            confidences_2d (List[List[float]]): Confidence probabilities per element.
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
            heuristic (Heuristic): Selection strategy.
            confidence_segments (Dict[(str, Tuple(int,int))]): A dictionary mapping
                segments to run KL Divergence.
        Returns:
            newly_sampled_queries_ids (List[int]): List of ids corresponding the newly sampled
                queries in the QueryCache.
        """

        params_rank_3d = {"confidences_3d": confidences_3d}
        if confidence_segments:
            params_rank_3d["confidence_segments"] = confidence_segments

        ranked_indices = (
            heuristic.rank_3d(**params_rank_3d)
            if confidences_3d
            else heuristic.rank_2d(confidences_2d)
        )
        newly_sampled_indices = ranked_indices[:sampling_size]
        remaining_indices = ranked_indices[sampling_size:]

        newly_sampled_queries_ids = [
            self.unsampled_queries.elements[i] for i in newly_sampled_indices
        ]
        self.update_sampled_queries(newly_sampled_queries_ids)
        self.update_unsampled_queries(remaining_indices)
        return newly_sampled_queries_ids

    @staticmethod
    def filter_queries_by_domain(query_list: ProcessedQueryList, domain: str):
        """Filter queries for training preperation.

        Args:
            query_list (list): List of queries to filter
            domain (str): Domain of desired queries

        Returns:
            filtered_queries_indices (list): List of indices of filtered queries.
            filtered_queries (list): List of filtered queries.
        """
        filtered_queries = []
        filtered_queries_indices = []
        for index, query in enumerate(query_list.processed_queries()):
            if query.domain == domain:
                filtered_queries_indices.append(index)
                filtered_queries.append(query)
        return filtered_queries_indices, filtered_queries


class DataBucketFactory:
    """Class to generate the initial data for experimentation. (Seed Queries, Remaining Queries,
    and Test Queries). Handles initial sampling and data split based on configuation details.
    """

    @staticmethod
    def get_data_bucket_for_strategy_tuning(
        app_path: str,
        tuning_level: str,
        train_pattern: str,
        test_pattern: str,
        train_seed_pct: float,
    ):
        """Creates a DataBucket to be used for strategy tuning.

        Args:
            app_path (str): Path to MindMeld application
            tuning_level (str): The hierarchy level to tune ("domain" or "intent")
            train_pattern (str): Regex pattern to match train files. (".*train.*.txt")
            test_pattern (str): Regex pattern to match test files. (".*test.*.txt")
            train_seed_pct (float): Percentage of training data to use as the initial seed

        Returns:
            strategy_tuning_data_bucket (DataBucket): DataBucket for tuning
        """
        label_map = LabelMap.create_label_map(app_path, train_pattern)
        resource_loader = ResourceLoader.create_resource_loader(app_path)

        train_query_list = resource_loader.get_flattened_label_set(
            label_set=train_pattern
        )
        train_class_labels = LabelMap.get_class_labels(tuning_level, train_query_list)
        ranked_indices = stratified_random_sample(train_class_labels)
        sampling_size = int(train_seed_pct * len(train_query_list))

        sampled_query_ids = [
            train_query_list.elements[i] for i in ranked_indices[:sampling_size]
        ]
        unsampled_query_ids = [
            train_query_list.elements[i] for i in ranked_indices[sampling_size:]
        ]

        sampled_queries = ProcessedQueryList(
            resource_loader.query_cache, sampled_query_ids
        )
        unsampled_queries = ProcessedQueryList(
            resource_loader.query_cache, unsampled_query_ids
        )
        test_queries = resource_loader.get_flattened_label_set(label_set=test_pattern)

        return DataBucket(
            label_map, resource_loader, test_queries, unsampled_queries, sampled_queries
        )

    @staticmethod
    def get_data_bucket_for_query_selection(
        app_path: str,
        tuning_level: str,
        train_pattern: str,
        test_pattern: str,
        unlabeled_logs_path: str,
        labeled_logs_pattern: str = None,
        log_usage_pct: float = AL_MAX_LOG_USAGE_PCT,
    ):
        """Creates a DataBucket to be used for log query selection.

        Args:
            app_path (str): Path to MindMeld application
            tuning_level (str): The hierarchy level to train ("domain" or "intent")
            train_pattern (str): Regex pattern to match train files. For example, ".*train.*.txt"
            test_pattern (str): Regex pattern to match test files. For example, ".*test.*.txt"
            unlabeled_logs_path (str): Path a logs text file with unlabeled queries
            labeled_logs_pattern (str): Pattern to obtain logs already labeled within a MindMeld app
            log_usage_pct (float): Percentage of the log data to use for selection

        Returns:
            query_selection_data_bucket (DataBucket): DataBucket for log query selection
        """
        label_map = LabelMap.create_label_map(app_path, train_pattern)
        resource_loader = ResourceLoader.create_resource_loader(app_path)

        if labeled_logs_pattern:
            log_query_list = resource_loader.get_flattened_label_set(
                label_set=labeled_logs_pattern
            )
        else:
            log_queries = LogQueriesLoader(
                app_path, tuning_level, unlabeled_logs_path
            ).queries
            log_queries_keys = [
                resource_loader.query_cache.get_key(q.domain, q.intent, q.query.text)
                for q in log_queries
            ]
            log_query_row_ids = [
                resource_loader.query_cache.put(key, query)
                for key, query in zip(log_queries_keys, log_queries)
            ]
            log_query_list = ProcessedQueryList(
                cache=resource_loader.query_cache, elements=log_query_row_ids
            )

        if log_usage_pct < AL_MAX_LOG_USAGE_PCT:
            sampling_size = int(log_usage_pct * len(log_query_list))
            log_class_labels = LabelMap.get_class_labels(tuning_level, log_query_list)
            ranked_indices = stratified_random_sample(log_class_labels)
            log_query_ids = [
                log_query_list.elements[i] for i in ranked_indices[:sampling_size]
            ]
            log_queries = ProcessedQueryList(log_query_list.cache, log_query_ids)

        sampled_queries = resource_loader.get_flattened_label_set(
            label_set=train_pattern
        )
        test_queries = resource_loader.get_flattened_label_set(label_set=test_pattern)

        return DataBucket(
            label_map, resource_loader, test_queries, log_query_list, sampled_queries
        )
