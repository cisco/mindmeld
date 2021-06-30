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
This module contains classifiers for the Active Learning Pipeline.
"""

import os
import logging
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import List, Dict
from sklearn.model_selection import StratifiedKFold
import numpy as np

from .data_loading import LabelMap, DataBucket
from .heuristics import (
    Heuristic,
    KLDivergenceSampling,
    DisagreementSampling,
    EnsembleSampling,
)

from ..components.classifier import Classifier
from ..components.nlp import NaturalLanguageProcessor
from ..constants import (
    TUNE_LEVEL_INTENT,
    ACTIVE_LEARNING_RANDOM_SEED,
    AL_DEFAULT_AGGREGATE_STATISTIC,
    AL_DEFAULT_CLASS_LEVEL_STATISTIC,
    AL_SUPPORTED_AGGREGATE_STATISTICS,
    AL_SUPPORTED_CLASS_LEVEL_STATISTICS,
)
from ..resource_loader import ProcessedQueryList

logger = logging.getLogger(__name__)

MULTI_MODEL_HEURISTICS = (KLDivergenceSampling, DisagreementSampling, EnsembleSampling)


class ALClassifier(ABC):
    """ Abstract class for Active Learning Classifiers."""

    def __init__(self, app_path: str, tuning_level: str):
        """
        Args:
            app_path (str): Path to MindMeld application
            tuning_level (str): The hierarchy level to tune ("domain" or "intent")
        """
        self.app_path = app_path
        self.tuning_level = tuning_level
        self.intent2idx, self.idx2intent, self.domain_indices = self._get_mappings()

    def _get_mappings(self):
        """Get mappings of intents to indices and the indices that map to each domain.

        Returns:
            intent2idx (Dict): Maps intents to indices
            idx2intent (Dict): Maps indices to intents
            domain_indices (Dict): Maps domains to a tuple containing the start and
                ending indexes of intents with the given domain.
        """
        idx = 0
        intent2idx, idx2intent, domain_indices = {}, {}, {}
        for domain in sorted(os.listdir(os.path.join(self.app_path, "domains"))):
            start_idx = idx
            for intent in sorted(
                os.listdir(os.path.join(self.app_path, "domains", domain))
            ):
                intent2idx[f"{domain}.{intent}"] = idx
                idx2intent[idx] = f"{domain}.{intent}"
                idx += 1
            end_idx = idx - 1
            domain_indices[domain] = (start_idx, end_idx)
        return intent2idx, idx2intent, domain_indices

    @abstractmethod
    def train(self):
        raise NotImplementedError(
            "Subclasses must implement their classifier's fit method."
        )


class MindMeldALClassifier(ALClassifier):
    """Active Learning classifier that uses MindMeld classifiers internally.
    Handles the training of MindMeld components (Domain or Intent classifiers)
    and collecting performance statistics (eval_stats)."""

    def __init__(
        self,
        app_path: str,
        tuning_level: str,
        n_classifiers: int,
        aggregate_statistic: str = None,
        class_level_statistic: str = None,
    ):
        """
        Args:
            app_path (str): Path to MindMeld application
            tuning_level (str): The hierarchy level to tune ("domain" or "intent")
            n_classifiers (int): Number of classifiers to be used by multi-model strategies.
        """
        super().__init__(app_path=app_path, tuning_level=tuning_level)
        self.nlp = NaturalLanguageProcessor(self.app_path)
        self.n_classifiers = n_classifiers
        self.aggregate_statistic = MindMeldALClassifier._validate_aggregate_statistic(
            aggregate_statistic
        )
        self.class_level_statistic = (
            MindMeldALClassifier._validate_class_level_statistic(class_level_statistic)
        )

    @staticmethod
    def _validate_aggregate_statistic(aggregate_statistic):
        """Method to validate the aggregate statistic. If an aggregate statistic is not provided
        the default is used. (Options: "accuracy", "f1_weighted", "f1_macro", "f1_micro".)

        Args:
            aggregate_statistic (str): Aggregate statistic to record.
        Returns:
            aggregate_statistic (str): Aggregate statistic to record.
        Raises:
            ValueError: If an invalid value is provided.
        """
        if not aggregate_statistic:
            logger.info(
                "Aggregate statistic not defined, using default: %r.",
                AL_DEFAULT_AGGREGATE_STATISTIC,
            )
            return AL_DEFAULT_AGGREGATE_STATISTIC
        if aggregate_statistic not in AL_SUPPORTED_AGGREGATE_STATISTICS:
            raise ValueError(
                "Not a valid aggregate statistic: {!r}.".format(aggregate_statistic)
            )
        return aggregate_statistic

    @staticmethod
    def _validate_class_level_statistic(class_level_statistic):
        """Method to validate the class-level statistic. If an class-level statistic is not provided
        the default is used. (Options: "f_beta", "percision", "recall")

        Args:
            class_level_statistic (str): Class_level statistic to record.
        Returns:
            class_level_statistic (str): Class_level statistic to record.
        Raises:
            ValueError: If an invalid value is provided.
        """
        if not class_level_statistic:
            logger.info(
                "Class-level statistic not defined, using default: %r.",
                AL_DEFAULT_CLASS_LEVEL_STATISTIC,
            )
            return AL_DEFAULT_CLASS_LEVEL_STATISTIC
        if class_level_statistic not in AL_SUPPORTED_CLASS_LEVEL_STATISTICS:
            raise ValueError(
                "Not a valid class-level statistic: {!r}.".format(class_level_statistic)
            )
        return class_level_statistic

    @staticmethod
    def _get_probs(
        classifier: Classifier, queries: ProcessedQueryList, nlp_component_to_id: Dict
    ):
        """Get the probability distribution for a query across domains or intents.

        Args:
            classifier (MindMeld Classifer): Domain or Intent Classifier
            queries (ProcessedQueryList): List of MindMeld queries
            nlp_component_to_id (Dict): Dictionary mapping domain or intent names to vector index
                positions.

        Returns:
            prob_vector (List[List]]): Probability distribution vectors for given queries.
        """
        queries_prob_vectors = []
        if queries:
            classifier_eval = classifier.evaluate(queries=queries)
            for x in classifier_eval.results:
                query_prob_vector = np.zeros(len(nlp_component_to_id))
                for nlp_component, index in x.probas.items():
                    query_prob_vector[nlp_component_to_id[nlp_component]] = index
                queries_prob_vectors.append(query_prob_vector)
            assert len(queries_prob_vectors) == len(queries)
        return queries_prob_vectors

    def _pad_intent_probs(
        self, ic_queries_prob_vectors: List[List[float]], intents: List
    ):
        """Pads the intent probability array with zeroes for out-of-domain intents.
        Args:
            ic_queries_prob_vectors (List[List[float]]]): 2D Array containing the probability
                distribution for a single query across intents in the query's domain.
            intents (List): List intents in the order that corresponds with the intent
                probabities for the queries. Intents are in the form "domain.intent".
        Returns:
            padded_ic_queries_prob_vectors (List[List[float]]]): 2D Array containing the probability
                distribution for a single query across all intents (including out-of-domain
                intents).
        """
        padded_ic_queries_prob_vectors = []
        for unordered_ic_query_prob_vector in ic_queries_prob_vectors:
            ordered_ic_query_prob_vector = np.zeros(len(self.intent2idx))
            for i, intent in enumerate(intents):
                ordered_ic_query_prob_vector[
                    self.intent2idx[intent]
                ] = unordered_ic_query_prob_vector[i]
            padded_ic_queries_prob_vectors.append(ordered_ic_query_prob_vector)
        return padded_ic_queries_prob_vectors

    def train(self, data_bucket: DataBucket, heuristic: Heuristic):
        """Main training function.

        Args:
            data_bucket (DataBucket): DataBucket for current iteration
            heuristic (Heuristic): Current Heuristic.

        Returns:
            eval_stats (defaultdict): Evaluation metrics to be included in accuracies.json
            confidences_2d (List[List]): 2D array with probability vectors for unsampled queries
            confidences_3d (List[List[List]]]): 3D array with probability vectors for unsampled
                queries from multiple classifiers
            domain_indices (Dict): Maps domains to a tuple containing the start and
                ending indexes of intents with the given domain.
        """
        eval_stats = defaultdict(dict)
        eval_stats["num_sampled"] = len(data_bucket.sampled_queries)
        confidences_2d = self.train_single(data_bucket, eval_stats)
        return_confidences_3d = isinstance(heuristic, MULTI_MODEL_HEURISTICS)
        confidences_3d = (
            self.train_multi(data_bucket) if return_confidences_3d else None
        )
        domain_indices = (
            self.domain_indices if isinstance(heuristic, KLDivergenceSampling) else None
        )
        return eval_stats, confidences_2d, confidences_3d, domain_indices

    def train_single(self, data_bucket: DataBucket, eval_stats: defaultdict = None):
        """Trains a single model to get a 2D probability array for single-model selection strategies.
        Args:
            data_bucket (DataBucket): Databucket for current iteration
            eval_stats (defaultdict): Evaluation metrics to be included in accuracies.json
        Returns:
            confidences_2d (List): 2D array with probability vectors for unsampled queries
        """
        return self._train_single(
            sampled_queries=data_bucket.sampled_queries,
            unsampled_queries=data_bucket.unsampled_queries,
            test_queries=data_bucket.test_queries,
            label_map=data_bucket.label_map,
            eval_stats=eval_stats,
        )

    def _train_single(
        self,
        sampled_queries: ProcessedQueryList,
        unsampled_queries: ProcessedQueryList,
        test_queries: ProcessedQueryList,
        label_map: LabelMap,
        eval_stats: Dict = None,
    ):
        """Helper function to train a single model and obtain a 2D probability array.
        Args:
            sampled_queries (ProcessedQueryList): Current set of sampled queries in DataBucket.
            unsampled_queries (ProcessedQueryList): Current set of unsampled queries in DataBucket.
            test_queries (ProcessedQueryList): Set of test queries in DataBucket.
            label_map LabelMap: Class that stores index mappings for a MindMeld app.
                (Eg. domain2id, domain_to_intent2id)
            eval_stats (Dict): Evaluation metrics to be included in accuracies.json
        Returns:
            confidences_2d (List): 2D array with probability vectors for unsampled queries
        """

        # Domain_Level
        dc_queries_prob_vectors, dc_eval_test = self.domain_classifier_fit_eval(
            sampled_queries=sampled_queries,
            unsampled_queries=unsampled_queries,
            test_queries=test_queries,
            domain2id=label_map.domain2id,
        )
        if eval_stats:
            self._update_eval_stats_domain_level(eval_stats, dc_eval_test)
        confidences_2d = dc_queries_prob_vectors

        # Intent_Level
        if self.tuning_level == TUNE_LEVEL_INTENT:
            (
                ic_queries_prob_vectors,
                ic_eval_test_dict,
            ) = self.intent_classifiers_fit_eval(
                sampled_queries=sampled_queries,
                unsampled_queries=unsampled_queries,
                test_queries=test_queries,
                domain_list=list(label_map.domain2id),
                domain_to_intent2id=label_map.domain_to_intent2id,
            )
            if eval_stats:
                self._update_eval_stats_intent_level(eval_stats, ic_eval_test_dict)
            confidences_2d = ic_queries_prob_vectors

        return confidences_2d

    def train_multi(self, data_bucket: DataBucket):
        """Trains multiple models to get a 3D probability array for multi-model selection strategies.
        Args:
            data_bucket (DataBucket): Databucket for current iteration
        Returns:
            confidences_3d (List[List[List]]]): 3D array with probability vectors for unsampled
                queries from multiple classifiers
        """
        return self._train_multi(
            sampled_queries=data_bucket.sampled_queries,
            unsampled_queries=data_bucket.unsampled_queries,
            test_queries=data_bucket.test_queries,
            label_map=data_bucket.label_map,
        )

    def _train_multi(
        self,
        sampled_queries: ProcessedQueryList,
        unsampled_queries: ProcessedQueryList,
        test_queries: ProcessedQueryList,
        label_map: LabelMap,
    ):
        """Helper function to train multiple models and obtain a 3D probability array.
        Args:
            sampled_queries (ProcessedQueryList): Current set of sampled queries in DataBucket.
            unsampled_queries (ProcessedQueryList): Current set of unsampled queries in DataBucket.
            test_queries (ProcessedQueryList): Set of test queries in DataBucket.
            label_map LabelMap: Class that stores index mappings for a MindMeld app.
                (Eg. domain2Id, domain_to_intent2id)
        Returns:
            confidences_3d (List[List[List]]]): 3D array with probability vectors for unsampled
                queries from multiple classifiers
        """

        sampled_queries_ids = sampled_queries.elements
        skf = StratifiedKFold(
            n_splits=self.n_classifiers,
            shuffle=True,
            random_state=ACTIVE_LEARNING_RANDOM_SEED,
        )
        y = [
            f"{domain}.{intent}"
            for domain, intent in zip(
                sampled_queries.domains(), sampled_queries.intents()
            )
        ]
        fold_sampled_queries_ids = [
            [sampled_queries_ids[i] for i in fold]
            for _, fold in skf.split(sampled_queries_ids, y)
        ]
        fold_sampled_queries_lists = [
            ProcessedQueryList(sampled_queries.cache, fold)
            for fold in fold_sampled_queries_ids
        ]
        confidences_3d = []
        for fold_sample_queries in fold_sampled_queries_lists:
            confidences_3d.append(
                self._train_single(
                    fold_sample_queries, unsampled_queries, test_queries, label_map
                )
            )
        return confidences_3d

    def domain_classifier_fit_eval(
        self,
        sampled_queries: ProcessedQueryList,
        unsampled_queries: ProcessedQueryList,
        test_queries: ProcessedQueryList,
        domain2id: Dict,
    ):
        """Fit and evaluate the domain classifier.
        Args:
            sampled_queries (ProcessedQueryList): List of Sampled Queries
            unsampled_queries (ProcessedQueryList): List of Unsampled Queries
            test_queries (ProcessedQueryList): List of Test Queries
            domain2id (Dict): Dictionary mapping domains to IDs

        Returns:
            dc_queries_prob_vectors (List[List]): List of probability distributions
                for unsampled queries.
            dc_eval_test (mindmeld.models.model.StandardModelEvaluation): Mindmeld evaluation
                object for the domain classifier.
        """
        dc = self.nlp.domain_classifier
        dc.fit(queries=sampled_queries)
        dc_eval_test = dc.evaluate(queries=test_queries)
        dc_queries_prob_vectors = MindMeldALClassifier._get_probs(
            dc, unsampled_queries, domain2id
        )
        return dc_queries_prob_vectors, dc_eval_test

    def _update_eval_stats_domain_level(self, eval_stats: Dict, dc_eval_test):
        """Update the eval_stats dictionary with evaluation metrics from the domain
        classifier.

        Args:
            eval_stats (Dict): Evaluation metrics to be included in accuracies.json
            dc_eval_test (mindmeld.models.model.StandardModelEvaluation): Mindmeld evaluation
                object for the domain classifier.
        """
        eval_stats["accuracies"]["overall"] = dc_eval_test.get_stats()["stats_overall"][
            self.aggregate_statistic
        ]
        logger.info(
            "Overall Domain-level Accuracy: %s", eval_stats["accuracies"]["overall"]
        )

    def intent_classifiers_fit_eval(
        self,
        sampled_queries: ProcessedQueryList,
        unsampled_queries: ProcessedQueryList,
        test_queries: ProcessedQueryList,
        domain_list: Dict,
        domain_to_intent2id: Dict,
    ):
        """Fit and evaluate the domain classifier.
        Args:
            sampled_queries (ProcessedQueryList): List of Sampled Queries.
            unsampled_queries (ProcessedQueryList): List of Unsampled Queries.
            test_queries (ProcessedQueryList): List of Test Queries.
            domain_list (List[str]): List of domains used by the application.
            domain_to_intent2id (Dict): Dictionary mapping intents to IDs.

        Returns:
            ic_queries_prob_vectors (List[List]): List of probability distributions
                for unsampled queries.
            ic_eval_test_dict (Dict): Dictionary mapping a domain (str) to the
                associated ic_eval_test object.
        """
        ic_eval_test_dict = {}
        unsampled_idx_preds_pairs = []
        for domain in domain_list:
            # Filter Queries
            _, filtered_sampled_queries = DataBucket.filter_queries_by_domain(
                sampled_queries, domain
            )
            (
                filtered_unsampled_queries_indices,
                filtered_unsampled_queries,
            ) = DataBucket.filter_queries_by_domain(unsampled_queries, domain)
            _, filtered_test_queries = DataBucket.filter_queries_by_domain(
                test_queries, domain
            )
            # Train
            ic = self.nlp.domains[domain].intent_classifier
            ic.fit(queries=filtered_sampled_queries)
            # Evaluate Test Queries
            ic_eval_test = ic.evaluate(queries=filtered_test_queries)
            if not ic_eval_test:
                unsampled_idx_preds_pairs.extend(
                    (i, np.zeros(len(self.intent2idx)))
                    for i in filtered_unsampled_queries_indices
                )
                continue
            ic_eval_test_dict[domain] = ic_eval_test
            # Get Probability Vectors
            ic_queries_prob_vectors = MindMeldALClassifier._get_probs(
                classifier=ic,
                queries=filtered_unsampled_queries,
                nlp_component_to_id=domain_to_intent2id[domain],
            )
            intents = [
                f"{domain}.{intent}"
                for intent in ic_eval_test.get_stats()["class_labels"]
            ]
            padded_ic_queries_prob_vectors = self._pad_intent_probs(
                ic_queries_prob_vectors, intents
            )
            for i in range(len(filtered_unsampled_queries)):
                unsampled_idx_preds_pairs.append(
                    (
                        filtered_unsampled_queries_indices[i],
                        padded_ic_queries_prob_vectors[i],
                    )
                )
        unsampled_idx_preds_pairs.sort(key=lambda x: x[0])
        padded_ic_queries_prob_vectors = [x[1] for x in unsampled_idx_preds_pairs]
        return padded_ic_queries_prob_vectors, ic_eval_test_dict

    def _update_eval_stats_intent_level(
        self, eval_stats: defaultdict, ic_eval_test_dict: Dict
    ):
        """Update the eval_stats dictionary with evaluation metrics from intent
        classifiers.

        Args:
            eval_stats (defaultdict): Evaluation metrics to be included in accuracies.json.
            ic_eval_test_dict (Dict): Dictionary mapping a domain (str) to the
                associated ic_eval_test object.
        """
        for domain, ic_eval_test in ic_eval_test_dict.items():
            eval_stats["accuracies"][domain] = {
                "overall": ic_eval_test.get_stats()["stats_overall"][
                    self.aggregate_statistic
                ]
            }
            for i, intent in enumerate(ic_eval_test.get_stats()["class_labels"]):
                eval_stats["accuracies"][domain][intent] = {
                    "overall": ic_eval_test.get_stats()["class_stats"][
                        self.class_level_statistic
                    ][i]
                }
