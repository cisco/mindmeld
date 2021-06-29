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
This module contains query selection heuristics for the Active Learning Pipeline.
"""

from abc import ABC, abstractmethod
from typing import List, Dict
from collections import defaultdict
from scipy.stats import entropy as scipy_entropy
import numpy as np

from ..constants import (
    ENTROPY_LOG_BASE,
    ACTIVE_LEARNING_RANDOM_SEED,
)


def stratified_random_sample(labels: List) -> List[int]:
    """Reorders indices in evenly repeating pattern for as long as possible and then
    shuffles and appends the remaining labels. The first part of this list will maintain
    a uniform distrubition across labels, however, since the labels may not be perfectly
    balanced the remaining portion will have a similar distribution as the original data.

                 |-------- Evenly Repeating --------||--- Shuffled Remaining ----|
    For Example: ["R","B","C","R","B","C","R","B","C","B","R","R","B","B","B","R"]

    Args:
        labels (List[str or int]): A list of labels. (Eg: labels = ["R", "B", "B", "C"])
    Returns:
        ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
    """
    np.random.seed(ACTIVE_LEARNING_RANDOM_SEED)

    label_to_indices = _get_labels_to_indices(labels)
    lowest_label_freq = min([len(indices) for indices in label_to_indices.values()])
    avg_label_freq = len(labels) // len(label_to_indices)
    sample_per_label = min(lowest_label_freq, avg_label_freq)

    selected_indices = []
    for i in range(sample_per_label):
        for indices in label_to_indices.values():
            selected_indices.append(indices[i])

    remaining_indices = [i for i in range(len(labels)) if i not in selected_indices]
    np.random.shuffle(remaining_indices)

    ranked_indices = selected_indices + remaining_indices
    return ranked_indices


def _get_labels_to_indices(labels: List) -> defaultdict:
    """Get a Dict mapping unique labels to indices in original data.
    For Example: {"R": [1,3,5], "B": [2,6], "C":[4]}

    Args:
        labels (List[str or int]): A list of labels. (Eg: labels = ["R", "B", "B", "C"])
    Returns:
        labels_to_indices (defaultdict): Dictionary that maps unique labels to indices
            where the label occurred in the original data.
    """
    labels_to_indices = defaultdict(list)
    for i, label in enumerate(labels):
        labels_to_indices[label].append(i)
    for v in labels_to_indices.values():
        np.random.shuffle(v)
    return labels_to_indices


class Heuristic(ABC):
    """ Heuristic base class used as Active Learning query selection strategies."""

    @staticmethod
    @abstractmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """Ranking method for 2d confidence arrays.
        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """Ranking method for 3d confidence arrays.
        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    def _convert_to_sample_ranks(ordered_sample_indices: List[int]):
        """
        Args:
            ordered_sample_indices (List[int]): List of indices corresponding to values ordered
            from least to greatest.
        Returns:
            sample_ranks (List[int]): List where the value at each index is the rank of the
                corresponding sample.
        """
        sample_ranks = np.zeros(len(ordered_sample_indices), dtype=int)
        for rank, sample_index in enumerate(ordered_sample_indices):
            sample_ranks[sample_index] = rank
        return sample_ranks

    @staticmethod
    def ordered_indices_list_to_final_rank(
        ordered_sample_indices_list: List[List[int]],
    ):
        """Converts multiple lists of ordered indices to a final rank.
        Args:
            ordered_sample_indices_list (List[List[int]]): Multiple lists of ordered sample indices.

        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.

        """
        all_sample_ranks = np.apply_along_axis(
            Heuristic._convert_to_sample_ranks, axis=1, arr=ordered_sample_indices_list
        )
        total_sample_ranks = all_sample_ranks.sum(axis=0)
        return list(np.argsort(total_sample_ranks))


class RandomSampling(ABC):
    @staticmethod
    def random_rank(num_elements: int) -> List[int]:
        """Randomly shuffles indices.
        Args:
            num_elements (int): Number of elements to randomly sample.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        ranked_indices = np.arange(num_elements)
        np.random.shuffle(ranked_indices)
        return list(ranked_indices)

    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """Randomly shuffles indices.
        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        num_elements = len(confidences_2d)
        return RandomSampling.random_rank(num_elements)

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """Randomly shuffles indices.
        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        _, num_elements, _ = np.array(confidences_3d).shape
        return RandomSampling.random_rank(num_elements)


class LeastConfidenceSampling(ABC):
    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """First calculates the highest (max) confidences per element and then returns
        the elements from lowest confidence to highest confidence.

        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        highest_confidence_per_element = np.max(confidences_2d, axis=1)
        return list(np.argsort(highest_confidence_per_element))

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """First calculates the highest (max) confidences per element and then returns
        the elements with the lowest max confidence. This is done for each confidence_2d
        in a confidence_3d. The rankings are added to generate a final ranking.

        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        all_ordered_sample_indices = [
            LeastConfidenceSampling.rank_2d(c) for c in confidences_3d
        ]
        return Heuristic.ordered_indices_list_to_final_rank(all_ordered_sample_indices)


class MarginSampling(ABC):
    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """Calculates the "margin" or difference between the highest and second highest
        confidence score per element. Elements are ranked from highest to lowest margin.

        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        _, element_size = np.array(confidences_2d).shape
        descending_confidences_per_element = np.partition(
            confidences_2d, kth=(element_size - 2)
        )
        highest_val_per_element = descending_confidences_per_element[:, -1]
        second_highest_val_per_element = descending_confidences_per_element[:, -2]
        margin_per_element = np.abs(
            highest_val_per_element - second_highest_val_per_element
        )
        ranked_indices_low_to_high_margin = np.argsort(margin_per_element)
        return list(ranked_indices_low_to_high_margin)

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """Calculates the "margin" or difference between the highest and second highest
        confidence score per element. Elements are ranked from highest to lowest margin.
        This is done for each confidence_2d in a confidence_3d. The rankings are added
        to generate a final ranking.

        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        all_ordered_sample_indices = [MarginSampling.rank_2d(c) for c in confidences_3d]
        return Heuristic.ordered_indices_list_to_final_rank(all_ordered_sample_indices)


class EntropySampling(ABC):
    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """Calculates the entropy score of the confidences per element. Elements are ranked from
        highest to lowest entropy.

        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        entropy_per_element = scipy_entropy(
            np.array(confidences_2d), axis=1, base=ENTROPY_LOG_BASE
        )
        high_to_low_entropy = np.argsort(entropy_per_element)[::-1]
        return list(high_to_low_entropy)

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """Calculates the entropy score of the confidences per element. Elements are ranked from
        highest to lowest entropy. This is done for each confidence_2d in a confidence_3d. The
        rankings are added to generate a final ranking.

        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        all_ordered_sample_indices = [
            EntropySampling.rank_2d(c) for c in confidences_3d
        ]
        return Heuristic.ordered_indices_list_to_final_rank(all_ordered_sample_indices)


class DisagreementSampling(ABC):
    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """Need confidences_2d from more than one model (confidences_3d) to run
        DisagreementSampling.

        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        raise NotImplementedError(
            "DisagreementSampling does not support 2d confidences."
        )

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """Finds the most frequent class label for a given element across all models.
        Calculates the agreement per element (% of models who voted the most frequent class).
        Ranks elements by highest to lowest disagreement.

        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        # X: Model, Y: Classes Chosen Per Element
        chosen_classes_per_model = np.argmax(confidences_3d, axis=2)
        disagreement_scores = 1 - np.array(
            [max(np.bincount(row)) / len(row) for row in chosen_classes_per_model.T]
        )
        high_to_low_disagreement = np.argsort(disagreement_scores)
        return list(high_to_low_disagreement)


class KLDivergenceSampling(ABC):
    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """Need confidences_2d from more than one model (confidences_3d) to run
        KLDivergenceSampling.

        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        raise NotImplementedError(
            "KLDivergenceSampling does not support 2d confidences."
        )

    @staticmethod
    def rank_3d(
        confidences_3d: List[List[List[float]]], confidence_segments: Dict = None
    ) -> List[int]:
        """Calculates the KL Divergence between the average confidence distribution across
        all models for a given class and the confidence distribution for a given element in
        said class. Elements are ranked from highest to lowest divergence.

        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
            confidence_segments (Dict[(str, Tuple(int,int))]): A dictionary mapping
                segments to run KL Divergence.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        if confidence_segments:
            divergences = (
                KLDivergenceSampling.get_divergences_per_element_with_segments(
                    confidences_3d, confidence_segments
                )
            )
        else:
            divergences = KLDivergenceSampling.get_divergences_per_element_no_segments(
                confidences_3d
            )
        divergence_per_element = np.max(divergences, axis=0)
        ranked_indices_high_to_low_divergence = np.argsort(divergence_per_element)[::-1]
        return list(ranked_indices_high_to_low_divergence)

    @staticmethod
    def get_divergences_per_element_no_segments(
        confidences_3d: List[List[List[float]]],
    ) -> List[List[float]]:
        """
        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            divergences (List[List[float]]): Divergences per model for each element.
        """
        # Calculate average prediction values.
        q_x = np.mean(confidences_3d, axis=0)
        # Duplicate the mean distribution by number of models
        num_models, _, _ = np.array(confidences_3d).shape
        q_x = [q_x for n in range(num_models)]
        # X: Model, Y: Divergence Per Element
        divergences = scipy_entropy(confidences_3d, q_x, axis=2, base=ENTROPY_LOG_BASE)
        return divergences

    @staticmethod
    def get_divergences_per_element_with_segments(
        confidences_3d: List[List[List[float]]], confidence_segments: Dict
    ) -> List[List[float]]:
        """Calculate divergences by segments defined in confidence segments where
        p_d is the probabilities within class X and q_d is the mean probability distribution
        for class X. Divergence(p_d, q_d) is calculated for each element in all classes.

        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
            confidence_segments (Dict[(str, Tuple(int,int))]): A dictionary mapping
                segments to run KL Divergence.
        Returns:
            divergences (List[List[float]]): Divergences per model for each element.
        """
        avg_preds = np.mean(confidences_3d, axis=0)
        divergences = []
        # Calculate q_d
        q_d = {d: [] for d in confidence_segments}
        for row in avg_preds:
            domain = KLDivergenceSampling.get_domain(confidence_segments, row)
            q_d[domain].append(row)
        for pred in confidences_3d:
            # Calculate p_d
            p_d = {d: [] for d in confidence_segments}
            for row in pred:
                domain = KLDivergenceSampling.get_domain(confidence_segments, row)
                p_d[domain].append(row)
            # Calculate Divergence Scores by Domain
            divergence_scores_by_domain = {d: [] for d in confidence_segments}
            for domain in confidence_segments:
                if len(p_d[domain]) > 0 and len(q_d[domain]) > 0:
                    divergence_scores_by_domain[domain] = scipy_entropy(
                        np.array(p_d[domain]),
                        np.array(q_d[domain]),
                        axis=1,
                        base=ENTROPY_LOG_BASE,
                    )
                else:
                    divergence_scores_by_domain[domain] = 0
            # Reorder Divergence Scores by Domain to Original order
            single_pred_divergence_scores = []
            domain_counter = {d: 0 for d in confidence_segments}
            for row in pred:
                domain = KLDivergenceSampling.get_domain(confidence_segments, row)
                single_pred_divergence_scores.append(
                    divergence_scores_by_domain[domain][domain_counter[domain]]
                )
                domain_counter[domain] += 1
            divergences.append(single_pred_divergence_scores)
        return divergences

    @staticmethod
    def get_domain(confidence_segments: Dict, row: List[List[float]]) -> str:
        """Get the domain for a given probability row, inferred based on the non-zero values.
        Args:
            confidence_segments (Dict[str, tuple(int, int)]): A mapping between domains (str) to the
                corresponding indices in the probability vector. Used for intent-level KLD.
            row (List[List[float]]): A single row representing a queries probability distrubition.
        Returns:
            domain (str): The domain that the given row belongs to.
        Raises:
            AssertionError: If a row does not have an associated domain.
        """
        if np.sum(row) == 0:
            row = list(np.repeat(1 / len(row), repeats=len(row)))
        for domain in confidence_segments:
            start, end = confidence_segments[domain]
            if sum(row[start : end + 1]) > 0:
                return domain
        raise AssertionError(f"Row does not have an associated domain, Row: {row}")


class EnsembleSampling(ABC):
    @staticmethod
    def get_heuristics_2d() -> tuple:
        return (LeastConfidenceSampling, MarginSampling, EntropySampling)

    @staticmethod
    def get_heuristics_3d() -> tuple:
        return (
            LeastConfidenceSampling,
            MarginSampling,
            EntropySampling,
            DisagreementSampling,
            KLDivergenceSampling,
        )

    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """Combine ranks from all heuristics that can support ranking given 2d confidence
        input.

        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        all_ordered_sample_indices = [
            heuristic.rank_2d(confidences_2d)
            for heuristic in EnsembleSampling.get_heuristics_2d()
        ]
        return Heuristic.ordered_indices_list_to_final_rank(all_ordered_sample_indices)

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """Combine ranks from all heuristics that can support ranking given 3d confidence
        input.

        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by the heuristic.
        """
        all_ordered_sample_indices = [
            heuristic.rank_3d(confidences_3d)
            for heuristic in EnsembleSampling.get_heuristics_3d()
        ]
        return Heuristic.ordered_indices_list_to_final_rank(all_ordered_sample_indices)


class HeuristicsFactory:
    """Heuristics Factory Class"""

    @staticmethod
    def get_heuristic(heuristic) -> Heuristic:
        """A static method to get a Heuristic class.

        Args:
            heuristic (str): Name of the desired Heuristic class
        Returns:
            (Heuristic): Heuristic Class
        """
        heuristic_classes = [
            RandomSampling,
            LeastConfidenceSampling,
            MarginSampling,
            EntropySampling,
            DisagreementSampling,
            EnsembleSampling,
            KLDivergenceSampling,
        ]
        for heuristic_class in heuristic_classes:
            if heuristic == heuristic_class.__name__:
                return heuristic_class()
        raise AssertionError(f" {heuristic} is not a valid heuristic.")
