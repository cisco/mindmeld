from abc import ABC, abstractmethod
from typing import List, Dict
from collections import Counter, defaultdict
from scipy.stats import entropy as scipy_entropy
import numpy as np

from ..constants import (
    ENTROPY_LOG_BASE,
    ACTIVE_LEARNING_RANDOM_SEED,
)


def stratified_random_sample(labels: List) -> List[int]:
    """
    Args:
        labels (List[str or int]): A list of labels. (Eg: labels = ["R", "B", "B", "C"])
    Returns:
        ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
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
    """
    Args:
        labels (List[str or int]): A list of labels. (Eg: labels = ["R", "B", "B", "C"])
    Returns:
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
        """
        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        raise NotImplementedError("Subclasses must implement this method")

    @staticmethod
    @abstractmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """
        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        raise NotImplementedError("Subclasses must implement this method")


class RandomSampling(ABC):
    @staticmethod
    def random_rank(num_elements: int) -> List[int]:
        """
        Args:
            num_elements (int): Number of elements to randomly sample.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by randomly.
        """
        return list(np.random.choice(range(num_elements), num_elements, replace=False))

    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """
        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        num_elements = len(confidences_2d)
        return RandomSampling.random_rank(num_elements)

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """
        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        _, num_elements, _ = np.array(confidences_3d).shape
        return RandomSampling.random_rank(num_elements)


class LeastConfidenceSampling(ABC):
    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """
        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        highest_confidence_per_element = np.max(confidences_2d, axis=1)
        return list(np.argsort(highest_confidence_per_element))

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """
        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        total_rank_score = np.sum(
            [LeastConfidenceSampling.rank_2d(c) for c in confidences_3d], axis=1
        )
        high_to_low_ranked_indices = np.argsort(total_rank_score)[::-1]
        return list(high_to_low_ranked_indices)


class MarginSampling(ABC):
    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """
        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        _, element_size = np.array(confidences_2d).shape
        descending_confidences_per_element = np.partition(
            confidences_2d, kth=(element_size - 1)
        )[::-1]
        highest_val_per_element = descending_confidences_per_element[:, 1]
        second_highest_val_per_element = descending_confidences_per_element[:, 2]
        margin_per_element = np.abs(
            highest_val_per_element - second_highest_val_per_element
        )
        ranked_indices_high_to_low_margin = np.argsort(margin_per_element)[::-1]
        return list(ranked_indices_high_to_low_margin)

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """
        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        total_rank_score = np.sum(
            [MarginSampling.rank_2d(c) for c in confidences_3d], axis=1
        )
        high_to_low_ranked_indices = np.argsort(total_rank_score)[::-1]
        return list(high_to_low_ranked_indices)


class EntropySampling(ABC):
    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """
        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        entropy_per_element = scipy_entropy(
            np.array(confidences_2d), axis=1, base=ENTROPY_LOG_BASE
        )
        high_to_low_entropy = np.argsort(entropy_per_element)[::-1]
        return list(high_to_low_entropy)

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """
        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        total_rank_score = np.sum(
            [EntropySampling.rank_2d(c) for c in confidences_3d], axis=1
        )
        high_to_low_ranked_indices = np.argsort(total_rank_score)[::-1]
        return list(high_to_low_ranked_indices)


class DisagreementSampling(ABC):
    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """
        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        raise NotImplementedError(
            "DisagreementSampling does not support 2d confidences."
        )

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """
        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        # X: Model, Y: Classes Chosen Per Element
        chosen_classes_per_model = np.argmax(confidences_3d, axis=2)
        # X: Element, Y: Class Chosen Per Model
        chosen_classes_per_element = chosen_classes_per_model.T
        # Calculate Disagreement Scores
        disagreement_scores = []
        for row in chosen_classes_per_element:
            class_counts_per_element = Counter(row)
            _, freq = class_counts_per_element.most_common()[0]
            num_models = len(row)
            percent_voted_most_req_class = freq / num_models
            disagreement_score = 1 - percent_voted_most_req_class
            disagreement_scores.append(disagreement_score)
        high_to_low_disagreement = np.argsort(disagreement_scores)
        return list(high_to_low_disagreement)


class KLDivergenceSampling(ABC):
    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """
        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        raise NotImplementedError(
            "KLDivergenceSampling does not support 2d confidences."
        )

    @staticmethod
    def rank_3d(
        confidences_3d: List[List[List[float]]], confidence_segments: Dict = None
    ) -> List[int]:
        """
        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
            confidence_segments (Dict[(str, Tuple(int,int))]): A dictionary mapping
                segments to run KL Divergence.
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

        # X: Element, Y: Divergence Per Model
        divergences = np.array(divergences).T
        divergence_per_element = np.max(divergences, axis=1)
        ranked_indices_high_to_low_divergence = np.argsort(divergence_per_element)[::-1]
        return list(ranked_indices_high_to_low_divergence)

    @staticmethod
    def get_divergences_per_element_no_segments(
        confidences_3d: List[List[List[float]]],
    ):
        # Calculate average prediction values.
        q_x = np.mean(confidences_3d, axis=0)
        # Duplicate the mean distribution by number of models
        num_models, _, _ = np.array(confidences_3d).shape
        q_x = [q_x for n in range(num_models)]
        # X: Model, Y: Divergence Per Element
        divergences = scipy_entropy(confidences_3d, q_x, axis=2, base=ENTROPY_LOG_BASE)
        return divergences

    @staticmethod
    def get_divergences_per_element_with_segments(confidences_3d, confidence_segments):
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
    def get_domain(confidence_segments, row) -> str:
        """Get the domain for a given probability row, inferred based on the non-zero values.
        Args:
            preds_multi (List[List[float]]): Probability scores for each data point for each
                class from multiple classifiers.
            avg_preds (List[float]): Average probability vector across all intents.
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
    def get_heuristics_2d():
        return (LeastConfidenceSampling, MarginSampling, EntropySampling)

    @staticmethod
    def get_heuristics_3d():
        return (
            LeastConfidenceSampling,
            MarginSampling,
            EntropySampling,
            DisagreementSampling,
            KLDivergenceSampling,
        )

    @staticmethod
    def rank_2d(confidences_2d: List[List[float]]) -> List[int]:
        """
        Args:
            confidences_2d (List[List[float]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        num_elements = len(confidences_2d)
        total_rank_score = np.zeros(num_elements, dtype=int)
        for heuristic in EnsembleSampling.get_heuristics_2d():
            total_rank_score += heuristic.rank_2d(confidences_2d)
        high_to_low_ranked_indices = np.argsort(total_rank_score)[::-1]
        return list(high_to_low_ranked_indices)

    @staticmethod
    def rank_3d(confidences_3d: List[List[List[float]]]) -> List[int]:
        """
        Args:
            confidences_3d (List[List[List[float]]]): Confidence probabilities per element.
        Returns:
            ranked_indices (List[int]): Indices corresponding to elements ranked by utility.
        """
        _, num_elements, _ = np.array(confidences_3d).shape
        total_rank_score = np.zeros(num_elements, dtype=int)
        for heuristic in EnsembleSampling.get_heuristics_3d():
            total_rank_score += heuristic.rank_3d(confidences_3d)
        high_to_low_ranked_indices = np.argsort(total_rank_score)[::-1]
        return list(high_to_low_ranked_indices)


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
