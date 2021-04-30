from abc import ABC, abstractmethod
from typing import List, Dict
from collections import Counter
from scipy.stats import entropy as scipy_entropy
import numpy as np

from ..constants import (
    ENTROPY_LOG_BASE,
    TRAIN_LEVEL_DOMAIN,
    TRAIN_LEVEL_INTENT,
    ACTIVE_LEARNING_RANDOM_SEED,
)


def custom_reordering(confidences, do_rank: bool = True, sampling_size: int = None):
    """
    Args:
        confidences: List of confidence scores from heuristic.
        do_rank: If True, this function will return rank-sorted confidence scores.
        sampling_size: Size of data being sampled in a turn.
    """
    if not do_rank:
        assert isinstance(sampling_size, int) and sampling_size > 0, print(
            type(sampling_size), sampling_size
        )
        if sampling_size >= len(confidences):
            return np.arange(len(confidences))
        # samples at indices {0,...,sampling_size-1}<={sampling_size}<={sampling_size+1,...}
        idxs = np.argpartition(confidences, sampling_size)
    else:
        idxs = np.argsort(confidences)  # ascending order sort
    return idxs


class Heuristic(ABC):
    """ Heuristic base class used as Active Learning query selection strategies."""

    @staticmethod
    def validate_sampling_size(sampling_size: int):
        """Verify the sampling size is a positive integer
        Args:
            sampling_size (int): Size of data being sampled in a turn.
        """
        assert isinstance(sampling_size, int)
        assert sampling_size > 0, print(
            f"Current Sampling Size: {sampling_size}, Required size > 0"
        )
        return sampling_size

    @abstractmethod
    def _extractor(self, **kwargs):
        raise NotImplementedError(
            "Subclasses must implement this method for heuristic logic."
        )

    def sample(self, **kwargs):
        """
        Args:
            sampled (list, optional): List of sampled queries
            unsampled (list, optional): List of unsampled queries
            do_rank (bool, optional): if True, returns a ranked list of the indices for
                confidence-sorting of data samples
            return_dict (bool, optional): if True, return a dict with sampled, unsampled, and
                ranked indicies.
        Returns:
            results_dict (dict, optional): A dictionary containing sampled, unsampled, and
                ranked indices.
            newly_sampled (list): Newly sampled queries
            sampled (list): Updated set of sampled queries with newly sampled queries added in.
            unsampled (list): Updated set of sampled queries with newly sampled queries removed.
            ranked (list): List of ranked queries after selection.
        """
        sampled, unsampled, do_rank = (
            kwargs.get("sampled", []),
            kwargs.get("unsampled"),
            kwargs.get("do_rank"),
        )
        sampled_indices, unsampled_indices, ranked_indices = self._extractor(**kwargs)
        if kwargs.get("return_dict"):
            return {
                "sampled_indices": sampled_indices,
                "unsampled_indices": unsampled_indices,
                "ranked_indices": ranked_indices,
            }
        else:
            newly_sampled = [unsampled[i] for i in sampled_indices]
            sampled += newly_sampled
            unsampled = [unsampled[i] for i in unsampled_indices]
            ranked = [unsampled[i] for i in ranked_indices] if do_rank else None
            return newly_sampled, sampled, unsampled, ranked


class StrategicRandomSampling(Heuristic):
    """Selection strategy that aims to randomly sample queries such that there is an even
    distribution across domains or intents.
    """

    def _extractor(
        self,
        sampling_size: int,
        unsampled: List,
        min_per_label: int = None,
        label_type: str = TRAIN_LEVEL_INTENT,
        allow_below_min: bool = False,
        **kwargs,
    ):
        """
        Args:
            sampling_size (int): Size of data being sampled in a turn.
            unsampled (list, optional): List of unsampled queries
            min_per_label (int): Min number of queries to select per label
            label_type (str): The level to split evenly by ('domain' or 'intent')
            allow_below_min (bool): Allow intents that do not have enough queries to
                meet the required count (sample_size/number_of_classes).
        Returns:
            sampled_indices (list): List of indices that were selected
            unsampled_indices (list): List of indices that were not selected
        """
        sampling_size = Heuristic.validate_sampling_size(sampling_size)
        class_labels = StrategicRandomSampling._get_class_labels(
            label_type=label_type, unsampled=unsampled
        )
        unique_labels = np.unique(class_labels)
        indices_per_label = StrategicRandomSampling._get_indices_per_label(
            unique_labels=unique_labels, class_labels=class_labels
        )
        if not min_per_label:
            min_per_label = min([len(v) for k, v in indices_per_label.items()])
            print(
                f"{len(unique_labels)} unique labels. Using min_per_label value of {min_per_label}"
            )
        evenly_sampled, outstanding_sampled = [], []
        for label in unique_labels:
            indices = indices_per_label[label]
            if len(indices) < min_per_label:
                if allow_below_min:
                    print(
                        f"Allowing {label}, Count {len(indices)}, min_per_label = {min_per_label}"
                    )
                    evenly_sampled.extend(indices)
                else:
                    min_count = min(
                        [len(indices_per_label[label]) for label in unique_labels]
                    )
                    raise ValueError(
                        f"{label} Count = {len(indices)} and min_per_label = {min_per_label}. "
                        f"The lowest label count is {min_count}."
                    )
            else:
                evenly_sampled.extend(indices[:min_per_label])
                outstanding_sampled.extend(indices[min_per_label:])
        np.random.seed(ACTIVE_LEARNING_RANDOM_SEED)
        if len(evenly_sampled) > sampling_size:
            np.random.shuffle(evenly_sampled)
            sampled_indices = evenly_sampled[:sampling_size]
            unsampled_indices = evenly_sampled[sampling_size:] + outstanding_sampled
        else:
            num_left_to_sample = sampling_size - len(evenly_sampled)
            outstanding_sampled_subset = np.random.choice(
                outstanding_sampled, num_left_to_sample, replace=False
            )
            sampled_indices = list(evenly_sampled) + list(outstanding_sampled_subset)
            unsampled_indices = [
                idx for idx, _ in enumerate(class_labels) if idx not in sampled_indices
            ]
        assert len(sampled_indices) + len(unsampled_indices) == len(class_labels)
        return sampled_indices, unsampled_indices, None

    @staticmethod
    def _get_class_labels(label_type: str, unsampled: List) -> List[str]:
        """Creates a class label for a set of queries. These labels are used to split
            queries by type.

        Args:
            unsampled (list): List of unsampled queries
            label_type (str): The level to split evenly by ('domain' or 'intent')
        Returns:
            class_labels (List[str]): list of labels for classification task. Labels follow
                the format of "domain" or "domain|intent". For example, "date|get_date".
        """
        if label_type == TRAIN_LEVEL_DOMAIN:
            return [f"{q.domain}" for q in unsampled]
        elif label_type == TRAIN_LEVEL_INTENT:
            return [f"{q.domain}|{q.intent}" for q in unsampled]
        else:
            raise ValueError(
                f"Invalid label_type {label_type}. Must be '{TRAIN_LEVEL_DOMAIN}'"
                f" or '{TRAIN_LEVEL_INTENT}'"
            )

    @staticmethod
    def _get_indices_per_label(unique_labels: List[str], class_labels: List[str]):
        """Gets a mapping between a unique class label and a shuffled list of query indices
        with that label.
        Args:
            unique_labels (List[str]): A list of unique class labels.
            class_labels (List[str]): list of labels for classification task. Labels follow
                the format of "domain" or "domain|intent". For example, "date|get_date".
        Returns:
            indices_per_label (Dict[str, List[int]]): A mapping between a unique class label
                and a shuffled list of query indices with that label.
        """
        indices_per_label = {}
        for label in unique_labels:
            indices_per_label[label] = [
                i for i in range(len(class_labels)) if class_labels[i] == label
            ]
            np.random.seed(ACTIVE_LEARNING_RANDOM_SEED)
            np.random.shuffle(indices_per_label[label])
        return indices_per_label


class RandomSampling(Heuristic):
    """ Selection strategy that randomly selects queries."""

    def _extractor(
        self, sampling_size: int, preds_single: List[List[float]] = None, **kwargs
    ):
        """
        Args:
            sampling_size (int): Size of data being sampled in a turn.
            preds_single: is a list[list[float]] and contains probability scores for each data
                point for each class.
        Returns:
            sampled_indices (list): List of indices that were selected.
            unsampled_indices (list): List of indices that were not selected.
        """
        sampling_size = Heuristic.validate_sampling_size(sampling_size)
        num_indicies = (
            len(preds_single) if preds_single else len(kwargs.get("unsampled"))
        )
        idxs = np.arange(num_indicies)
        np.random.shuffle(idxs)
        sampled_indices, unsampled_indices = (
            idxs[:sampling_size].tolist(),
            idxs[sampling_size:].tolist(),
        )
        return sampled_indices, unsampled_indices, None


class LeastConfidenceSampling(Heuristic):
    """ Selection strategy that select queries with the lowest max confidence."""

    def _extractor(
        self,
        sampling_size: int,
        preds_single: List[List[float]],
        do_rank: bool = True,
        **kwargs,
    ):
        """
        Args:
            sampling_size (int): Size of data being sampled in a turn.
            preds_single: is a list[list[float]] and contains probability scores for each data
                point for each class.
            do_rank: if True, returns a ranked list of the indices for confidence-sorting of data
                samples.
        """
        sampling_size = Heuristic.validate_sampling_size(sampling_size)
        confidences = np.max(preds_single, axis=1)
        idxs = custom_reordering(
            confidences, do_rank=do_rank, sampling_size=sampling_size
        )
        sampled_indices, unsampled_indices, ranked_indices = (
            idxs[:sampling_size].tolist(),
            idxs[sampling_size:].tolist(),
            idxs.tolist() if do_rank else None,
        )
        return sampled_indices, unsampled_indices, ranked_indices


class MarginSampling(Heuristic):
    """Selection strategy that select queries with the greatest difference between the highest
    and second highest confidence value."""

    def _extractor(
        self,
        sampling_size: int,
        preds_single: List[List[float]],
        do_rank: bool = True,
        **kwargs,
    ):
        """
        Args:
            sampling_size (int): Size of data being sampled in a turn.
            preds_single: is a list[list[float]] and contains probability scores for each data
                point for each class.
            do_rank: if True, returns a ranked list of the indices for confidence-sorting of data
                samples.
        """
        sampling_size = Heuristic.validate_sampling_size(sampling_size)
        # partition along last axis by default
        partition_ = np.partition(preds_single, len(preds_single[0]) - 2)
        # diff between the highest two values
        confidences = np.abs(partition_[:, -1] - partition_[:, -2])
        idxs = custom_reordering(
            confidences, do_rank=do_rank, sampling_size=sampling_size
        )
        sampled_indices, unsampled_indices, ranked_indices = (
            idxs[:sampling_size].tolist(),
            idxs[sampling_size:].tolist(),
            idxs.tolist() if do_rank else None,
        )
        return sampled_indices, unsampled_indices, ranked_indices


class EntropySampling(Heuristic):
    """ Selection strategy that select queries with the highest entropy."""

    def _extractor(
        self,
        sampling_size: int,
        preds_single: List[List[float]],
        do_rank: bool = True,
        **kwargs,
    ):
        """
        Args:
            sampling_size (int): Size of data being sampled in a turn.
            preds_single: is a list[list[float]] and contains probability scores for each data
                point for each class.
            do_rank: if True, returns a ranked list of the indices for confidence-sorting of data
                samples.
        """
        sampling_size = Heuristic.validate_sampling_size(sampling_size)
        entropies = scipy_entropy(np.array(preds_single), axis=1, base=ENTROPY_LOG_BASE)
        confidences = -entropies
        idxs = custom_reordering(
            confidences, do_rank=do_rank, sampling_size=sampling_size
        )
        sampled_indices, unsampled_indices, ranked_indices = (
            idxs[:sampling_size].tolist(),
            idxs[sampling_size:].tolist(),
            idxs.tolist() if do_rank else None,
        )
        return sampled_indices, unsampled_indices, ranked_indices


class DisagreementSampling(Heuristic):
    """ Selection strategy that measures 'disagreement' across multiple classifiers."""

    def _extractor(
        self,
        sampling_size: int,
        preds_multi: List[List[List[float]]],
        do_rank: bool = True,
        **kwargs,
    ):
        """
        Args:
            sampling_size (int): Size of data being sampled in a turn.
            preds_multi: is a list of list[list[float]] and contains probability scores for each
                data point for each class.
            do_rank: if True, returns a ranked list of the indices for confidence-sorting of data
                samples.
        """
        sampling_size = Heuristic.validate_sampling_size(sampling_size)
        choices = []
        for pred in preds_multi:
            model_choices = np.argmax(pred, axis=1).tolist()
            choices.append(model_choices)
        # Transpose to make rows correspond to datapoints, and columns to models
        choices = np.array(choices).T
        confidences = [
            Counter(row.tolist()).most_common(1)[0][1] / len(row) for row in choices
        ]
        idxs = custom_reordering(
            confidences, do_rank=do_rank, sampling_size=sampling_size
        )
        sampled_indices, unsampled_indices, ranked_indices = (
            idxs[:sampling_size].tolist(),
            idxs[sampling_size:].tolist(),
            idxs.tolist() if do_rank else None,
        )
        return sampled_indices, unsampled_indices, ranked_indices


class EnsembleSampling(Heuristic):
    """ Selection strategy that combines other selection strategies using a ranked approach."""

    @staticmethod
    def get_sampling_methods():
        """
        Returns:
            sampling_methods (Dict): Dict that maps a strategy abbreviation to instance.
        """
        return {
            "lcs": LeastConfidenceSampling(),
            "es": EntropySampling(),
            "ms": MarginSampling(),
            "ds": DisagreementSampling(),
            "kld": KLDivergenceSampling(),
        }

    def _extractor(
        self,
        sampling_size: int,
        preds_single: List[List[float]],
        preds_multi: List[List[List[float]]],
        **kwargs,
    ):
        """
        Args:
            sampling_size (int): Size of data being sampled in a turn.
            preds_single: is a list[list[float]] and contains probability scores for each data
                point for each class will be used to sample with lcs, es and ms.
            preds_multi: is a list of list[list[float]] and contains probability scores for each
                data point for each class will be used to sample with ds.
        """
        sampling_size = Heuristic.validate_sampling_size(sampling_size)
        sampling_methods = EnsembleSampling.get_sampling_methods()
        assert len(preds_single) != 0, print(f"{len(preds_single)}")
        if preds_multi:
            assert len(preds_single) == len(preds_multi[0]), print(
                f"{len(preds_single)}, {len(preds_multi)}"
            )
        qid2rank = {i: 0 for i in range(len(preds_single))}
        num_strategies_used = 0
        heuristic_params = {
            "sampling_size": sampling_size,
            "sampled": kwargs["sampled"],
            "unsampled": kwargs["unsampled"],
            "preds_single": preds_single,
            "preds_multi": preds_multi,
            "do_rank": True,
            "return_dict": True,
        }

        strategies = ["lcs", "ms", "es"]
        strategies = strategies + ["ds", "kld"] if preds_multi else strategies

        for strategy in strategies:
            ranked_inds = sampling_methods[strategy].sample(**heuristic_params)[
                "ranked_indices"
            ]
            for rank, qid in enumerate(ranked_inds):
                qid2rank[qid] += rank
            num_strategies_used += 1

        qid2rank = [(k, v / num_strategies_used) for k, v in qid2rank.items()]
        all_qids, all_ranks = list(zip(*qid2rank))

        reordered = custom_reordering(
            all_ranks, do_rank=False, sampling_size=sampling_size
        )
        ranked_indices = [all_qids[i] for i in reordered]
        sampled_indices, unsampled_indices = (
            ranked_indices[:sampling_size],
            ranked_indices[sampling_size:],
        )
        return sampled_indices, unsampled_indices, ranked_indices


class KLDivergenceSampling(Heuristic):
    """Selection strategy that calculates the KL divergence between the posterior
    distribution of multiple classifiers."""

    def _extractor(
        self,
        sampling_size: int,
        preds_multi: List[List[List[float]]],
        do_rank: bool = True,
        domain_indices: Dict = None,
        **kwargs,
    ):
        """
        Args:
            preds_multi (List[List[float]]): Probability scores for each data point for each
                class from multiple classifiers.
            do_rank: if True, returns a ranked list of the indices for confidence-sorting of data
                samples.
            domain_indices (Dict[str, tuple(int, int)]): A mapping between domains (str) to the
                corresponding indices in the probability vector. Used for intent-level KLD.
        """
        sampling_size = Heuristic.validate_sampling_size(sampling_size)

        # Calculate average prediction values.
        avg_preds = None
        for pred in preds_multi:
            if avg_preds is None:
                avg_preds = np.array(pred, dtype=np.float64)
            else:
                avg_preds += np.array(pred, dtype=np.float64)
        avg_preds /= len(preds_multi)
        # Estimate divergence.
        divergences = None
        if not domain_indices:
            divergences = KLDivergenceSampling.get_divergence_all_domains(
                preds_multi, avg_preds
            )
        else:
            divergences = KLDivergenceSampling.get_divergences_within_domain(
                preds_multi, avg_preds, domain_indices
            )
        # after transpose: row -> data point, column -> model
        divergences = np.array(divergences).T
        confidences = -np.max(divergences, axis=1)
        idxs = custom_reordering(
            confidences, do_rank=do_rank, sampling_size=sampling_size
        )
        sampled_indices, unsampled_indices, ranked_indices = (
            idxs[:sampling_size].tolist(),
            idxs[sampling_size:].tolist(),
            idxs.tolist() if do_rank else None,
        )
        return sampled_indices, unsampled_indices, ranked_indices

    @staticmethod
    def get_divergence_all_domains(preds_multi, avg_preds):
        """
        Args:
            preds_multi (List[List[float]]): Probability scores for each data point for each
                class from multiple classifiers.
            avg_preds (List[float]): Average probability vector across all intents.
        Returns:
            divergences: List of divergence values for each query compared to the average
                within-intent probability distribution.
        """
        divergences = []
        for pred in preds_multi:
            kldivergences = scipy_entropy(
                np.array(pred), avg_preds, axis=1, base=ENTROPY_LOG_BASE
            )
            divergences.append(kldivergences.tolist())
        return divergences

    @staticmethod
    def get_divergences_within_domain(preds_multi, avg_preds, domain_indices):
        """
        Args:
            preds_multi (List[List[float]]): Probability scores for each data point for each
                class from multiple classifiers.
            avg_preds (List[float]): Average probability vector across all intents.
            domain_indices (Dict[str, tuple(int, int)]): A mapping between domains (str) to the
                corresponding indices in the probability vector. Used for intent-level KLD.
        Returns:
            divergences: List of divergence values for each query compared to the average
                within-intent probability distribution.
        """
        divergences = []
        # Calculate q_d
        q_d = {d: [] for d in domain_indices}
        for row in avg_preds:
            domain = KLDivergenceSampling.get_domain(domain_indices, row)
            q_d[domain].append(row)
        for pred in preds_multi:
            # Calculate p_d
            p_d = {d: [] for d in domain_indices}
            for row in pred:
                domain = KLDivergenceSampling.get_domain(domain_indices, row)
                p_d[domain].append(row)
            # Calculate Divergence Scores by Domain
            divergence_scores_by_domain = {d: [] for d in domain_indices}
            for domain in domain_indices:
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
            domain_counter = {d: 0 for d in domain_indices}
            for row in pred:
                domain = KLDivergenceSampling.get_domain(domain_indices, row)
                single_pred_divergence_scores.append(
                    divergence_scores_by_domain[domain][domain_counter[domain]]
                )
                domain_counter[domain] += 1
            divergences.append(single_pred_divergence_scores)
        return divergences

    @staticmethod
    def get_domain(domain_indices, row):
        """Get the domain for a given probability row, inferred based on the non-zero values.
        Args:
            preds_multi (List[List[float]]): Probability scores for each data point for each
                class from multiple classifiers.
            avg_preds (List[float]): Average probability vector across all intents.
            domain_indices (Dict[str, tuple(int, int)]): A mapping between domains (str) to the
                corresponding indices in the probability vector. Used for intent-level KLD.
            row (List[List[float]]): A single row representing a queries probability distrubition.
        Returns:
            domain (str): The domain that the given row belongs to.
        Raises:
            AssertionError: If a row does not have an associated domain.
        """
        if np.sum(row) == 0:
            row = [1 / len(row)] * len(row)
        for domain in domain_indices:
            start, end = domain_indices[domain]
            if sum(row[start : end + 1]) > 0:
                return domain
        raise AssertionError(f"Row does not have an associated domain, Row: {row}")


class HeuristicsFactory:
    """Heuristics Factory Class"""

    @staticmethod
    def get_heuristic(heuristic):
        """A static method to get a translator

        Args:
            heuristic (str): Name of the desired Heuristic class
        Returns:
            (Heuristic): Heuristic Class
        """
        heuristic_classes = [
            StrategicRandomSampling,
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
        raise AssertionError(f" {heuristic} is not a valid 'heuristic'.")