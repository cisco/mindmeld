from abc import ABC, abstractmethod
from typing import List, Dict
from collections import Counter
from scipy.stats import entropy as scipy_entropy
import numpy as np

from ..constants import ENTROPY_LOG_BASE, ACTIVE_LEARNING_RANDOM_SEED


def custom_reordering(confidences, do_rank: bool = False, sampling_size: int = None):
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
        if sampling_size is not None:
            print(
                f"not using the value of sampling_size={sampling_size} since do_rank={do_rank}"
            )
        # ascending order sort
        idxs = np.argsort(confidences)
    return idxs


class Heuristic(ABC):
    def __init__(self, sampling_size):
        self.sampling_size = sampling_size
        self.check_sampling_size()

    def check_sampling_size(self):
        """Verify the sampling size is a positive integer"""
        assert isinstance(self.sampling_size, int)
        assert self.sampling_size > 0, print(
            f"Current Sampling Size: {self.sampling_size}, Required size > 0"
        )

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
    def __init__(self, sampling_size):
        super().__init__(sampling_size=sampling_size)

    def _extractor(
        self,
        unsampled: List,
        min_per_label: int = None,
        label_type: bool = "intent",
        allow_below_min: bool = False,
        **kwargs,
    ):
        """
        Args:
            class_labels (list): list of labels for classification task
        """
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
        if len(evenly_sampled) > self.sampling_size:
            np.random.shuffle(evenly_sampled)
            sampled_indices = evenly_sampled[: self.sampling_size]
            unsampled_indices = (
                evenly_sampled[self.sampling_size :] + outstanding_sampled
            )
        else:
            num_left_to_sample = self.sampling_size - len(evenly_sampled)
            outstanding_sampled_subset = np.random.choice(
                outstanding_sampled, num_left_to_sample, replace=False
            )
            sampled_indices = list(evenly_sampled) + list(outstanding_sampled_subset)
            unsampled_indices = [
                idx for idx, _ in enumerate(class_labels) if idx not in sampled_indices
            ]
        assert len(sampled_indices) + len(unsampled_indices) == len(class_labels)
        ranked_indices = None
        return sampled_indices, unsampled_indices, ranked_indices

    @staticmethod
    def _get_class_labels(label_type: str, unsampled: List):
        if label_type == "domain":
            return [f"{q.domain}" for q in unsampled]
        elif label_type == "intent":
            return [f"{q.domain}|{q.intent}" for q in unsampled]
        else:
            raise ValueError(
                f"Invalid label_type {label_type}. Must be 'domain' or 'intent'"
            )

    @staticmethod
    def _get_indices_per_label(unique_labels: List, class_labels: List):
        indices_per_label = {}
        for label in unique_labels:
            indices_per_label[label] = [
                i for i in range(len(class_labels)) if class_labels[i] == label
            ]
            np.random.seed(ACTIVE_LEARNING_RANDOM_SEED)
            np.random.shuffle(indices_per_label[label])
        return indices_per_label


class RandomSampling(Heuristic):
    def __init__(self, sampling_size):
        super().__init__(sampling_size=sampling_size)

    def _extractor(self, preds_single: List[List[float]] = None, **kwargs):
        """
        Args:
            preds_single: is a list[list[float]] and contains probability scores for each data
                point for each class.
        """
        num_indicies = (
            len(preds_single) if preds_single else len(kwargs.get("unsampled"))
        )
        idxs = np.arange(num_indicies)
        np.random.shuffle(idxs)
        sampled_indices, unsampled_indices, ranked_indices = (
            idxs[: self.sampling_size].tolist(),
            idxs[self.sampling_size :].tolist(),
            None,
        )
        return sampled_indices, unsampled_indices, ranked_indices


class LeastConfidenceSampling(Heuristic):
    def __init__(self, sampling_size):
        super().__init__(sampling_size=sampling_size)

    def _extractor(
        self, preds_single: List[List[float]], do_rank: bool = False, **kwargs
    ):
        """
        Args:
            preds_single: is a list[list[float]] and contains probability scores for each data
                point for each class.
            do_rank: if True, returns a ranked list of the indices for confidence-sorting of data
                samples.
        """
        confidences = np.max(preds_single, axis=1)
        idxs = custom_reordering(
            confidences, do_rank=do_rank, sampling_size=self.sampling_size
        )
        sampled_indices, unsampled_indices, ranked_indices = (
            idxs[: self.sampling_size].tolist(),
            idxs[self.sampling_size :].tolist(),
            idxs.tolist() if do_rank else None,
        )
        return sampled_indices, unsampled_indices, ranked_indices


class MarginSampling(Heuristic):
    def __init__(self, sampling_size):
        super().__init__(sampling_size=sampling_size)

    def _extractor(
        self, preds_single: List[List[float]], do_rank: bool = False, **kwargs
    ):
        """
        Args:
            preds_single: is a list[list[float]] and contains probability scores for each data
                point for each class.
            do_rank: if True, returns a ranked list of the indices for confidence-sorting of data
                samples.
        """
        # partition along last axis by default
        partition_ = np.partition(preds_single, len(preds_single[0]) - 2)
        # diff between the highest two values
        confidences = np.abs(partition_[:, -1] - partition_[:, -2])
        idxs = custom_reordering(
            confidences, do_rank=do_rank, sampling_size=self.sampling_size
        )
        sampled_indices, unsampled_indices, ranked_indices = (
            idxs[: self.sampling_size].tolist(),
            idxs[self.sampling_size :].tolist(),
            idxs.tolist() if do_rank else None,
        )
        return sampled_indices, unsampled_indices, ranked_indices


class EntropySampling(Heuristic):
    def __init__(self, sampling_size):
        super().__init__(sampling_size=sampling_size)

    def _extractor(
        self, preds_single: List[List[float]], do_rank: bool = False, **kwargs
    ):
        """
        Args:
            preds_single: is a list[list[float]] and contains probability scores for each data
                point for each class.
            do_rank: if True, returns a ranked list of the indices for confidence-sorting of data
                samples.
        """
        entropies = scipy_entropy(np.array(preds_single), axis=1, base=ENTROPY_LOG_BASE)
        confidences = -entropies
        idxs = custom_reordering(
            confidences, do_rank=do_rank, sampling_size=self.sampling_size
        )
        sampled_indices, unsampled_indices, ranked_indices = (
            idxs[: self.sampling_size].tolist(),
            idxs[self.sampling_size :].tolist(),
            idxs.tolist() if do_rank else None,
        )
        return sampled_indices, unsampled_indices, ranked_indices


class DisagreementSampling(Heuristic):
    def __init__(self, sampling_size):
        super().__init__(sampling_size=sampling_size)

    def _extractor(
        self, preds_multi: List[List[List[float]]], do_rank: bool = False, **kwargs
    ):
        """
        Args:
            preds_multi: is a list of list[list[float]] and contains probability scores for each
                data point for each class.
            do_rank: if True, returns a ranked list of the indices for confidence-sorting of data
                samples.
        """
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
            confidences, do_rank=do_rank, sampling_size=self.sampling_size
        )
        sampled_indices, unsampled_indices, ranked_indices = (
            idxs[: self.sampling_size].tolist(),
            idxs[self.sampling_size :].tolist(),
            idxs.tolist() if do_rank else None,
        )
        return sampled_indices, unsampled_indices, ranked_indices


class EnsembleSampling(Heuristic):
    def __init__(self, sampling_size):
        super().__init__(sampling_size=sampling_size)
        self.sampling_methods = {
            "lcs": LeastConfidenceSampling(self.sampling_size),
            "es": EntropySampling(self.sampling_size),
            "ms": MarginSampling(self.sampling_size),
            "ds": DisagreementSampling(self.sampling_size),
        }

    def _extractor(
        self,
        preds_single: List[List[float]],
        preds_multi: List[List[List[float]]],
        **kwargs,
    ):
        """
        Args:
            preds_single: is a list[list[float]] and contains probability scores for each data
                point for each class will be used to sample with lcs, es and ms.
            preds_multi: is a list of list[list[float]] and contains probability scores for each
                data point for each class will be used to sample with ds.
        """
        assert len(preds_single) != 0, print(f"{len(preds_single)}")
        if preds_multi:
            assert len(preds_single) == len(preds_multi[0]), print(
                f"{len(preds_single)}, {len(preds_multi)}"
            )
        qid2rank = {i: 0 for i in range(len(preds_single))}
        num_strategies_used = 0
        heuristic_params = {
            "sampled": kwargs["sampled"],
            "unsampled": kwargs["unsampled"],
            "preds_single": preds_single,
            "preds_multi": preds_multi,
            "do_rank": True,
            "return_dict": True,
        }
        if preds_single:
            ranked_inds = self.sampling_methods["lcs"].sample(**heuristic_params)[
                "ranked_indices"
            ]
            for rank, qid in enumerate(ranked_inds):
                qid2rank[qid] += rank
            num_strategies_used += 1

            ranked_inds = self.sampling_methods["ms"].sample(**heuristic_params)[
                "ranked_indices"
            ]
            for rank, qid in enumerate(ranked_inds):
                qid2rank[qid] += rank
            num_strategies_used += 1
            ranked_inds = self.sampling_methods["es"].sample(**heuristic_params)[
                "ranked_indices"
            ]
            for rank, qid in enumerate(ranked_inds):
                qid2rank[qid] += rank
            num_strategies_used += 1
        if preds_multi:
            ranked_inds = self.sampling_methods["ds"].sample(**heuristic_params)[
                "ranked_indices"
            ]
            for rank, qid in enumerate(ranked_inds):
                qid2rank[qid] += rank
            num_strategies_used += 1
        qid2rank = [(k, v / num_strategies_used) for k, v in qid2rank.items()]
        all_qids, all_ranks = list(zip(*qid2rank))
        # a higher rank now indicates higher confidence
        # do_rank can also be True here!
        reordered = custom_reordering(
            all_ranks, do_rank=False, sampling_size=self.sampling_size
        )
        ranked_indices = [all_qids[i] for i in reordered]
        sampled_indices, unsampled_indices = (
            ranked_indices[: self.sampling_size],
            ranked_indices[self.sampling_size :],
        )
        return sampled_indices, unsampled_indices, ranked_indices


class KLDivergenceSampling(Heuristic):
    def __init__(self, sampling_size):
        super().__init__(sampling_size=sampling_size)

    def _extractor(
        self,
        preds_multi: List[List[List[float]]],
        do_rank: bool = False,
        domain_indices: Dict = None,
        **kwargs,
    ):
        """
        Args:
            preds: is a list of list[list[float]] and contains probability scores for each data
                point for each class.
            do_rank: if True, returns a ranked list of the indices for confidence-sorting of data
                samples.
        """
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
            divergences = KLDivergenceSampling.get_divergence_all_domains(preds_multi, avg_preds)
        else:
            divergences = KLDivergenceSampling.get_divergences_within_domain(
                preds_multi, avg_preds, domain_indices
            )
        # after transpose: row -> data point, column -> model
        divergences = np.array(divergences).T
        confidences = -np.max(divergences, axis=1)
        idxs = custom_reordering(
            confidences, do_rank=do_rank, sampling_size=self.sampling_size
        )
        sampled_indices, unsampled_indices, ranked_indices = (
            idxs[: self.sampling_size].tolist(),
            idxs[self.sampling_size :].tolist(),
            idxs.tolist() if do_rank else None,
        )
        return sampled_indices, unsampled_indices, ranked_indices

    @staticmethod
    def get_divergence_all_domains(preds_multi, avg_preds):
        divergences = []
        for pred in preds_multi:
            kldivergences = scipy_entropy(
                np.array(pred), avg_preds, axis=1, base=ENTROPY_LOG_BASE
            )
            divergences.append(kldivergences.tolist())
        return divergences

    @staticmethod
    def get_divergences_within_domain(preds_multi, avg_preds, domain_indices):
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
        if np.sum(row) == 0:
            row = [1 / len(row)] * len(row)
        for domain in domain_indices:
            start, end = domain_indices[domain]
            if sum(row[start : end + 1]) > 0:
                return domain
        raise (f"Row does not have an associated domain, Row: {row}")


class HeuristicsFactory:
    """Heuristics Factory Class"""

    @staticmethod
    def get_heuristic(heuristic, sampling_size):
        """A static method to get a translator

        Args:
            heuristic (str): Name of the desired Heuristic class
            sampling_size (int): Heuristic sample size
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
                return heuristic_class(sampling_size=sampling_size)
        raise AssertionError(f" {heuristic} is not a valid 'heuristic'.")
