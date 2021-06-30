#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test Heuristics
----------------------------------

Tests for Heuristics in the `active_learning.heuristics` module.
"""

from mindmeld.active_learning.heuristics import (
    stratified_random_sample,
    Heuristic,
    LeastConfidenceSampling,
    MarginSampling,
    EntropySampling,
    EnsembleSampling,
    DisagreementSampling,
    KLDivergenceSampling,
)

example = [
    [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]],
    [[0.1, 0.9], [0.4, 0.6], [0.75, 0.25]],
    [[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]],
]


def test_stratified_random_sample():
    labels = ["R", "B", "C", "C", "B", "R", "R", "R", "C", "B", "B", "B", "R"]
    selected_indices = stratified_random_sample(labels)
    sampled_labels = [labels[i] for i in selected_indices]
    expected_labels = ["R", "B", "C", "R", "B", "C", "R", "B", "C", "R", "B", "B", "R"]
    assert sampled_labels == expected_labels


def test_convert_to_sample_ranks():
    ordered_indices = [3, 0, 2, 4, 1]
    expected_sample_ranks = [1, 4, 2, 0, 3]
    predicted_sample_ranks = list(Heuristic._convert_to_sample_ranks(ordered_indices))
    assert expected_sample_ranks == predicted_sample_ranks


def test_ordered_indices_list_to_final_rank():
    ordered_indices_list = [[3, 0, 2, 4, 1], [0, 2, 1, 3, 4], [4, 3, 1, 2, 0]]
    """
    Sample ranks for each example:
    0th example [1 4 2 0 3]
    1st example [0 2 1 3 4]
    2nd example [4 2 3 1 0]

    sum         [5 8 6 4 7]

    argsort of above --> [3, 0, 2, 4, 1]
    """
    expected_sample_ranks = [3, 0, 2, 4, 1]
    predicted_sample_ranks = Heuristic.ordered_indices_list_to_final_rank(
        ordered_indices_list
    )
    assert expected_sample_ranks == predicted_sample_ranks


def test_least_confidence_sampling():
    """
    max confidence for each sample --> [[0.9,0.5,0.8],[0.9,0.6,0.75],[0.8,0.6,0.9]]
    argsort of max confidences -->[[1,2,0],[1,2,0],[1,0,2]]
    """
    expected_LC_sampling_2d_result = [[1, 2, 0], [1, 2, 0], [1, 0, 2]]
    """
    Sum of ranks for each example:
    0th example (2+2+1) = 5
    1st example (0+0+0) = 0
    2nd example (1+1+2) = 4

    argsort of above --> [1,2,0]
    """
    expected_LC_sampling_3d_result = [1, 2, 0]
    predicted_LC_sampling_2d_result = [
        LeastConfidenceSampling.rank_2d(c) for c in example
    ]
    predicted_LC_sampling_3d_result = LeastConfidenceSampling.rank_3d(example)

    assert expected_LC_sampling_2d_result == predicted_LC_sampling_2d_result
    assert expected_LC_sampling_3d_result == predicted_LC_sampling_3d_result


def test_margin_sampling():
    """
    margin for each sample --> [[0.8,0,0.6],[0.8,0.2,0.5],[0.6,0.2,0.8]]
    argsort of lowest to highest margin -->[[1,2,0],[1,2,0],[1,0,2]]
    """
    expected_margin_sampling_2d_result = [[1, 2, 0], [1, 2, 0], [1, 0, 2]]
    """
    Sum of ranks for each example:
    0th example (2+2+1) = 5
    1st example (0+0+0) = 0
    2nd example (1+1+2) = 4

    argsort of above --> [1,2,0]
    """
    expected_margin_sampling_3d_result = [1, 2, 0]
    predicted_margin_sampling_2d_result = [MarginSampling.rank_2d(c) for c in example]
    predicted_margin_sampling_3d_result = MarginSampling.rank_3d(example)

    assert expected_margin_sampling_2d_result == predicted_margin_sampling_2d_result
    assert expected_margin_sampling_3d_result == predicted_margin_sampling_3d_result


def test_entropy_sampling():
    """
    entropy for each sample --> [[0.47,1.0,0.72],[0.47,0.97,0.81],[0.72,0.97,0.47]]
    argsort of highest to lowest entropy -->[[1,2,0], [1,2,0], [1,0,2]]
    """
    expected_entropy_sampling_2d_result = [[1, 2, 0], [1, 2, 0], [1, 0, 2]]
    """
    Sum of ranks for each example:
    0th example (2+2+1) = 5
    1st example (0+0+0) = 0
    2nd example (1+1+2) = 2

    argsort of above --> [1,2,0]
    """
    expected_entropy_sampling_3d_result = [1, 2, 0]
    predicted_entropy_sampling_2d_result = [EntropySampling.rank_2d(c) for c in example]
    predicted_entropy_sampling_3d_result = EntropySampling.rank_3d(example)

    assert expected_entropy_sampling_2d_result == predicted_entropy_sampling_2d_result
    assert expected_entropy_sampling_3d_result == predicted_entropy_sampling_3d_result


def test_disagreement_sampling():
    """
    disagreement score for each sample --> [0, 0.33, 0]
    argsort of highest to lowest entropy -->[0,2,1]
    """
    expected_disagreement_sampling_3d_result = [0, 2, 1]
    predicted_disagreement_sampling_3d_result = DisagreementSampling.rank_3d(example)
    assert (
        expected_disagreement_sampling_3d_result
        == predicted_disagreement_sampling_3d_result
    )


def test_kl_divergence_sampling():
    """
    Max KL divergence score for each sample --> [0.0246 0.029 0.0387]
    argsort of highest to lowest KL divergence -->[2,1,0]
    """
    expected_kl_divergence_sampling_3d_result = [2, 1, 0]
    predicted_kl_divergence_sampling_3d_result = KLDivergenceSampling.rank_3d(example)
    assert (
        expected_kl_divergence_sampling_3d_result
        == predicted_kl_divergence_sampling_3d_result
    )


def test_ensemble_sampling():
    example = [
        [[0.1, 0.9], [0.5, 0.5], [0.8, 0.2]],
        [[0.1, 0.9], [0.4, 0.6], [0.75, 0.25]],
        [[0.2, 0.8], [0.6, 0.4], [0.9, 0.1]],
    ]

    expected_ensemble_sampling_2d_result = [[1, 2, 0], [1, 2, 0], [1, 0, 2]]
    expected_ensemble_sampling_3d_result = [1, 2, 0]
    predicted_ensemble_sampling_2d_result = [
        EnsembleSampling.rank_2d(c) for c in example
    ]
    predicted_ensemble_sampling_3d_result = EnsembleSampling.rank_3d(example)

    assert expected_ensemble_sampling_2d_result == predicted_ensemble_sampling_2d_result
    assert expected_ensemble_sampling_3d_result == predicted_ensemble_sampling_3d_result
