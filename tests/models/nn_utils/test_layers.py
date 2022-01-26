#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Tests for `layers` submodule of nn_utils
"""

import numpy as np
import pytest

from mindmeld.models.nn_utils.layers import (
    EmbeddingLayer,
    CnnLayer,
    LstmLayer,
    PoolingLayer,
    SplittingAndPoolingLayer
)


@pytest.mark.extras
@pytest.mark.torch
def test_embedding_layer():
    import torch
    from torch.nn.utils.rnn import pad_sequence

    kwargs = {
        "num_tokens": 5,
        "emb_dim": 10,
        "padding_idx": 2,
    }
    seq_ids = [[1, 2, 3], [0, 1, 1, 2, 4, 0]]

    # check forward method of the embedding layer
    emb_layer = EmbeddingLayer(**kwargs)
    emb_layer.eval()
    with torch.no_grad():
        inputs = pad_sequence(
            [torch.tensor(seq) for seq in seq_ids],
            padding_value=kwargs["padding_idx"],
            batch_first=True)
        outputs = emb_layer(inputs)
        assert outputs.shape == (2, 6, 10)
        assert outputs[0][1][0].item() == 0.0
        assert sum(outputs[0][-1]).item() == 0.0

    # check if inputted embedding_weights are used for the embedding layer initialization
    embedding_weights = np.random.random((5, 10))
    kwargs = {**kwargs, "embedding_weights": embedding_weights}
    emb_layer = EmbeddingLayer(**kwargs)
    emb_layer.eval()
    with torch.no_grad():
        inputs = pad_sequence(
            [torch.tensor(seq) for seq in seq_ids],
            padding_value=kwargs["padding_idx"],
            batch_first=True)
        outputs = emb_layer(inputs)
        assert outputs[0][0][-1] == embedding_weights[1][-1]


@pytest.mark.extras
@pytest.mark.torch
def test_cnn_layer():
    import torch

    kwargs = {
        "emb_dim": 10,
        "kernel_sizes": [2, 3],
        "num_kernels": [5, 6],
    }

    # check forward method of the cnn layer
    cnn_layer = CnnLayer(**kwargs)
    cnn_layer.eval()
    with torch.no_grad():
        batch_size, max_seq_len = 4, 15
        inputs = torch.randn((batch_size, max_seq_len, kwargs["emb_dim"]))
        outputs = cnn_layer(inputs)
        assert len(outputs.shape) == 2
        assert outputs.shape == (batch_size, sum(kwargs["num_kernels"]))


@pytest.mark.extras
@pytest.mark.torch
def test_lstm_layer():
    import torch

    kwargs = {
        "emb_dim": 10,
        "hidden_dim": 5,
        "num_layers": 2,
        "lstm_dropout": 0.3,
        "bidirectional": True,
    }

    # check forward method of the lstm layer
    lstm_layer = LstmLayer(**kwargs)
    lstm_layer.eval()
    with torch.no_grad():
        batch_size, max_seq_len = 4, 15
        input_lengths = torch.randint(2, max_seq_len, (batch_size,))
        max_seq_len = max(input_lengths)
        inputs = torch.randn((batch_size, max_seq_len, kwargs["emb_dim"]))
        outputs = lstm_layer(inputs, input_lengths)
        assert len(outputs.shape) == 3
        assert outputs.shape == (batch_size, max_seq_len, 2 * kwargs["hidden_dim"])

        if min(input_lengths) != max(input_lengths):  # not all same lengths
            batch_dim_pointer1 = np.argmin(input_lengths)
            batch_dim_pointer2 = np.argmax(input_lengths)
            assert all(outputs[batch_dim_pointer1][-1] == torch.zeros((2 * kwargs["hidden_dim"],)))
            assert any(outputs[batch_dim_pointer2][-1] != torch.zeros((2 * kwargs["hidden_dim"],)))

    kwargs.update({
        "bidirectional": False,
    })

    lstm_layer = LstmLayer(**kwargs)
    lstm_layer.eval()
    with torch.no_grad():
        batch_size, max_seq_len = 4, 15
        input_lengths = torch.randint(2, max_seq_len, (batch_size,))
        max_seq_len = max(input_lengths)
        inputs = torch.randn((batch_size, max_seq_len, kwargs["emb_dim"]))
        outputs = lstm_layer(inputs, input_lengths)
        assert len(outputs.shape) == 3
        assert outputs.shape == (batch_size, max_seq_len, kwargs["hidden_dim"])

        if min(input_lengths) != max(input_lengths):  # not all same lengths
            batch_dim_pointer1 = np.argmin(input_lengths)
            batch_dim_pointer2 = np.argmax(input_lengths)
            assert all(outputs[batch_dim_pointer1][-1] == torch.zeros((kwargs["hidden_dim"],)))
            assert any(outputs[batch_dim_pointer2][-1] != torch.zeros((kwargs["hidden_dim"],)))


@pytest.mark.extras
@pytest.mark.torch
def test_pooling_layer():
    import torch

    # check forward method of the pooling layer for different pooling types

    kwargs = {
        "pooling_type": "first",
    }
    pooling_layer = PoolingLayer(**kwargs)
    pooling_layer.eval()
    with torch.no_grad():
        batch_size, max_seq_len, emb_dim = 4, 15, 10
        input_lengths = None
        inputs = torch.randn((batch_size, max_seq_len, emb_dim))
        outputs = pooling_layer(inputs, input_lengths)
        assert len(outputs.shape) == 2
        assert outputs.shape == (batch_size, emb_dim)
        assert all(outputs.reshape(-1, ) == inputs[:, 0, :].reshape(-1, ))

    kwargs = {
        "pooling_type": "last",
    }
    pooling_layer = PoolingLayer(**kwargs)
    pooling_layer.eval()
    with torch.no_grad():
        batch_size, max_seq_len, emb_dim = 4, 15, 10
        input_lengths = torch.randint(2, max_seq_len, (batch_size,))
        max_seq_len = max(input_lengths)
        inputs = torch.randn((batch_size, max_seq_len, emb_dim))
        outputs = pooling_layer(inputs, input_lengths)
        assert len(outputs.shape) == 2
        assert outputs.shape == (batch_size, emb_dim)
        for i, _output in enumerate(outputs):
            assert all(inputs[i, input_lengths[i] - 1] == _output)

        with pytest.raises(ValueError):
            _ = pooling_layer(inputs, None)

    kwargs = {
        "pooling_type": "max",
    }
    pooling_layer = PoolingLayer(**kwargs)
    pooling_layer.eval()
    with torch.no_grad():
        batch_size, max_seq_len, emb_dim = 4, 15, 10
        input_lengths = torch.randint(2, max_seq_len, (batch_size,))
        max_seq_len = max(input_lengths)
        inputs = torch.randn((batch_size, max_seq_len, emb_dim))
        outputs = pooling_layer(inputs, input_lengths)
        assert len(outputs.shape) == 2
        assert outputs.shape == (batch_size, emb_dim)
        for i, _output in enumerate(outputs):
            expected_output, _ = torch.max(inputs[i, :input_lengths[i]], dim=0)
            assert all(expected_output == _output)

    kwargs = {
        "pooling_type": "mean",
    }
    pooling_layer = PoolingLayer(**kwargs)
    pooling_layer.eval()
    with torch.no_grad():
        batch_size, max_seq_len, emb_dim = 4, 15, 10
        input_lengths = torch.randint(2, max_seq_len, (batch_size,))
        max_seq_len = max(input_lengths)
        inputs = torch.randn((batch_size, max_seq_len, emb_dim))
        outputs = pooling_layer(inputs, input_lengths)
        assert len(outputs.shape) == 2
        assert outputs.shape == (batch_size, emb_dim)
        for i, _output in enumerate(outputs):
            expected_output = torch.mean(inputs[i, :input_lengths[i]], dim=0)
            assert (expected_output - _output).pow(2).sum(-1).sqrt() < 1e-6


@pytest.mark.extras
@pytest.mark.torch
def test_splitting_and_pooling_layer():
    import torch

    # check forward method of the pooling layer for select pooling types; other types already
    # checked through the pooling layer testing

    kwargs = {
        "pooling_type": "first",
        "number_of_terminal_tokens": 2,
    }
    splitting_and_pooling_layer = SplittingAndPoolingLayer(**kwargs)
    splitting_and_pooling_layer.eval()
    with torch.no_grad():
        batch_size, emb_dim = 4, 10
        span_lengths = [
            torch.randint(1, 4, (n_sub_groups,))
            for n_sub_groups in torch.randint(1, 4, (batch_size,))
        ]
        max_sum_lengths = max([sum(_x) for _x in span_lengths])
        max_n_lengths = max([len(_x) for _x in span_lengths])

        # inputs w/ terminal token embeddings
        inputs = torch.randn((batch_size, max_sum_lengths + 2, emb_dim))
        outputs = splitting_and_pooling_layer(inputs, span_lengths, discard_terminals=True)
        assert len(outputs.shape) == 3
        assert outputs.shape == (batch_size, max_n_lengths, emb_dim)

        # inputs w/o terminal token embeddings
        inputs = torch.randn((batch_size, max_sum_lengths, emb_dim))
        outputs = splitting_and_pooling_layer(inputs, span_lengths, discard_terminals=False)
        assert len(outputs.shape) == 3
        assert outputs.shape == (batch_size, max_n_lengths, emb_dim)

        # check if padding done correctly
        for i, _span_lengths in enumerate(span_lengths):
            if len(_span_lengths) < max_n_lengths:
                assert all(outputs[i][-1] == torch.zeros((emb_dim,)))
            if len(_span_lengths) == max_n_lengths:
                assert any(outputs[i][-1] != torch.zeros((emb_dim,)))

        # check if splitting is done correctly
        for i, _span_lengths in enumerate(span_lengths):
            select_indices = [0, ] + [_x.item() for _x in np.cumsum(_span_lengths.numpy())[:-1]]
            for j, select_index in enumerate(select_indices):
                assert all(inputs[i][select_index] == outputs[i][j])

    kwargs = {
        "pooling_type": "last",
        "number_of_terminal_tokens": 2,
    }
    splitting_and_pooling_layer = SplittingAndPoolingLayer(**kwargs)
    splitting_and_pooling_layer.eval()
    with torch.no_grad():
        batch_size, emb_dim = 4, 10
        span_lengths = [
            torch.randint(1, 4, (n_sub_groups,))
            for n_sub_groups in torch.randint(1, 4, (batch_size,))
        ]
        max_sum_lengths = max([sum(_x) for _x in span_lengths])
        max_n_lengths = max([len(_x) for _x in span_lengths])

        # inputs w/ terminal token embeddings
        inputs = torch.randn((batch_size, max_sum_lengths + 2, emb_dim))
        outputs = splitting_and_pooling_layer(inputs, span_lengths, discard_terminals=True)
        assert len(outputs.shape) == 3
        assert outputs.shape == (batch_size, max_n_lengths, emb_dim)

        # inputs w/o terminal token embeddings
        inputs = torch.randn((batch_size, max_sum_lengths, emb_dim))
        outputs = splitting_and_pooling_layer(inputs, span_lengths, discard_terminals=False)
        assert len(outputs.shape) == 3
        assert outputs.shape == (batch_size, max_n_lengths, emb_dim)

        # check if padding done correctly
        for i, _span_lengths in enumerate(span_lengths):
            if len(_span_lengths) < max_n_lengths:
                assert all(outputs[i][-1] == torch.zeros((emb_dim,)))
            if len(_span_lengths) == max_n_lengths:
                assert any(outputs[i][-1] != torch.zeros((emb_dim,)))

        # check if splitting is done correctly
        for i, _span_lengths in enumerate(span_lengths):
            select_indices = [_x.item() - 1 for _x in np.cumsum(_span_lengths.numpy())]
            for j, select_index in enumerate(select_indices):
                assert all(inputs[i][select_index] == outputs[i][j])
