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
import logging
import math
import os
import re

import numpy as np
from sklearn.externals import joblib
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf

from .embeddings import CharacterSequenceEmbedding, WordSequenceEmbedding
from .taggers import Tagger, extract_sequence_features

DEFAULT_ENTITY_TOKEN_SPAN_INDEX = 2
GAZ_PATTERN_MATCH = r"in-gaz\|type:(\w+)\|pos:(\w+)\|"
REGEX_TYPE_POSITIONAL_INDEX = 1
DEFAULT_LABEL = "B|UNK"
DEFAULT_GAZ_LABEL = "O"
RANDOM_SEED = 1
ZERO_INITIALIZER_VALUE = 0

logger = logging.getLogger(__name__)


class LstmModel(Tagger):  # pylint: disable=too-many-instance-attributes
    """This class encapsulates the bi-directional LSTM model and provides
    the correct interface for use by the tagger model"""

    def fit(self, X, y):
        examples_arr = np.asarray(X, dtype="float32")
        labels_arr = np.asarray(y, dtype="int32")
        self._fit(examples_arr, labels_arr)
        return self

    def predict(self, X, dynamic_resource=None):
        encoded_examples_arr = np.asarray(X, dtype="float32")
        tags_by_example_arr = self._predict(encoded_examples_arr)

        resized_predicted_tags = []
        for query, seq_len in zip(tags_by_example_arr, self.sequence_lengths):
            resized_predicted_tags.append(query[:seq_len])

        return resized_predicted_tags

    def set_params(self, **parameters):
        """
        Initialize params for the LSTM. The keys in the parameters dictionary
        are as follows:

        Args:
            parameters (dict): The keys in the parameters dictionary are as follows:

            number_of_epochs: The number of epochs to run (int)

            batch_size: The batch size for mini-batch training (int)

            token_lstm_hidden_state_dimension: The hidden state
                dimension of the LSTM cell (int)

            learning_rate: The learning rate of the optimizer (int)

            optimizer: The optimizer used to train the network
                is the number of entities in the dataset (str)

            display_epoch: The number of epochs after which the
                network displays common stats like accuracy (int)

            padding_length: The length of each query, which is
                fixed, so some queries will be cut short in length
                representing the word embedding, the row index
                is the word's index (int)

            token_embedding_dimension: The embedding dimension of the word (int)

            token_pretrained_embedding_filepath: The pretrained embedding file-path (str)

            dense_keep_prob: The dropout rate of the dense layers (float)

            lstm_input_keep_prob: The dropout rate of the inputs to the LSTM cell (float)

            lstm_output_keep_prob: The dropout rate of the outputs of the LSTM cell (float)

            gaz_encoding_dimension: The gazetteer encoding dimension (int)
        """
        self.number_of_epochs = parameters.get("number_of_epochs", 20)
        self.batch_size = parameters.get("batch_size", 20)
        self.token_lstm_hidden_state_dimension = parameters.get(
            "token_lstm_hidden_state_dimension", 300
        )

        self.learning_rate = parameters.get("learning_rate", 0.005)
        self.optimizer_tf = parameters.get("optimizer", "adam")
        self.padding_length = parameters.get("padding_length", 20)
        self.display_epoch = parameters.get("display_epoch", 20)

        self.token_embedding_dimension = parameters.get(
            "token_embedding_dimension", 300
        )
        self.token_pretrained_embedding_filepath = parameters.get(
            "token_pretrained_embedding_filepath"
        )

        self.dense_keep_probability = parameters.get("dense_keep_prob", 0.5)
        self.lstm_input_keep_prob = parameters.get("lstm_input_keep_prob", 0.5)
        self.lstm_output_keep_prob = parameters.get("lstm_output_keep_prob", 0.5)
        self.gaz_encoding_dimension = parameters.get("gaz_encoding_dimension", 100)
        self.use_crf_layer = parameters.get("use_crf_layer", True)

        self.use_char_embeddings = parameters.get("use_character_embeddings", False)
        self.char_window_sizes = parameters.get("char_window_sizes", [5])
        self.max_char_per_word = parameters.get("maximum_characters_per_word", 20)
        self.character_embedding_dimension = parameters.get(
            "character_embedding_dimension", 10
        )
        self.word_level_character_embedding_size = parameters.get(
            "word_level_character_embedding_size", 40
        )

    def get_params(self, deep=True):
        return self.__dict__

    def construct_tf_variables(self):
        """
        Constructs the variables and operations in the TensorFlow session graph
        """
        with self.graph.as_default():
            self.dense_keep_prob_tf = tf.placeholder(
                tf.float32, name="dense_keep_prob_tf"
            )
            self.lstm_input_keep_prob_tf = tf.placeholder(
                tf.float32, name="lstm_input_keep_prob_tf"
            )
            self.lstm_output_keep_prob_tf = tf.placeholder(
                tf.float32, name="lstm_output_keep_prob_tf"
            )

            self.query_input_tf = tf.placeholder(
                tf.float32,
                [None, self.padding_length, self.token_embedding_dimension],
                name="query_input_tf",
            )

            self.gaz_input_tf = tf.placeholder(
                tf.float32,
                [None, self.padding_length, self.gaz_dimension],
                name="gaz_input_tf",
            )

            self.label_tf = tf.placeholder(
                tf.int32,
                [None, int(self.padding_length), self.output_dimension],
                name="label_tf",
            )

            self.batch_sequence_lengths_tf = tf.placeholder(
                tf.int32, shape=[None], name="batch_sequence_lengths_tf"
            )

            self.batch_sequence_mask_tf = tf.placeholder(
                tf.bool, shape=[None], name="batch_sequence_mask_tf"
            )

            if self.use_char_embeddings:
                self.char_input_tf = tf.placeholder(
                    tf.float32,
                    [
                        None,
                        self.padding_length,
                        self.max_char_per_word,
                        self.character_embedding_dimension,
                    ],
                    name="char_input_tf",
                )

            combined_embedding_tf = self._construct_embedding_network()
            self.lstm_output_tf = self._construct_lstm_network(combined_embedding_tf)
            self.lstm_output_softmax_tf = tf.nn.softmax(
                self.lstm_output_tf, name="output_softmax_tensor"
            )
            self.optimizer_tf, self.cost_tf = self._define_optimizer_and_cost()

            self.global_init = tf.global_variables_initializer()
            self.local_init = tf.local_variables_initializer()

            self.saver = tf.train.Saver()

    def extract_features(self, examples, config, resources, y=None, fit=True):
        """Transforms a list of examples into features that are then used by the
        deep learning model.

        Args:
            examples (list of mindmeld.core.Query): a list of queries
            config (ModelConfig): The ModelConfig which may contain information used for feature
                                  extraction
            resources (dict): Resources which may be used for this model's feature extraction
            y (list): A list of label sequences

        Returns:
            (sequence_embeddings, encoded_labels, groups): features for the LSTM network
        """
        del fit  # unused -- we use the value of y to determine whether to encode labels
        if y:
            # Train time
            self.resources = resources

            padded_y = self._pad_labels(y, DEFAULT_LABEL)
            y_flat = [item for sublist in padded_y for item in sublist]
            encoded_labels_flat = self.label_encoder.fit_transform(y_flat)
            encoded_labels = []

            start_index = 0
            for label_sequence in padded_y:
                encoded_labels.append(
                    encoded_labels_flat[start_index : start_index + len(label_sequence)]
                )
                start_index += len(label_sequence)

            gaz_entities = list(self.resources.get("gazetteers", {}).keys())
            gaz_entities.append(DEFAULT_GAZ_LABEL)
            self.gaz_encoder.fit(gaz_entities)

            # The gaz dimension are the sum total of the gazetteer entities and
            # the 'other' gaz entity, which is the entity for all non-gazetteer tokens
            self.gaz_dimension = len(gaz_entities)
            self.output_dimension = len(self.label_encoder.classes_)
        else:
            # Predict time
            encoded_labels = None

        # Extract features and classes
        (
            x_sequence_embeddings_arr,
            self.gaz_features_arr,
            self.char_features_arr,
        ) = self._get_features(examples)

        self.sequence_lengths = self._extract_seq_length(examples)

        # There are no groups in this model
        groups = None
        return x_sequence_embeddings_arr, encoded_labels, groups

    def setup_model(self, config):
        self.set_params(**config.params)
        self.label_encoder = LabelBinarizer()
        self.gaz_encoder = LabelBinarizer()

        self.graph = tf.Graph()
        self.saver = None

        self.example_type = config.example_type
        self.features = config.features

        self.query_encoder = WordSequenceEmbedding(
            self.padding_length,
            self.token_embedding_dimension,
            self.token_pretrained_embedding_filepath,
        )

        if self.use_char_embeddings:
            self.char_encoder = CharacterSequenceEmbedding(
                self.padding_length,
                self.character_embedding_dimension,
                self.max_char_per_word,
            )

    def construct_feed_dictionary(
        self, batch_examples, batch_char, batch_gaz, batch_seq_len, batch_labels=None
    ):
        """Constructs the feed dictionary that is used to feed data into the tensors

        Args:
            batch_examples (ndarray): A batch of examples
            batch_char (ndarray): A batch of character features
            batch_gaz (ndarray): A batch of gazetteer features
            batch_seq_len (ndarray): A batch of sequence length of each query
            batch_labels (ndarray): A batch of labels

        Returns:
            The feed dictionary
        """
        if batch_labels is None:
            batch_labels = []

        return_dict = {
            self.query_input_tf: batch_examples,
            self.batch_sequence_lengths_tf: batch_seq_len,
            self.gaz_input_tf: batch_gaz,
            self.dense_keep_prob_tf: self.dense_keep_probability,
            self.lstm_input_keep_prob_tf: self.lstm_input_keep_prob,
            self.lstm_output_keep_prob_tf: self.lstm_output_keep_prob,
            self.batch_sequence_mask_tf: self._generate_boolean_mask(batch_seq_len),
        }

        if len(batch_labels) > 0:
            return_dict[self.label_tf] = batch_labels

        if len(batch_char) > 0:
            return_dict[self.char_input_tf] = batch_char

        return return_dict

    def _construct_embedding_network(self):
        """Constructs a network based on the word embedding and gazetteer
        inputs and concatenates them together

        Returns:
            Combined embeddings of the word and gazetteer embeddings
        """
        initializer = tf.contrib.layers.xavier_initializer(seed=RANDOM_SEED)

        dense_gaz_embedding_tf = tf.contrib.layers.fully_connected(
            inputs=self.gaz_input_tf,
            num_outputs=self.gaz_encoding_dimension,
            weights_initializer=initializer,
        )

        batch_size_dim = tf.shape(self.query_input_tf)[0]

        if self.use_char_embeddings:
            word_level_char_embeddings_list = []

            for window_size in self.char_window_sizes:
                word_level_char_embeddings_list.append(
                    self.apply_convolution(
                        self.char_input_tf, batch_size_dim, window_size
                    )
                )

            word_level_char_embedding = tf.concat(word_level_char_embeddings_list, 2)

            # Combined the two embeddings
            combined_embedding_tf = tf.concat(
                [self.query_input_tf, word_level_char_embedding], axis=2
            )
        else:
            combined_embedding_tf = self.query_input_tf

        combined_embedding_tf = tf.concat(
            [combined_embedding_tf, dense_gaz_embedding_tf], axis=2
        )

        return combined_embedding_tf

    def apply_convolution(self, input_tensor, batch_size, char_window_size):
        """Constructs a convolution network of a specific window size

        Args:
            input_tensor (tensor): The input tensor to the network
            batch_size (int): The batch size of the training data
            char_window_size (int): The character window size of each stride

        Returns:
            (Tensor): Convolved output tensor
        """
        convolution_reshaped_char_embedding = tf.reshape(
            input_tensor,
            [
                -1,
                self.padding_length,
                self.max_char_per_word,
                self.character_embedding_dimension,
                1,
            ],
        )

        # Index 0 dimension is 1 because we want to apply this to every word. Index 1 dimension is
        # char_window_size since this is the convolution window size. Index 3 dimension is
        # 1 since the input channel is 1 dimensional (the sequence string). Index 4 dimension is
        # the output dimension which is a hyper-parameter.
        char_convolution_filter = tf.Variable(
            tf.random_normal(
                [
                    1,
                    char_window_size,
                    self.character_embedding_dimension,
                    1,
                    self.word_level_character_embedding_size,
                ],
                dtype=tf.float32,
            )
        )

        # Strides is None because we want to advance one character at a time and one word at a time
        conv_output = tf.nn.convolution(
            convolution_reshaped_char_embedding, char_convolution_filter, padding="SAME"
        )

        # Max pool over each word, captured by the size of the filter corresponding to an entire
        # single word
        max_pool = tf.nn.pool(
            conv_output,
            window_shape=[
                1,
                self.max_char_per_word,
                self.character_embedding_dimension,
            ],
            pooling_type="MAX",
            padding="VALID",
        )

        # Transpose because shape before is batch_size BY query_padding_length BY 1 BY 1
        # BY num_filters. This transform  rearranges the dimension of each rank such that
        # the num_filters dimension comes after the query_padding_length, so the last index
        # 4 is brought after the index 1.
        max_pool = tf.transpose(max_pool, [0, 1, 4, 2, 3])
        max_pool = tf.reshape(
            max_pool,
            [batch_size, self.padding_length, self.word_level_character_embedding_size],
        )

        char_convolution_bias = tf.Variable(
            tf.random_normal(
                [
                    self.word_level_character_embedding_size,
                ]
            )
        )

        char_convolution_bias = tf.tile(char_convolution_bias, [self.padding_length])
        char_convolution_bias = tf.reshape(
            char_convolution_bias,
            [self.padding_length, self.word_level_character_embedding_size],
        )

        char_convolution_bias = tf.tile(char_convolution_bias, [batch_size, 1])
        char_convolution_bias = tf.reshape(
            char_convolution_bias,
            [batch_size, self.padding_length, self.word_level_character_embedding_size],
        )

        word_level_char_embedding = tf.nn.relu(max_pool + char_convolution_bias)
        return word_level_char_embedding

    def _define_optimizer_and_cost(self):
        """This function defines the optimizer and cost function of the LSTM model

        Returns:
            AdamOptimizer, Tensor: The optimizer function to reduce loss and the loss values
        """
        if self.use_crf_layer:
            flattened_labels = tf.cast(tf.argmax(self.label_tf, axis=2), tf.int32)
            log_likelihood, _ = tf.contrib.crf.crf_log_likelihood(
                self.lstm_output_tf, flattened_labels, self.batch_sequence_lengths_tf
            )
            cost_tf = tf.reduce_mean(-log_likelihood, name="cost_tf")
        else:
            masked_logits = tf.boolean_mask(
                tf.reshape(self.lstm_output_tf, [-1, self.output_dimension]),
                self.batch_sequence_mask_tf,
            )

            masked_labels = tf.boolean_mask(
                tf.reshape(self.label_tf, [-1, self.output_dimension]),
                self.batch_sequence_mask_tf,
            )

            softmax_loss_tf = tf.nn.softmax_cross_entropy_with_logits(
                logits=masked_logits, labels=masked_labels, name="softmax_loss_tf"
            )

            cost_tf = tf.reduce_mean(softmax_loss_tf, name="cost_tf")

        optimizer_tf = tf.train.AdamOptimizer(
            learning_rate=float(self.learning_rate)
        ).minimize(cost_tf)

        return optimizer_tf, cost_tf

    def _calculate_score(self, output_arr, label_arr, seq_lengths_arr):
        """This function calculates the sequence score of all the queries,
        that is, the total number of queries where all the tags are predicted
        correctly.

        Args:
            output_arr (ndarray): Output array of the LSTM network
            label_arr (ndarray): Label array of the true labels of the data
            seq_lengths_arr (ndarray): A real sequence lengths of each example

        Returns:
            int: The number of queries where all the tags are correct
        """
        reshaped_output_arr = np.reshape(
            output_arr, [-1, int(self.padding_length), self.output_dimension]
        )
        reshaped_output_arr = np.argmax(reshaped_output_arr, 2)
        reshaped_labels_arr = np.argmax(label_arr, 2)

        score = 0
        for idx, _ in enumerate(reshaped_output_arr):
            seq_len = seq_lengths_arr[idx]
            predicted_tags = reshaped_output_arr[idx][:seq_len]
            actual_tags = reshaped_labels_arr[idx][:seq_len]
            if np.array_equal(predicted_tags, actual_tags):
                score += 1

        return score

    def _pad_labels(self, list_of_sequences, default_token):
        """
        Pads the label sequence

        Args:
            list_of_sequences (list): A list of label sequences
            default_token (str): The default label token for padding purposes

        Returns:
            list: padded output
        """
        padded_output = []
        for sequence in list_of_sequences:
            padded_seq = [default_token] * self.padding_length
            for idx, _ in enumerate(sequence):
                if idx < self.padding_length:
                    padded_seq[idx] = sequence[idx]
            padded_output.append(padded_seq)
        return padded_output

    def _generate_boolean_mask(self, seq_lengths):
        """
        Generates boolean masks for each query in a query list

        Args:
            seq_lengths (list): A list of sequence lengths

        Return:
            list: A list of boolean masking values
        """
        mask = [False] * (len(seq_lengths) * self.padding_length)
        for idx, seq_len in enumerate(seq_lengths):
            start_index = idx * self.padding_length
            for i in range(start_index, start_index + seq_len):
                mask[i] = True
        return mask

    @staticmethod
    def _construct_lstm_state(initializer, hidden_dimension, batch_size, name):
        """Construct the LSTM initial state

        Args:
            initializer (tf.contrib.layers.xavier_initializer): initializer used
            hidden_dimension: num dimensions of the hidden state variable
            batch_size: the batch size of the data
            name: suffix of the variable going to be used

        Returns:
            (LSTMStateTuple): LSTM state information
        """

        initial_cell_state = tf.get_variable(
            "initial_cell_state_{}".format(name),
            shape=[1, hidden_dimension],
            dtype=tf.float32,
            initializer=initializer,
        )

        initial_output_state = tf.get_variable(
            "initial_output_state_{}".format(name),
            shape=[1, hidden_dimension],
            dtype=tf.float32,
            initializer=initializer,
        )

        c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
        h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))

        return tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

    def _construct_regularized_lstm_cell(self, hidden_dimensions, initializer):
        """Construct a regularized lstm cell based on a dropout layer

        Args:
            hidden_dimensions: num dimensions of the hidden state variable
            initializer (tf.contrib.layers.xavier_initializer): initializer used

        Returns:
            (DropoutWrapper): regularized LSTM cell
        """

        lstm_cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
            hidden_dimensions,
            forget_bias=1.0,
            initializer=initializer,
            state_is_tuple=True,
        )

        lstm_cell = tf.contrib.rnn.DropoutWrapper(
            lstm_cell,
            input_keep_prob=self.lstm_input_keep_prob_tf,
            output_keep_prob=self.lstm_output_keep_prob_tf,
        )

        return lstm_cell

    def _construct_lstm_network(self, input_tensor):
        """This function constructs the Bi-Directional LSTM network

        Args:
            input_tensor (Tensor): Input tensor to the LSTM network

        Returns:
            output_tensor (Tensor): The output layer of the LSTM network
        """
        n_hidden = int(self.token_lstm_hidden_state_dimension)

        # We cannot use the static batch size variable since for the last batch set
        # of data, the data size could be less than the batch size
        batch_size_dim = tf.shape(input_tensor)[0]

        # We use the xavier initializer for some of it's gradient control properties
        initializer = tf.contrib.layers.xavier_initializer(seed=RANDOM_SEED)

        # Forward LSTM construction
        lstm_cell_forward_tf = self._construct_regularized_lstm_cell(
            n_hidden, initializer
        )
        initial_state_forward_tf = self._construct_lstm_state(
            initializer, n_hidden, batch_size_dim, "lstm_cell_forward_tf"
        )

        # Backward LSTM construction
        lstm_cell_backward_tf = self._construct_regularized_lstm_cell(
            n_hidden, initializer
        )
        initial_state_backward_tf = self._construct_lstm_state(
            initializer, n_hidden, batch_size_dim, "lstm_cell_backward_tf"
        )

        # Combined the forward and backward LSTM networks
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_cell_forward_tf,
            cell_bw=lstm_cell_backward_tf,
            inputs=input_tensor,
            sequence_length=self.batch_sequence_lengths_tf,
            dtype=tf.float32,
            initial_state_fw=initial_state_forward_tf,
            initial_state_bw=initial_state_backward_tf,
        )

        # Construct the output later
        output_tf = tf.concat([output_fw, output_bw], axis=-1)
        output_tf = tf.nn.dropout(output_tf, self.dense_keep_prob_tf)

        output_weights_tf = tf.get_variable(
            name="output_weights_tf",
            shape=[2 * n_hidden, self.output_dimension],
            dtype="float32",
            initializer=initializer,
        )
        output_weights_tf = tf.tile(output_weights_tf, [batch_size_dim, 1])
        output_weights_tf = tf.reshape(
            output_weights_tf, [batch_size_dim, 2 * n_hidden, self.output_dimension]
        )

        zero_initializer = tf.constant_initializer(ZERO_INITIALIZER_VALUE)
        output_bias_tf = tf.get_variable(
            name="output_bias_tf",
            shape=[self.output_dimension],
            dtype="float32",
            initializer=zero_initializer,
        )

        output_tf = tf.add(
            tf.matmul(output_tf, output_weights_tf),
            output_bias_tf,
            name="output_tensor",
        )
        return output_tf

    def _get_model_constructor(self):
        return self

    def _extract_seq_length(self, examples):
        """Extract sequence lengths from the input examples
        Args:
            examples (list of Query objects): List of input queries

        Returns:
            (list): List of seq lengths for each query
        """
        seq_lengths = []
        for example in examples:
            if len(example.normalized_tokens) > self.padding_length:
                seq_lengths.append(self.padding_length)
            else:
                seq_lengths.append(len(example.normalized_tokens))

        return seq_lengths

    def _get_features(self, examples):
        """Extracts the word and gazetteer embeddings from the input examples

        Args:
            examples (list of mindmeld.core.Query): a list of queries

        Returns:
            (tuple): Word embeddings and Gazetteer one-hot embeddings
        """
        x_feats_array = []
        gaz_feats_array = []
        char_feats_array = []
        for example in examples:
            x_feat, gaz_feat, char_feat = self._extract_features(example)
            x_feats_array.append(x_feat)
            gaz_feats_array.append(gaz_feat)
            char_feats_array.append(char_feat)

        # save all the embeddings used for model saving purposes
        self.query_encoder.save_embeddings()
        if self.use_char_embeddings:
            self.char_encoder.save_embeddings()

        x_feats_array = np.asarray(x_feats_array)
        gaz_feats_array = np.asarray(gaz_feats_array)
        char_feats_array = (
            np.asarray(char_feats_array) if self.use_char_embeddings else []
        )

        return x_feats_array, gaz_feats_array, char_feats_array

    def _gaz_transform(self, list_of_tokens_to_transform):
        """This function is used to handle special logic around SKLearn's LabelBinarizer
        class which behaves in a non-standard way for 2 classes. In a 2 class system,
        it encodes the classes as [0] and [1]. However, in a 3 class system, it encodes
        the classes as [0,0,1], [0,1,0], [1,0,0] and sustains this behavior for num_class > 2.

        We want to encode 2 class systems as [0,1] and [1,0]. This function does that.

        Args:
            list_of_tokens_to_transform (list): A sequence of class labels

        Returns:
            (array): corrected encoding from the binarizer
        """
        output = self.gaz_encoder.transform(list_of_tokens_to_transform)
        if len(self.gaz_encoder.classes_) == 2:
            output = np.hstack((1 - output, output))
        return output

    def _extract_features(self, example):
        """Extracts feature dicts for each token in an example.

        Args:
            example (mindmeld.core.Query): an query

        Returns:
            (list of dict): features
        """
        default_gaz_one_hot = self._gaz_transform([DEFAULT_GAZ_LABEL]).tolist()[0]
        extracted_gaz_tokens = [default_gaz_one_hot] * self.padding_length
        extracted_sequence_features = extract_sequence_features(
            example, self.example_type, self.features, self.resources
        )

        for index, extracted_gaz in enumerate(extracted_sequence_features):
            if index >= self.padding_length:
                break

            if extracted_gaz == {}:
                continue

            combined_gaz_features = set()
            for key in extracted_gaz.keys():
                regex_match = re.match(GAZ_PATTERN_MATCH, key)
                if regex_match:
                    # Examples of gaz features here are:
                    # in-gaz|type:city|pos:start|p_fe,
                    # in-gaz|type:city|pos:end|pct-char-len
                    # There were many gaz features of the same type that had
                    # bot start and end position tags for a given token.
                    # Due to this, we did not implement functionality to
                    # extract the positional information due to the noise
                    # associated with it.
                    combined_gaz_features.add(
                        regex_match.group(REGEX_TYPE_POSITIONAL_INDEX)
                    )

            if len(combined_gaz_features) != 0:
                total_encoding = np.zeros(self.gaz_dimension, dtype=np.int)
                for encoding in self._gaz_transform(list(combined_gaz_features)):
                    total_encoding = np.add(total_encoding, encoding)
                extracted_gaz_tokens[index] = total_encoding.tolist()

        padded_query = self.query_encoder.encode_sequence_of_tokens(
            example.normalized_tokens
        )

        if self.use_char_embeddings:
            padded_char = self.char_encoder.encode_sequence_of_tokens(
                example.normalized_tokens
            )
        else:
            padded_char = None

        return padded_query, extracted_gaz_tokens, padded_char

    def _fit(self, X, y):
        """Trains a classifier without cross-validation. It iterates through
        the data, feeds batches to the tensorflow session graph and fits the
        model based on the feed forward and back propagation steps.

        Args:
            X (list of list of list of str): a list of queries to train on
            y (list of list of str): a list of expected labels
        """
        self.construct_tf_variables()
        self.session = tf.Session(graph=self.graph)

        self.session.run([self.global_init, self.local_init])

        for epochs in range(int(self.number_of_epochs)):
            logger.info("Training epoch : %s", epochs)

            indices = list(range(len(X)))
            np.random.shuffle(indices)

            gaz = self.gaz_features_arr[indices]
            char = self.char_features_arr[indices] if self.use_char_embeddings else []

            examples = X[indices]
            labels = y[indices]
            batch_size = int(self.batch_size)
            num_batches = int(math.ceil(len(examples) / batch_size))
            seq_len = np.array(self.sequence_lengths)[indices]

            for batch in range(num_batches):
                batch_start_index = batch * batch_size
                batch_end_index = (batch * batch_size) + batch_size

                batch_info = {
                    "batch_examples": examples[batch_start_index:batch_end_index],
                    "batch_labels": labels[batch_start_index:batch_end_index],
                    "batch_gaz": gaz[batch_start_index:batch_end_index],
                    "batch_seq_len": seq_len[batch_start_index:batch_end_index],
                    "batch_char": char[batch_start_index:batch_end_index],
                }

                if batch % int(self.display_epoch) == 0:
                    output, loss, _ = self.session.run(
                        [self.lstm_output_tf, self.cost_tf, self.optimizer_tf],
                        feed_dict=self.construct_feed_dictionary(**batch_info),
                    )

                    score = self._calculate_score(
                        output, batch_info["batch_labels"], batch_info["batch_seq_len"]
                    )
                    accuracy = score / (len(batch_info["batch_examples"]) * 1.0)

                    logger.info(
                        "Trained batch from index {} to {}, "
                        "Mini-batch loss: {:.5f}, "
                        "Training sequence accuracy: {:.5f}".format(
                            batch * batch_size,
                            (batch * batch_size) + batch_size,
                            loss,
                            accuracy,
                        )
                    )
                else:
                    self.session.run(
                        self.optimizer_tf,
                        feed_dict=self.construct_feed_dictionary(**batch_info),
                    )
        return self

    def _predict(self, X):
        """Predicts tags for query sequence

        Args:
            X (list of list of list of str): a list of input representations

        Returns:
            (list): A list of decoded labelled predicted by the model
        """
        seq_len_arr = np.array(self.sequence_lengths)

        # During predict time, we make sure no nodes are dropped out
        self.dense_keep_probability = 1.0
        self.lstm_input_keep_prob = 1.0
        self.lstm_output_keep_prob = 1.0

        output = self.session.run(
            [self.lstm_output_softmax_tf],
            feed_dict=self.construct_feed_dictionary(
                X, self.char_features_arr, self.gaz_features_arr, seq_len_arr
            ),
        )

        output = np.reshape(
            output, [-1, int(self.padding_length), self.output_dimension]
        )
        output = np.argmax(output, 2)

        decoded_queries = []
        for idx, encoded_predict in enumerate(output):
            decoded_query = []
            for tag in encoded_predict[: self.sequence_lengths[idx]]:
                decoded_query.append(self.label_encoder.classes_[tag])
            decoded_queries.append(decoded_query)

        return decoded_queries

    def _predict_proba(self, X):
        """Predict tags for query sequence with their confidence scores

        Args:
            X (list of list of list of str): a list of input representations

        Returns:
            (list): A list of decoded labelled predicted by the model with confidence scores
        """

        seq_len_arr = np.array(self.sequence_lengths)

        # During predict time, we make sure no nodes are dropped out
        self.dense_keep_probability = 1.0
        self.lstm_input_keep_prob = 1.0
        self.lstm_output_keep_prob = 1.0

        output = self.session.run(
            [self.lstm_output_softmax_tf],
            feed_dict=self.construct_feed_dictionary(
                X, self.char_features_arr, self.gaz_features_arr, seq_len_arr
            ),
        )

        output = np.reshape(
            output, [-1, int(self.padding_length), self.output_dimension]
        )
        class_output = np.argmax(output, 2)

        decoded_queries = []
        for idx, encoded_predict in enumerate(class_output):
            decoded_query = []
            for token_idx, tag in enumerate(
                encoded_predict[: self.sequence_lengths[idx]]
            ):
                decoded_query.append(
                    [self.label_encoder.classes_[tag], output[idx][token_idx][tag]]
                )
            decoded_queries.append(decoded_query)

        return decoded_queries

    def dump(self, path, config):
        """
        Saves the Tensorflow model

        Args:
            path (str): the folder path for the entity model folder
            config (dict): The model config
        """
        path = path.split(".pkl")[0] + "_model_files"
        config["model"] = path
        config["serializable"] = False

        if not os.path.isdir(path):
            os.makedirs(path)

        if not self.saver:
            # This conditional happens when there are not entities for the associated
            # model
            return

        self.saver.save(self.session, os.path.join(path, "lstm_model"))

        # Save feature extraction variables
        variables_to_dump = {
            "resources": self.resources,
            "gaz_dimension": self.gaz_dimension,
            "output_dimension": self.output_dimension,
            "gaz_features": self.gaz_features_arr,
            "sequence_lengths": self.sequence_lengths,
            "gaz_encoder": self.gaz_encoder,
            "label_encoder": self.label_encoder,
        }

        joblib.dump(variables_to_dump, os.path.join(path, ".feature_extraction_vars"))

    def unload(self):
        self.graph = None
        self.session = None
        self.resources = None
        self.gaz_dimension = None
        self.output_dimension = None
        self.gaz_features = None
        self.sequence_lengths = None
        self.gaz_encoder = None
        self.label_encoder = None

    def load(self, path):
        """
        Loads the Tensorflow model

        Args:
            path (str): the folder path for the entity model folder
        """
        path = path.split(".pkl")[0] + "_model_files"

        if not os.path.exists(os.path.join(path, "lstm_model.meta")):
            # This conditional is for models with no labels where no TF graph was built
            # for this.
            return

        self.graph = tf.Graph()
        self.session = tf.Session(graph=self.graph)

        with self.graph.as_default():
            saver = tf.train.import_meta_graph(os.path.join(path, "lstm_model.meta"))
            saver.restore(self.session, os.path.join(path, "lstm_model"))

            # Restore tensorflow graph variables
            self.dense_keep_prob_tf = self.session.graph.get_tensor_by_name(
                "dense_keep_prob_tf:0"
            )

            self.lstm_input_keep_prob_tf = self.session.graph.get_tensor_by_name(
                "lstm_input_keep_prob_tf:0"
            )

            self.lstm_output_keep_prob_tf = self.session.graph.get_tensor_by_name(
                "lstm_output_keep_prob_tf:0"
            )

            self.query_input_tf = self.session.graph.get_tensor_by_name(
                "query_input_tf:0"
            )

            self.gaz_input_tf = self.session.graph.get_tensor_by_name("gaz_input_tf:0")

            self.label_tf = self.session.graph.get_tensor_by_name("label_tf:0")

            self.batch_sequence_lengths_tf = self.session.graph.get_tensor_by_name(
                "batch_sequence_lengths_tf:0"
            )

            self.batch_sequence_mask_tf = self.session.graph.get_tensor_by_name(
                "batch_sequence_mask_tf:0"
            )

            self.lstm_output_tf = self.session.graph.get_tensor_by_name(
                "output_tensor:0"
            )

            self.lstm_output_softmax_tf = self.session.graph.get_tensor_by_name(
                "output_softmax_tensor:0"
            )

            if self.use_char_embeddings:
                self.char_input_tf = self.session.graph.get_tensor_by_name(
                    "char_input_tf:0"
                )

        # Load feature extraction variables
        variables_to_load = joblib.load(os.path.join(path, ".feature_extraction_vars"))
        self.resources = variables_to_load["resources"]
        self.gaz_dimension = variables_to_load["gaz_dimension"]
        self.output_dimension = variables_to_load["output_dimension"]
        self.gaz_features = variables_to_load["gaz_features"]
        self.sequence_lengths = variables_to_load["sequence_lengths"]
        self.gaz_encoder = variables_to_load["gaz_encoder"]
        self.label_encoder = variables_to_load["label_encoder"]
