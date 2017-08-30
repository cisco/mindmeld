import tensorflow as tf
import numpy as np
import math

import logging

logger = logging.getLogger(__name__)


class LstmNetwork:
    """
    This class is the implementation of a Bi-Directional LSTM network in Tensorflow.
    """

    def __init__(self,
                 seq_len,
                 maximum_number_of_epochs,
                 batch_size,
                 token_lstm_hidden_state_dimension,
                 learning_rate,
                 optimizer,
                 output_dimension,
                 display_step,
                 padding_length,
                 embedding_matrix,
                 token_embedding_dimension,
                 token_pretrained_embedding_filepath,
                 dense_keep_probability,
                 lstm_input_keep_prob,
                 lstm_output_keep_prob,
                 labels_dict,
                 embedding_gaz_matrix,
                 gaz_features):

        tf.reset_default_graph()
        self.session = tf.Session()
        self.seq_len = seq_len
        self.maximum_number_of_epochs = maximum_number_of_epochs
        self.batch_size = batch_size
        self.token_lstm_hidden_state_dimension = token_lstm_hidden_state_dimension
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.output_dimension = output_dimension
        self.padding_length = padding_length
        self.display_step = display_step
        self.embedding_matrix = embedding_matrix
        self.token_embedding_dimension = token_embedding_dimension
        self.token_pretrained_embedding_filepath = token_pretrained_embedding_filepath
        self.dense_keep_probability = dense_keep_probability
        self.lstm_input_keep_prob = lstm_input_keep_prob
        self.lstm_output_keep_prob = lstm_output_keep_prob
        self.labels_dict = labels_dict
        self.embedding_gaz_matrix = embedding_gaz_matrix
        self.gaz_features = gaz_features

        self.word_embedding_dimension = self.embedding_matrix.shape[0]
        self.word_vocab_size = self.embedding_matrix.shape[1]
        self.input = tf.placeholder(tf.int32, [None, self.padding_length])
        self.gaz_input = tf.placeholder(tf.int32, [None, self.padding_length])
        self.embedding_matrix_tensor = tf.placeholder(tf.float32, [None, self.token_embedding_dimension])
        self.embedding_matrix_gaz_tensor = tf.placeholder(tf.float32, [None, self.embedding_gaz_matrix.shape[1]])
        self.label_tensor = tf.placeholder(tf.int32, [None, int(self.padding_length), self.output_dimension])
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None], name="sequence_lengths")

        # input_tensor = tf.nn.embedding_lookup(self.embedding_matrix_tensor, self.input)
        # gaz_tensor = tf.nn.embedding_lookup(self.embedding_matrix_gaz_tensor, self.gaz_input)
        #
        # x_concat = tf.concat([input_tensor, gaz_tensor], axis=-1)

        combined_embedding = self._construct_embedding_network()
        self.output_tensor = self._construct_network(combined_embedding)

        self.optimizer, self.cost = self._define_optimizer_and_cost(
            self.output_tensor, self.label_tensor, self.sequence_lengths)

        self.accuracy = self._calculate_accuracy(self.output_tensor, self.label_tensor)

    def _construct_embedding_network(self):
        """
        Constructs all aspects of the network pre-LSTM.
        :return: just word embedding, character level word embedding, combined (including gaz or ortho if used),
        gaz representation (for optional appending to LSTM output)
        """

        # Sequence lengths tensor
        batch_size = tf.shape(self.sequence_lengths)[0]

        # GAZ CONSTRUCTION
        gaz_one_hot = tf.one_hot(
            indices=tf.cast(self.gaz_input, tf.int32),
            depth=self.embedding_gaz_matrix.shape[1],
            dtype=tf.float32)

        gaz_embedding_matrix_weights = tf.Variable(
            self.embedding_gaz_matrix,
            dtype=tf.float32,
            trainable=False)

        gaz_embedding_matrix_weights = tf.reshape(
            tf.tile(gaz_embedding_matrix_weights, [batch_size, 1]),
            [batch_size, self.embedding_gaz_matrix.shape[0],
             self.embedding_gaz_matrix.shape[1]])

        # Each word gets turned into a binary vector that represents presense in gazetteers
        gaz_embedding = tf.matmul(gaz_embedding_matrix_weights, gaz_one_hot, transpose_b=True)
        gaz_embedding = tf.transpose(gaz_embedding, [0, 2, 1])

        dense_gaz_embedding = tf.contrib.layers.fully_connected(gaz_embedding, 100)

        # WORD EMBEDDING CONSTRUCTION

        # Shape: batch_size BY query_padding_length BY word_vocab_size
        words_one_hot = tf.one_hot(
            indices=tf.cast(self.input, tf.int32),
            depth=self.word_vocab_size,
            dtype=tf.float32)

        word_embedding_matrix_weights = tf.Variable(
            self.embedding_matrix,
            dtype=tf.float32,
            trainable=False)

        # Duplicate/tile the matrix weights batch_size number of times so each example has
        # the weights to multiply by. Shape: (word_embedding_dimension*batch_size) BY word_vocab_size
        word_embedding_matrix_weights = tf.tile(word_embedding_matrix_weights, [batch_size, 1])

        # Must reshape so each batch_size dimension has a copy of the word embedding matrix weights
        # Shape: batch_size BY word_embedding_dimension BY word_vocab_size
        word_embedding_matrix_weights = tf.reshape(word_embedding_matrix_weights,
                                                   [batch_size,
                                                    self.word_embedding_dimension,
                                                    self.word_vocab_size])

        # Extract the column vector that corresponds to the embedding
        # Shape: batch_size BY word_embedding_dimension BY query_padding_length
        word_embedding = tf.matmul(word_embedding_matrix_weights, words_one_hot, transpose_b=True)

        # Transpose because for each batch we want the first dimension to correspond to words, second to embedding
        word_embedding = tf.transpose(word_embedding, [0, 2, 1])

        combined_embedding = tf.concat([word_embedding, dense_gaz_embedding], axis=2)

        return combined_embedding

    def _define_optimizer_and_cost(self, output_tensor, label_tensor, sequence_lengths):
        """ This function defines the optimizer and cost function of the LSTM model
        Args:
            output_tensor (Tensor): Output tensor of the LSTM network
            label_tensor (Tensor): Label tensor of the true labels of the data
            sequence_lengths (Tensor): The sequence lengths of each query

        Returns:
            The optimizer function to reduce loss and the loss values
        """
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=output_tensor,
            labels=tf.reshape(label_tensor, [-1, self.output_dimension]), name='softmax')
        cost = tf.reduce_mean(losses, name='cross_entropy_mean_loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=float(self.learning_rate)).minimize(
            cost)
        return optimizer, cost

    def _calculate_accuracy(self, output_tensor, label_tensor):
        """ This function calculates the sequence accuracy of all the queries, that is, the total number of queries
        where all the tags are predicted correctly.
        Args:
            output_tensor (Tensor): Output tensor of the LSTM network
            label_tensor (Tensor): Label tensor of the true labels of the data

        Returns:
            The number of queries where all the tags are correct
        """
        output_tensor = tf.reshape(output_tensor,
                                   [-1, int(self.padding_length), self.output_dimension])
        accuracy = tf.cast(
            tf.equal(tf.argmax(output_tensor, 2), tf.argmax(label_tensor, 2)), tf.float32)
        accuracy = tf.cast(tf.reduce_sum(accuracy, 1), tf.int32)
        accuracy = tf.reduce_sum(
            tf.cast(tf.equal(accuracy, tf.cast(
                tf.constant(int(self.padding_length)), tf.int32)), tf.int32))
        return accuracy

    def _construct_network(self, input_tensor):
        """ This function calculates the sequence accuracy of all the queries, that is, the total number of queries
        where all the tags are predicted correctly.
        Args:
            output_tensor (Tensor): Output tensor of the LSTM network
            label_tensor (Tensor): Label tensor of the true labels of the data

        Returns:
            The number of queries where all the tags are correct
        """
        n_hidden = int(self.token_lstm_hidden_state_dimension)
        initial_state = {}
        batch_size_dim = tf.shape(input_tensor)[0]
        initializer = tf.contrib.layers.xavier_initializer()

        lstm_cell_for = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
            n_hidden, forget_bias=1.0, initializer=initializer, state_is_tuple=True)

        lstm_cell_back = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
            n_hidden, forget_bias=1.0, initializer=initializer, state_is_tuple=True)

        lstm_cell_for = tf.contrib.rnn.DropoutWrapper(
            lstm_cell_for, input_keep_prob=self.lstm_input_keep_prob, output_keep_prob=self.lstm_output_keep_prob
        )

        lstm_cell_back = tf.contrib.rnn.DropoutWrapper(
            lstm_cell_back, input_keep_prob=self.lstm_input_keep_prob, output_keep_prob=self.lstm_output_keep_prob
        )

        initial_cell_state = tf.get_variable(
            "initial_cell_state", shape=[1, n_hidden], dtype=tf.float32, initializer=initializer)

        initial_output_state = tf.get_variable(
            "initial_output_state", shape=[1, n_hidden], dtype=tf.float32, initializer=initializer)

        initial_cell_state_2 = tf.get_variable(
            "initial_cell_state_2", shape=[1, n_hidden], dtype=tf.float32, initializer=initializer)

        initial_output_state_2 = tf.get_variable(
            "initial_output_state_2", shape=[1, n_hidden], dtype=tf.float32, initializer=initializer)

        c_states = tf.tile(initial_cell_state, tf.stack([batch_size_dim, 1]))
        h_states = tf.tile(initial_output_state, tf.stack([batch_size_dim, 1]))
        initial_state["forward"] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

        c_states = tf.tile(initial_cell_state_2, tf.stack([batch_size_dim, 1]))
        h_states = tf.tile(initial_output_state_2, tf.stack([batch_size_dim, 1]))
        initial_state["backward"] = tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            lstm_cell_for, lstm_cell_back,
            inputs=input_tensor,
            sequence_length=self.sequence_lengths,
            dtype=tf.float32,
            initial_state_fw=initial_state["forward"],
            initial_state_bw=initial_state["backward"])

        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.reshape(output, [-1, 2 * n_hidden])
        output = tf.nn.dropout(output, float(self.dense_keep_probability))

        weights = tf.get_variable("weights_out", shape=[2 * n_hidden, self.output_dimension],
                                  dtype="float32", initializer=tf.random_normal_initializer())

        biases = tf.get_variable("bias_out", shape=[self.output_dimension],
                                 dtype="float32", initializer=tf.random_normal_initializer())

        output_tensor = tf.matmul(output, weights) + biases
        return output_tensor

    def construct_feed_dictionary(self, batch_examples, batch_gaz, batch_seq_len, batch_labels=list()):
        return_dict = {
            self.input: batch_examples,
            self.sequence_lengths: batch_seq_len,
            self.gaz_input: batch_gaz,
            self.embedding_matrix_tensor: self.embedding_matrix,
            self.embedding_matrix_gaz_tensor: self.embedding_gaz_matrix
        }

        if len(batch_labels) > 0:
            return_dict[self.label_tensor] = batch_labels

        return return_dict

    def fit(self, examples, labels):
        """ Main function that iterates through the data, feeds batches to the tensorflow session graph and fits
        the model based on the feed forward and back propogation steps.
        Args:
            example (ndarray): Matrix of encoded queries
            labels (ndarray): Matrix of encoded labels
        """
        self.session.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

        for epochs in range(int(self.maximum_number_of_epochs)):
            print("Epoch : {}".format(epochs))

            indices = [x for x in range(len(examples))]
            np.random.shuffle(indices)
            gaz = self.gaz_features[indices]
            examples = examples[indices]
            labels = labels[indices]
            batch_size = int(self.batch_size)
            num_batches = int(math.ceil(len(examples) / batch_size))

            for batch in range(num_batches):
                batch_examples = examples[batch * batch_size: (batch * batch_size) + batch_size]
                batch_labels = labels[batch * batch_size: (batch * batch_size) + batch_size]
                batch_gaz = gaz[batch * batch_size: (batch * batch_size) + batch_size]
                seq_len = np.ones(len(batch_examples)) * int(self.padding_length)

                if batch % int(self.display_step) == 0:
                    acc, loss = self.session.run([self.accuracy, self.cost],
                                                 feed_dict=self.construct_feed_dictionary(
                                                     batch_examples, batch_gaz, seq_len, batch_labels))
                    accuracy = acc / (len(batch_examples) * 1.0)

                    print("Iteration number " + str(batch * batch_size) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(accuracy))

                    logger.info("Iteration number " + str(batch * batch_size) + ", Minibatch Loss= " +
                          "{:.6f}".format(loss) + ", Training Accuracy= " +
                          "{:.5f}".format(accuracy))
                else:
                    self.session.run(self.optimizer,
                                     feed_dict=self.construct_feed_dictionary(
                                         batch_examples, batch_gaz, seq_len, batch_labels))

        return self

    def predict(self, examples):
        """ Predicts the entity tags of the examples
        Args:
            example (ndarray): Matrix of encoded queries

        Returns:
            Array of decoded predicted tags
        """
        gaz = self.gaz_features
        seq_len = np.ones(len(examples)) * int(self.padding_length)

        self.dense_keep_probability = 1.0
        self.lstm_input_keep_prob = 1.0
        self.lstm_output_keep_prob = 1.0

        output_tensor = self.session.run(
            self.output_tensor,
            feed_dict=self.construct_feed_dictionary(examples, gaz, seq_len))

        output_tensor = tf.reshape(
            output_tensor, [-1, int(self.padding_length), self.output_dimension])
        output_tensor = tf.argmax(output_tensor, 2)
        output_array = self.session.run(output_tensor)

        id_to_label = {}
        for key_name in self.labels_dict.keys():
            id_to_label[self.labels_dict[key_name]] = key_name

        decoded_queries = []
        for i in range(len(output_array)):
            decoded_query = []
            for j in range(len(output_array[0])):
                decoded_query.append(id_to_label[output_array[i][j]])
            decoded_queries.append(decoded_query)

        return decoded_queries
