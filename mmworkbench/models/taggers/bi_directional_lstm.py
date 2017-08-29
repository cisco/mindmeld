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
                 dropout_rate,
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
        self.dropout_rate = dropout_rate
        self.labels_dict = labels_dict
        self.embedding_gaz_matrix = embedding_gaz_matrix
        self.gaz_features = gaz_features

        self.input = tf.placeholder(tf.int32, [None, self.padding_length])
        self.gaz_input = tf.placeholder(tf.int32, [None, self.padding_length])
        self.embedding_matrix_tensor = tf.placeholder(tf.float32,
                                                      [None, self.token_embedding_dimension])
        self.embedding_matrix_gaz_tensor = tf.placeholder(tf.float32,
                                                          [None,
                                                           self.embedding_gaz_matrix.shape[1]])
        self.label_tensor = tf.placeholder(tf.int32,
                                           [None, int(self.padding_length), self.output_dimension])
        self.sequence_lengths = tf.placeholder(tf.int32, shape=[None],
                                               name="sequence_lengths")
        self.tf_dropout_rate = tf.placeholder(tf.float32, shape=None)

        input_tensor = tf.nn.embedding_lookup(self.embedding_matrix_tensor, self.input)
        gaz_tensor = tf.nn.embedding_lookup(self.embedding_matrix_gaz_tensor, self.gaz_input)

        x_concat = tf.concat([input_tensor, gaz_tensor], axis=-1)

        self.output_tensor = self._construct_network(x_concat)

        self.optimizer, self.cost = self._define_optimizer_and_cost(
            self.output_tensor, self.label_tensor, self.sequence_lengths)

        self.accuracy = self._calculate_accuracy(self.output_tensor, self.label_tensor)

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
        output = tf.nn.dropout(output, float(self.dropout_rate))

        weights = tf.get_variable("weights_out", shape=[2 * n_hidden, self.output_dimension],
                                  dtype="float32", initializer=tf.random_normal_initializer())

        biases = tf.get_variable("bias_out", shape=[self.output_dimension],
                                 dtype="float32", initializer=tf.random_normal_initializer())

        output_tensor = tf.matmul(output, weights) + biases
        return output_tensor

    def construct_feed_dictionary(self, batch_examples, batch_labels, batch_gaz, batch_seq_len):
        return {
            self.input: batch_examples,
            self.label_tensor: batch_labels,
            self.sequence_lengths: batch_seq_len,
            self.gaz_input: batch_gaz,
            self.embedding_matrix_tensor: self.embedding_matrix,
            self.embedding_matrix_gaz_tensor: self.embedding_gaz_matrix,
            self.tf_dropout_rate: self.dropout_rate
        }

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

                print("display step: {}".format(self.display_step))

                if batch % int(self.display_step) == 0:
                    acc, loss = self.session.run([self.accuracy, self.cost],
                                                 feed_dict=self.construct_feed_dictionary(
                                                     batch_examples, batch_labels, batch_gaz, seq_len))
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
                                         batch_examples, batch_labels, batch_gaz, seq_len))

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
        output_tensor = self.session.run(
            self.output_tensor,
            feed_dict={self.input: examples,
                       self.sequence_lengths: seq_len,
                       self.gaz_input: gaz,
                       self.embedding_matrix_tensor: self.embedding_matrix,
                       self.embedding_matrix_gaz_tensor: self.embedding_gaz_matrix,
                       self.tf_dropout_rate: 1.0})

        output_tensor = tf.reshape(output_tensor,
                                   [-1, int(self.padding_length), self.output_dimension])
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
