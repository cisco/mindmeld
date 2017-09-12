import numpy as np
import tensorflow as tf
import re
import math
import logging

from .taggers import Tagger, extract_sequence_features
from .embeddings import LabelSequenceEmbedding, \
    WordSequenceEmbedding, \
    GazetteerSequenceEmbedding

DEFAULT_ENTITY_TOKEN_SPAN_INDEX = 2
GAZ_PATTERN_MATCH = 'in-gaz\|type:(\w+)\|pos:(\w+)\|'
REGEX_TYPE_POSITIONAL_INDEX = 1
DEFAULT_LABEL = 'B|UNK'
DEFAULT_PADDED_TOKEN = '<UNK>'
DEFAULT_GAZ_LABEL = 'O'
RANDOM_SEED = 1

logger = logging.getLogger(__name__)


class LstmModel(Tagger):
    """"This class encapsulates the bi-directional LSTM model and provides
    the correct interface for use by the tagger model"""

    def fit(self, X, y):
        examples = np.asarray(X, dtype='float32')
        labels = np.asarray(y, dtype='int32')
        self._fit(examples, labels)
        return self

    def predict(self, X):
        encoded_examples = np.asarray(X, dtype='float32')
        tags_by_example = self._predict(encoded_examples)

        resized_predicted_tags = []
        for query, seq_len in zip(tags_by_example, self.sequence_lengths):
            resized_predicted_tags.append(query[:seq_len])

        return resized_predicted_tags

    def set_params(self, **parameters):
        """Initialize params

        Args:
            The keys in the parameters dictionary are as follows:

            number_of_epochs (int): The number of epochs to run
            batch_size (int): The batch size for mini-batch training
            token_lstm_hidden_state_dimension (int): The hidden state
            dimension of the LSTM cell
            learning_rate (int): The learning rate of the optimizer
            optimizer (str): The optimizer used to train the network
            is the number of entities in the dataset
            display_epoch (int): The number of epochs after which the
            network displays common stats like accuracy
            padding_length (int): The length of each query, which is
            fixed, so some queries will be cut short in length
            representing the word embedding, the row index
            is the word's index
            token_embedding_dimension (int): The embedding dimension of the word
            token_pretrained_embedding_filepath (str): The pretrained embedding file-path
            dense_keep_prob (float): The dropout rate of the dense layers
            lstm_input_keep_prob (float): The dropout rate of the inputs to the LSTM cell
            lstm_output_keep_prob (float): The dropout rate of the outputs of the LSTM cell
            gaz_encoding_dimension (int): The gazetteer encoding dimension
        """
        self.number_of_epochs = parameters.get('number_of_epochs', 20)
        self.batch_size = parameters.get('batch_size', 20)
        self.token_lstm_hidden_state_dimension = \
            parameters.get('token_lstm_hidden_state_dimension', 300)
        self.learning_rate = parameters.get('learning_rate', 0.005)
        self.optimizer = parameters.get('optimizer', 'adam')
        self.padding_length = parameters.get('padding_length', 20)
        self.display_epoch = parameters.get('display_epoch', 20)
        self.token_embedding_dimension = parameters.get('token_embedding_dimension', 300)
        self.token_pretrained_embedding_filepath = \
            parameters.get('token_pretrained_embedding_filepath')
        self.dense_keep_probability = parameters.get('dense_keep_prob', 0.5)
        self.lstm_input_keep_prob = parameters.get('lstm_input_keep_prob', 0.5)
        self.lstm_output_keep_prob = parameters.get('lstm_output_keep_prob', 0.5)
        self.gaz_encoding_dimension = parameters.get('gaz_encoding_dimension', 100)

    def get_params(self, deep=True):
        return self.__dict__

    def construct_tf_variables(self):
        """
        Constructs the variables and operations in the tensorflow session graph
        """
        self.tf_dense_keep_prob = tf.placeholder(tf.float32, name='dense_keep_prob')
        self.tf_lstm_input_keep_prob = tf.placeholder(tf.float32, name='input_keep_prob')
        self.tf_lstm_output_keep_prob = tf.placeholder(tf.float32, name='output_keep_prob')

        self.tf_query_input = tf.placeholder(tf.float32,
                                             [None,
                                              self.padding_length,
                                              self.token_embedding_dimension],
                                             name='tf_query_input')

        self.tf_gaz_input = tf.placeholder(tf.float32,
                                           [None,
                                            self.padding_length,
                                            self.gaz_dimension],
                                           name='tf_gaz_input')

        self.tf_label = tf.placeholder(tf.int32,
                                       [None,
                                        int(self.padding_length),
                                        self.output_dimension],
                                       name='tf_label')

        self.tf_sequence_length = tf.placeholder(tf.int32, shape=[None],
                                                 name='tf_sequence_length')

        word_and_gaz_embedding = self._construct_embedding_network()
        self.tf_lstm_output = self._construct_lstm_network(word_and_gaz_embedding)

        self.optimizer, self.cost = self._define_optimizer_and_cost(
            self.tf_lstm_output, self.tf_label)

    def extract_features(self, examples, config, resources, y=None, fit=True):
        if y:
            # Train time
            self.resources = resources

            # The gaz dimension are the sum total of the gazetteer entities and
            # the 'other' gaz entity, which is the entity for all non-gazetteer tokens
            self.gaz_dimension = len(self.resources['gazetteers'].keys()) + 1

            self.example_type = config.example_type
            self.features = config.features

            self.token_pretrained_embedding_filepath = \
                config.params.get('token_pretrained_embedding_filepath')

            self.padding_length = config.params.get('padding_length')

            self.label_encoder = LabelSequenceEmbedding(self.padding_length,
                                                        DEFAULT_LABEL)

            self.query_encoder = WordSequenceEmbedding(
                self.padding_length, DEFAULT_PADDED_TOKEN, True,
                self.token_embedding_dimension, self.token_pretrained_embedding_filepath)

            self.gaz_encoder = GazetteerSequenceEmbedding(self.padding_length,
                                                          DEFAULT_GAZ_LABEL,
                                                          self.gaz_dimension)

            encoded_labels = []
            for sequence in y:
                encoded_labels.append(self.label_encoder.encode_sequence_of_tokens(sequence))

            embedded_labels = self.label_encoder.get_embeddings_from_encodings(encoded_labels)

            self.output_dimension = len(self.label_encoder.token_to_encoding_mapping.keys())
        else:
            # Predict time
            embedded_labels = None

        # Extract features and classes
        X, gaz = self._get_features(examples)
        self.gaz_features = np.asarray(gaz, dtype='float32')
        self.sequence_lengths = self._extract_seq_length(examples)

        # There are no groups in this model
        groups = None

        return X, embedded_labels, groups

    def setup_model(self, config=None):
        # We have to reset the graph on every dataset since the input, gaz and output
        # dimensions for each domain,intent training data is different. So the graph
        # cannot be reused.
        tf.reset_default_graph()
        self.session = tf.Session()

    def construct_feed_dictionary(self,
                                  batch_examples,
                                  batch_gaz,
                                  batch_seq_len,
                                  batch_labels=list()):
        """Constructs the feed dictionary that is used to feed data into the tensors

        Args:
            batch_examples (ndarray): A batch of examples
            batch_gaz (ndarray): A batch of gazetteer features
            batch_seq_len (ndarray): A batch of sequence length of each query
            batch_labels (ndarray): A batch of labels

        Returns:
            The feed dictionary
        """
        return_dict = {
            self.tf_query_input: batch_examples,
            self.tf_sequence_length: batch_seq_len,
            self.tf_gaz_input: batch_gaz,
            self.tf_dense_keep_prob: self.dense_keep_probability,
            self.tf_lstm_input_keep_prob: self.lstm_input_keep_prob,
            self.tf_lstm_output_keep_prob: self.lstm_output_keep_prob
        }

        if len(batch_labels) > 0:
            return_dict[self.tf_label] = batch_labels

        return return_dict

    def _construct_embedding_network(self):
        """ Constructs a network based on the word embedding and gazetteer
        inputs and concatenates them together

        Returns:
            Combined embeddings of the word and gazetteer embeddings
        """
        initializer = tf.contrib.layers.xavier_initializer(seed=RANDOM_SEED)

        dense_gaz_embedding = tf.contrib.layers.fully_connected(
            inputs=self.tf_gaz_input,
            num_outputs=self.gaz_encoding_dimension,
            weights_initializer=initializer)

        # Combined the two embeddings
        combined_embedding = tf.concat([self.tf_query_input, dense_gaz_embedding], axis=2)
        return combined_embedding

    def _define_optimizer_and_cost(self, output_tensor, label_tensor):
        """ This function defines the optimizer and cost function of the LSTM model

        Args:
            output_tensor (Tensor): Output tensor of the LSTM network
            label_tensor (Tensor): Label tensor of the true labels of the data

        Returns:
            The optimizer function to reduce loss and the loss values
        """
        losses = tf.nn.softmax_cross_entropy_with_logits(
            logits=output_tensor,
            labels=tf.reshape(label_tensor, [-1, self.output_dimension]), name='softmax')
        cost = tf.reduce_mean(losses, name='cross_entropy_mean_loss')
        optimizer = tf.train.AdamOptimizer(learning_rate=float(self.learning_rate)).minimize(cost)
        return optimizer, cost

    def _calculate_score(self, output_array, label_array, seq_lengths):
        """ This function calculates the sequence score of all the queries,
        that is, the total number of queries where all the tags are predicted
        correctly.

        Args:
            output_array (ndarray): Output array of the LSTM network
            label_array (ndarray): Label array of the true labels of the data
            seq_lengths (ndarray): A real sequence lengths of each example

        Returns:
            The number of queries where all the tags are correct
        """
        reshaped_output = np.reshape(
            output_array, [-1, int(self.padding_length), self.output_dimension])
        reshaped_output = np.argmax(reshaped_output, 2)
        reshaped_labels = np.argmax(label_array, 2)

        score = 0
        for idx, query in enumerate(reshaped_output):
            seq_len = seq_lengths[idx]
            predicted_tags = reshaped_output[idx][:seq_len]
            actual_tags = reshaped_labels[idx][:seq_len]
            if np.array_equal(predicted_tags, actual_tags):
                score += 1

        return score

    def _construct_lstm_state(self, initializer, hidden_dimension, batch_size, name):
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
            initializer=initializer)

        initial_output_state = tf.get_variable(
            "initial_output_state_{}".format(name),
            shape=[1, hidden_dimension],
            dtype=tf.float32,
            initializer=initializer)

        c_states = tf.tile(initial_cell_state, tf.stack([batch_size, 1]))
        h_states = tf.tile(initial_output_state, tf.stack([batch_size, 1]))

        return tf.contrib.rnn.LSTMStateTuple(c_states, h_states)

    def _construct_regularized_lstm_cell(self, hidden_dimensions, initializer):
        """Construct a regularized lstm cell based on a dropout layer

        Args:
            initializer (tf.contrib.layers.xavier_initializer): initializer used
            hidden_dimensions: num dimensions of the hidden state variable

        Returns:
            (DropoutWrapper): regularized LSTM cell
        """

        lstm_cell = tf.contrib.rnn.CoupledInputForgetGateLSTMCell(
            hidden_dimensions, forget_bias=1.0, initializer=initializer, state_is_tuple=True)

        lstm_cell = tf.contrib.rnn.DropoutWrapper(
            lstm_cell,
            input_keep_prob=self.tf_lstm_input_keep_prob,
            output_keep_prob=self.tf_lstm_output_keep_prob
        )

        return lstm_cell

    def _construct_lstm_network(self, input_tensor):
        """ This function constructs the Bi-Directional LSTM network

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
        lstm_cell_forward = self._construct_regularized_lstm_cell(n_hidden, initializer)
        initial_state_forward = self._construct_lstm_state(
            initializer, n_hidden, batch_size_dim, 'forward')

        # Backward LSTM construction
        lstm_cell_backward = self._construct_regularized_lstm_cell(n_hidden, initializer)
        initial_state_backward = self._construct_lstm_state(
            initializer, n_hidden, batch_size_dim, 'backward')

        # Combined the forward and backward LSTM networks
        (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=lstm_cell_forward,
            cell_bw=lstm_cell_backward,
            inputs=input_tensor,
            sequence_length=self.tf_sequence_length,
            dtype=tf.float32,
            initial_state_fw=initial_state_forward,
            initial_state_bw=initial_state_backward)

        # Construct the output later
        output = tf.concat([output_fw, output_bw], axis=-1)
        output = tf.reshape(output, [-1, 2 * n_hidden])
        output = tf.nn.dropout(output, self.tf_dense_keep_prob)

        weights = tf.get_variable('output_weights', shape=[2 * n_hidden, self.output_dimension],
                                  dtype='float32', initializer=initializer)

        biases = tf.get_variable('output_bias', shape=[self.output_dimension],
                                 dtype='float32', initializer=initializer)

        output_tensor = tf.matmul(output, weights) + biases
        return output_tensor

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
            examples (list of mmworkbench.core.Query): a list of queries
        Returns:
            (tuple): Word embeddings and Gazetteer one-hot embeddings
        """
        x_feats = []
        gaz_feats = []
        for idx, example in enumerate(examples):
            x_feat, gaz_feat = self._extract_features(example)
            x_feats.append(x_feat)
            gaz_feats.append(gaz_feat)

        x_feats = self.query_encoder.get_embeddings_from_encodings(x_feats)
        gaz_feats = self.gaz_encoder.get_embeddings_from_encodings(gaz_feats)
        return x_feats, gaz_feats

    def _extract_features(self, example):
        """Extracts feature dicts for each token in an example.

        Args:
            example (mmworkbench.core.Query): an query
        Returns:
            (list dict): features
        """
        extracted_gaz_tokens = []
        extracted_sequence_features = extract_sequence_features(
            example, self.example_type, self.features, self.resources)

        for index, extracted_gaz in enumerate(extracted_sequence_features):
            if extracted_gaz == {}:
                extracted_gaz_tokens.append(DEFAULT_GAZ_LABEL)
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
                        regex_match.group(REGEX_TYPE_POSITIONAL_INDEX))

            if len(combined_gaz_features) == 0:
                extracted_gaz_tokens.append(DEFAULT_GAZ_LABEL)
            else:
                extracted_gaz_tokens.append(",".join(list(combined_gaz_features)))

        assert len(extracted_gaz_tokens) == len(example.normalized_tokens), \
            "The length of the gaz and example query have to be the same"

        encoded_gaz = self.gaz_encoder.encode_sequence_of_tokens(extracted_gaz_tokens)
        padded_query = self.query_encoder.encode_sequence_of_tokens(example.normalized_tokens)

        return padded_query, encoded_gaz

    def _fit(self, X, y):
        """Trains a classifier without cross-validation. It iterates through
        the data, feeds batches to the tensorflow session graph and fits the
        model based on the feed forward and back propagation steps.

        Args:
            X (list of list of list of str): a list of queries to train on
            y (list of list of str): a list of expected labels
        """
        self.construct_tf_variables()

        self.session.run([tf.global_variables_initializer(),
                          tf.local_variables_initializer()])

        for epochs in range(int(self.number_of_epochs)):
            logger.info("Epoch : {}".format(epochs))

            indices = [x for x in range(len(X))]
            np.random.shuffle(indices)

            gaz = self.gaz_features[indices]
            examples = X[indices]
            labels = y[indices]
            batch_size = int(self.batch_size)
            num_batches = int(math.ceil(len(examples) / batch_size))
            seq_len = np.array(self.sequence_lengths)[indices]

            for batch in range(num_batches):

                batch_start_index = batch * batch_size
                batch_end_index = (batch * batch_size) + batch_size

                batch_examples = examples[batch_start_index:batch_end_index]
                batch_labels = labels[batch_start_index:batch_end_index]
                batch_gaz = gaz[batch_start_index:batch_end_index]
                batch_seq_len = seq_len[batch_start_index:batch_end_index]

                if batch % int(self.display_epoch) == 0:
                    output, loss = self.session.run([self.tf_lstm_output, self.cost],
                                                    feed_dict=self.construct_feed_dictionary(
                                                        batch_examples,
                                                        batch_gaz,
                                                        batch_seq_len,
                                                        batch_labels))

                    score = self._calculate_score(output, batch_labels, batch_seq_len)
                    accuracy = score / (len(batch_examples) * 1.0)

                    logger.info("Iteration number " + str(batch * batch_size) +
                                ", Minibatch Loss= " + "{:.5f}".format(loss) +
                                ", Training Accuracy= " + "{:.5f}".format(accuracy))
                else:
                    self.session.run(self.optimizer,
                                     feed_dict=self.construct_feed_dictionary(
                                         batch_examples,
                                         batch_gaz,
                                         batch_seq_len,
                                         batch_labels))

        return self

    def _predict(self, X):
        """Trains a classifier without cross-validation.

        Args:
            X (list of list of list of str): a list of queries to train on
            params (dict): Parameters of the classifier
        """
        gaz = self.gaz_features
        seq_len = np.array(self.sequence_lengths)

        # During predict time, we make sure no nodes are dropped out
        self.dense_keep_probability = 1.0
        self.lstm_input_keep_prob = 1.0
        self.lstm_output_keep_prob = 1.0

        output = self.session.run(
            [self.tf_lstm_output],
            feed_dict=self.construct_feed_dictionary(X, gaz, seq_len))

        output = np.reshape(output, [-1, int(self.padding_length), self.output_dimension])
        output = np.argmax(output, 2)

        decoded_queries = []
        for idx, encoded_predict in enumerate(output):
            decoded_query = []
            for tag in encoded_predict[:self.sequence_lengths[idx]]:
                decoded_query.append(self.label_encoder.encoding_to_token_mapping[tag])
            decoded_queries.append(decoded_query)

        return decoded_queries

    def dump(self, path='lstm-model'):
        """
        Saves the Tensorflow model
        """
        saver = tf.train.Saver()
        saver.save(self.session, path)
        self.session.close()
        # tf.reset_default_graph()

    def load(self, path='lstm-model'):
        """
        Loads the Tensorflow model
        """
        self.session = tf.Session()
        saver = tf.train.import_meta_graph(path)
        saver.restore(self.session, tf.train.latest_checkpoint('./'))
