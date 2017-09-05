import numpy as np
import re

from .taggers import Tagger, extract_sequence_features
from .bi_directional_lstm import LstmNetwork
from .embeddings import Embedding
from ..helpers import get_label_encoder

DEFAULT_PADDED_TOKEN = '<UNK>'
DEFAULT_LABEL = 'B|UNK'
DEFAULT_GAZ_LABEL = 'O'
DEFAULT_ENTITY_TOKEN_SPAN_INDEX = 2
GAZ_PATTERN_MATCH = 'in-gaz\|type:(\w+)\|pos:(\w+)\|'
REGEX_TYPE_POSITIONAL_INDEX = 1


class LstmModel(Tagger):
    """"This class encapsulates the bi-directional LSTM model and provides
    the correct interface for use"""

    def fit(self, X, encoded_labels, resources=None):
        examples = np.asarray(X, dtype='float32')
        labels = np.asarray(encoded_labels, dtype='int32')
        gaz = np.asarray(self.gaz, dtype='float32')
        self.config.params['gaz_features'] = gaz
        self._fit(examples, labels, **self.config.params)
        return self

    def process_and_predict(self, examples, config=None, resources=None):
        return self.predict(examples)

    def predict(self, X):
        encoded_examples = np.asarray(X, dtype='float32')
        tags_by_example = self._predict(encoded_examples, **self.config.params)
        seq_lens = self.config.params['sequence_lengths']

        resized_predicted_tags = []
        for query, seq_len in zip(tags_by_example, seq_lens):
            resized_predicted_tags.append(query[:seq_len])

        return resized_predicted_tags

    def set_params(self, **parameters):
        self._clf = LstmNetwork(**parameters)

    def _get_model_constructor(self):
        return LstmNetwork

    def _preprocess_query_data(self, list_of_gold_queries, padding_length):
        queries = []
        for label_query in list_of_gold_queries:
            padded_query = [DEFAULT_PADDED_TOKEN] * padding_length

            max_sequence_length = min(len(label_query.query.normalized_tokens), padding_length)
            for i in range(max_sequence_length):
                padded_query[i] = label_query.query.normalized_tokens[i]
            queries.append(padded_query)
        return queries

    def _extract_seq_length(self, examples):
        """Extract sequence lengths from the input examples
        Args:
            examples (list of Query objects): List of input queries

        Returns:
            (list): List of seq lengths for each query
        """
        seq_lengths = []
        for example in examples:
            if len(example.normalized_tokens) > self.config.params['padding_length']:
                seq_lengths.append(int(self.config.params['padding_length']))
            else:
                seq_lengths.append(len(example.normalized_tokens))

        return seq_lengths

    def extract_features(self, examples, config, resources, y=None, fit=True):
        self.config = config
        self._resources = resources

        self.config.params['gaz_dimension'] = len(self._resources['gazetteers'].keys())

        self.embedding = Embedding(self.config.params)
        self._tag_scheme = self.config.model_settings.get('tag_scheme', 'IOB').upper()
        self._label_encoder = get_label_encoder(self.config)

        # Extract features and classes
        X, self.gaz = self._get_features(examples)

        self.config.params['gaz_features'] = self.gaz
        self.config.params['sequence_lengths'] = self._extract_seq_length(examples)

        if y:
            encoded_labels = self.embedding.encode_labels(y)
            self.config.params['output_dimension'] = len(self.embedding.label_encoding.keys())
            self.config.params['labels_dict'] = self.embedding.label_encoding
        else:
            # Predict time since the label is not available
            encoded_labels = None

        # There are no groups in this model
        groups = None

        return X, encoded_labels, groups

    def setup_model(self, selector_type=None, scale_type=None):
        # This is a no-op since the model setup is taken care by the underlying model
        return

    def _get_features(self, examples):
        """Extracts the word and gazetteer embeddings from the input examples

        Args:
            examples (list of mmworkbench.core.Query): a list of queries
        Returns:
            (list of list of str): features in CRF suite format
        """
        x_feats = []
        gaz_feats = []
        for idx, example in enumerate(examples):
            x_feat, gaz_feat = self._extract_features(example)
            x_feats.append(x_feat)
            gaz_feats.append(gaz_feat)

        embedding_matrix = self.embedding.get_encoding_matrix()
        embedding_gaz_matrix = self.embedding.get_gaz_encoding_matrix()

        x_feats = self.embedding.transform_query_using_embeddings(x_feats, embedding_matrix)
        gaz_feats = self.embedding.transform_query_using_embeddings(gaz_feats, embedding_gaz_matrix)
        return x_feats, gaz_feats

    def _extract_features(self, example):
        """Extracts feature dicts for each token in an example.

        Args:
            example (mmworkbench.core.Query): an query
        Returns:
            (list dict): features
        """
        padding_length = self.config.params['padding_length']

        extracted_gaz_tokens = [DEFAULT_GAZ_LABEL] * padding_length
        extracted_sequence_features = extract_sequence_features(
            example, self.config.example_type, self.config.features, self._resources)

        for index, extracted_gaz in enumerate(extracted_sequence_features):
            if len(extracted_gaz.keys()) > 0 and index < padding_length:
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
                    extracted_gaz_tokens[index] = DEFAULT_GAZ_LABEL
                else:
                    extracted_gaz_tokens[index] = ",".join(list(combined_gaz_features))

        padded_query = [DEFAULT_PADDED_TOKEN] * padding_length
        max_sequence_length = min(len(example.normalized_tokens), padding_length)
        for i in range(max_sequence_length):
            padded_query[i] = example.normalized_tokens[i]

        encoded_gaz = self.embedding.transform_gaz_query(extracted_gaz_tokens)
        padded_query = self.embedding.transform_example(padded_query)

        return padded_query, encoded_gaz

    def _fit(self, X, y, **params):
        """Trains a classifier without cross-validation.

        Args:
            X (list of list of list of str): a list of queries to train on
            y (list of list of str): a list of expected labels
            params (dict): Parameters of the classifier
        """
        self._clf.set_params(**params)
        self._clf.construct_tf_variables()
        return self._clf.fit(X, y)

    def _predict(self, X, **params):
        """Trains a classifier without cross-validation.

        Args:
            X (list of list of list of str): a list of queries to train on
            params (dict): Parameters of the classifier
        """
        self._clf.set_params(**params)
        return self._clf.predict(X)
