from ...path import WORKBENCH_ROOT
import os
import numpy as np

DEFAULT_LABEL = 'B|UNK'


class Embedding:
    """
    This class encodes and constructs embeddings for the tokens of the input queries,
    gazetteers and labels.
    """
    def __init__(self,
                 token_pretrained_embedding_filepath,
                 token_embedding_dimension,
                 gaz_dimension,
                 padding_length):

        self.token_pretrained_embedding_filepath = token_pretrained_embedding_filepath
        self.token_embedding_dimension = token_embedding_dimension
        self.gaz_dimension = gaz_dimension
        self.padding_length = padding_length
        self.word_to_embedding = {}
        self._extract_embeddings()
        self.word_to_encoding = {}
        self.encoding_to_word = {}
        self.gaz_word_to_encoding = {}
        self.gaz_encoding_to_word = {}
        self.gaz_word_to_encoding_extracted = {}
        self.gaz_encoding_to_word_extracted = {}
        self.label_encoding = {}
        self.next_available_token = 0
        self.next_available_gaz_token = 0
        self.next_available_gaz_token_extracted = 0
        self.next_available_label_token = 0

    def _extract_embeddings(self):
        """ Extracts embeddings from the embedding file and stores these vectors in a dictionary
        """
        glove_file_name = self.token_pretrained_embedding_filepath
        glove_lines = open(os.path.abspath(os.path.join(WORKBENCH_ROOT, glove_file_name)))
        for line in glove_lines:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.word_to_embedding[word] = coefs
        glove_lines.close()

    def transform_example(self, list_of_tokens):
        """ Transforms input query into an encoded query using integer tokens
        Args:
            list_of_tokens (list): A list of tokens

        Returns:
            A list of the encoded tokens
        """
        encoded_query = []
        for token in list_of_tokens:
            if token not in self.word_to_encoding:
                self.word_to_encoding[token] = self.next_available_token
                self.encoding_to_word[self.next_available_token] = token
                self.next_available_token += 1
            encoded_query.append(self.word_to_encoding[token])
        return encoded_query

    def transform(self, queries):
        """ Transforms an input list of queries into encoded queries using integer tokens
        Args:
            queries (list): A list of queries

        Returns:
            A list of the encoded queries
        """
        encoded_queries = []
        for query in queries:
            encoded_queries.append(self.transform_example(query))
        return encoded_queries

    def transform_gaz_query(self, list_of_gaz_tokens):
        """
        Transforms a list of gaz tokens to binary encodings

        Args:
            list_of_gaz_tokens (list): A list of gaz tokens

        Returns:
            A list of the binary encodings of the gaz tokens
        """
        encoded_query = []
        for token in list_of_gaz_tokens:
            if token not in self.gaz_word_to_encoding:
                gaz_indices = set(token.split(","))
                for i in gaz_indices:
                    if i not in self.gaz_word_to_encoding_extracted:
                        self.gaz_word_to_encoding_extracted[i] = \
                            self.next_available_gaz_token_extracted
                        self.gaz_encoding_to_word_extracted[
                            self.next_available_gaz_token_extracted] = i
                        self.next_available_gaz_token_extracted += 1

                self.gaz_word_to_encoding[token] = self.next_available_gaz_token
                self.gaz_encoding_to_word[self.next_available_gaz_token] = token
                self.next_available_gaz_token += 1
            encoded_query.append(self.gaz_word_to_encoding[token])
        return encoded_query

    def transform_gaz(self, gaz_queries):
        """
        Transforms a list of gaz tokens to binary encodings

        Args:
            gaz_queries (list): A list of gaz token queries

        Returns:
            A list of the binary encodings of the gaz tokens
        """
        encoded_queries = []
        for query in gaz_queries:
            encoded_queries.append(self.transform_gaz_query(query))
        return encoded_queries

    def get_encoding_matrix(self):
        """
        Constructs the encoding matrix of word encoding to word embedding

        Returns:
            Embedding matrix ndarray
        """
        num_words = len(self.word_to_encoding.keys())
        embedding_matrix = np.zeros((num_words, self.token_embedding_dimension))
        for word, i in self.word_to_encoding.items():
            embedding_vector = self.word_to_embedding.get(word)
            if embedding_vector is None:
                random_word = np.random.uniform(-1, 1, size=(self.token_embedding_dimension,))
                embedding_matrix[i] = random_word
                self.word_to_embedding[word] = random_word
            else:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        return embedding_matrix

    def get_gaz_encoding_matrix(self):
        """
        Constructs the encoding matrix of gaz encoding to gaz embedding

        Returns:
            Embedding matrix ndarray
        """
        gaz_dim = self.gaz_dimension
        num_entites = len(self.gaz_word_to_encoding.keys())
        embedding_matrix_gaz = np.zeros((num_entites, gaz_dim))

        for word, i in self.gaz_word_to_encoding.items():
            in_vec = np.zeros(gaz_dim)
            for entity in word.split(","):
                in_vec[self.gaz_word_to_encoding_extracted[entity]] = 1
            embedding_matrix_gaz[i] = in_vec

        return embedding_matrix_gaz

    def encode_labels(self, labels):
        """Encodes the labels of the queries

        Args:
            labels (list): List of labels

        Returns:
            list of encoded labels
        """
        padding_length = self.padding_length
        transformed_labels = []
        for query_label in labels:
            # We pad the query to the padding length size
            if len(query_label) > padding_length:
                query_label = query_label[:padding_length]
            else:
                diff = padding_length - len(query_label)
                for i in range(diff):
                    query_label.append(DEFAULT_LABEL)

            transformed_labels.append(query_label)

            for label in query_label:
                if label not in self.label_encoding:
                    self.label_encoding[label] = self.next_available_label_token
                    self.next_available_label_token += 1

        num_labels = len(self.label_encoding.keys())
        encoded_labels = []

        for query_label in transformed_labels:
            encoded_sentence_label = []
            for label in query_label:
                encoded_label = np.zeros(num_labels, dtype=np.int)
                encoded_label[self.label_encoding[label]] = 1
                encoded_sentence_label.append(encoded_label)
            encoded_labels.append(encoded_sentence_label)

        return encoded_labels

    @staticmethod
    def transform_query_using_embeddings(encoded_examples, embeddings_matrix):
        """Transform the encoded examples to its respective embeddings based on the
        embeddings matrix. The encoded examples could be queries or gazetteers

        Args:
            encoded_examples (ndarray): encoded examples
            embeddings_matrix (ndarray): embedding matrix

        Returns:
            transformed embedding matrix
        """
        examples_shape = np.shape(encoded_examples)
        final_dimension = np.shape(embeddings_matrix)[1]

        transformed_examples = np.zeros((examples_shape[0], examples_shape[1], final_dimension))

        for query_index in range(len(encoded_examples)):
            for word_index in range(len(transformed_examples[query_index])):
                transformed_examples[query_index][word_index] = \
                    embeddings_matrix[encoded_examples[query_index][word_index]]

        return transformed_examples
