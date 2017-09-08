from ...path import WORKBENCH_ROOT
from tqdm import tqdm
import numpy as np
import os
import sys
import zipfile
import logging

logger = logging.getLogger(__name__)

DEFAULT_LABEL = 'B|UNK'
DEFAULT_PADDED_TOKEN = '<UNK>'
DEFAULT_GAZ_LABEL = 'O'

GLOVE_DOWNLOAD_LINK = 'http://nlp.stanford.edu/data/glove.6B.zip'
EMBEDDINGS_LOCAL_DIR = 'data/glove.6B.zip'
EMBEDDING_FILE_TEMPLATE = 'glove.6B.{}d.txt'


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


class GloVeEmbeddingDict:
    def __init__(self,
                 token_dimension=300,
                 token_pretrained_embedding_filepath=None):

        self.token_pretrained_embedding_filepath = \
            token_pretrained_embedding_filepath

        allowed_dims = "50,100,200,300"

        if str(token_dimension) not in allowed_dims:
            logger.info("Token dimension {} not supported, "
                        "chose from these dimensions: {}. "
                        "Selected 300 by default".format(token_dimension,
                                                         allowed_dims))
            self.token_dimension = 300

        self.token_dimension = token_dimension
        self.word_to_embedding = {}
        self._extract_embeddings()

    def get_pretrained_word_to_embeddings_dict(self):
        return self.word_to_embedding

    def _download_embeddings_amd_return_zip_handle(self):
        if sys.version_info[0] >= 3:
            from urllib.request import urlretrieve
        else:
            # Python 2 support
            from urllib import urlretrieve

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                      desc=GLOVE_DOWNLOAD_LINK) as t:

            try:
                file_handle, status = urlretrieve(GLOVE_DOWNLOAD_LINK,
                            os.path.join(WORKBENCH_ROOT, EMBEDDINGS_LOCAL_DIR),
                            reporthook=t.update_to)
            except Exception:
                logger.error("There was an issue downloading this "
                             "link".format(GLOVE_DOWNLOAD_LINK))
                return

            file_name = EMBEDDING_FILE_TEMPLATE.format(self.token_dimension)
            zip_file_object = zipfile.ZipFile(file_handle, 'r')

            if file_name not in zip_file_object.namelist():
                logger.info("Embedding file with {} dimensions "
                            "not found".format(self.token_dimension))
                return

            return zip_file_object

    def _extract_and_map(self, glove_file):
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.word_to_embedding[word] = coefs

    def _extract_embeddings(self):
        file_location = self.token_pretrained_embedding_filepath
        logger.info("Attempting to extract embeddings from {}".format(file_location))
        if file_location and os.path.isfile(file_location):
            logger.info("{} is valid".format(file_location))
            with open(file_location) as embedding_file:
                self._extract_and_map(embedding_file)
            return

        file_name = EMBEDDING_FILE_TEMPLATE.format(self.token_dimension)

        zip_folder_location = os.path.join(WORKBENCH_ROOT, EMBEDDINGS_LOCAL_DIR)
        logger.info("Attempting to extract embeddings from {}".format(zip_folder_location))
        if os.path.isfile(zip_folder_location):
            zip_file_object = zipfile.ZipFile(zip_folder_location, 'r')
            with zip_file_object.open(file_name) as embedding_file:
                self._extract_and_map(embedding_file)
            return

        logger.info("Attempting to download embeddings")
        zip_file_object = self._download_embeddings_amd_return_zip_handle()

        if not zip_file_object:
            logger.error("Failed to download embeddings")
            return

        with zip_file_object.open(file_name) as embedding_file:
            self._extract_and_map(embedding_file)
        return


class TokenSequenceEmbedding(object):
    def __init__(self,
                 sequence_padding_length,
                 default_token,
                 token_embedding_dimension=None,
                 token_pretrained_embedding_filepath=None):

        self.token_pretrained_embedding_filepath = token_pretrained_embedding_filepath
        self.token_embedding_dimension = token_embedding_dimension
        self.sequence_padding_length = sequence_padding_length

        if self.token_pretrained_embedding_filepath:
            self.token_to_embedding_mapping = \
                GloVeEmbeddingDict(
                    token_embedding_dimension,
                    token_pretrained_embedding_filepath).get_pretrained_word_to_embeddings_dict()

        self.token_to_encoding_mapping = {}
        self.encoding_to_token_mapping = {}

        self.available_token_encoding = 0
        self.default_token = default_token
        self.token_encoding_to_embedding_matrix = {}

    def encode_sequence_of_tokens(self, token_sequence):
        self._encode_token(self.default_token)
        encoded_query = [self.token_to_encoding_mapping[self.default_token]] * self.sequence_padding_length

        for idx, token in enumerate(token_sequence):
            if idx >= self.sequence_padding_length:
                break

            self._encode_token(token)
            encoded_query[idx] = self.token_to_encoding_mapping[token]

        return encoded_query

    def _construct_embedding_matrix_from_token_encoding(self):
        raise NotImplementedError

    def _encode_token(self, token):
        if token not in self.token_to_encoding_mapping:
            self.token_to_encoding_mapping[token] = self.available_token_encoding
            self.encoding_to_token_mapping[self.available_token_encoding] = token
            self.available_token_encoding += 1

    def get_embeddings_from_encodings(self, encoded_sequences):
        """Transform the encoded examples to its respective embeddings based on the
        embeddings matrix. The encoded examples could be queries or gazetteers

        Args:
            encoded_examples (ndarray): encoded examples
            embeddings_matrix (ndarray): embedding matrix

        Returns:
            transformed embedding matrix
        """
        self.token_encoding_to_embedding_matrix = self._construct_embedding_matrix_from_token_encoding()

        examples_shape = np.shape(encoded_sequences)
        final_dimension = np.shape(self.token_encoding_to_embedding_matrix)[1]

        sequence_embeddings = np.zeros((examples_shape[0], examples_shape[1], final_dimension))

        for query_index in range(len(encoded_sequences)):
            for word_index in range(len(sequence_embeddings[query_index])):
                sequence_embeddings[query_index][word_index] = \
                    self.token_encoding_to_embedding_matrix[encoded_sequences[query_index][word_index]]

        return sequence_embeddings


class WordTokenSequenceEmbedding(TokenSequenceEmbedding):

    def _construct_embedding_matrix_from_token_encoding(self):
        """
        Constructs the encoding matrix of word encoding to word embedding

        Returns:
            Embedding matrix ndarray
        """
        num_words = len(self.token_to_embedding_mapping.keys())
        embedding_matrix = np.zeros((num_words, self.token_embedding_dimension))
        for word, i in self.token_to_embedding_mapping.items():
            embedding_vector = self.token_to_embedding_mapping.get(word)
            if embedding_vector is None:
                random_word = np.random.uniform(-1, 1, size=(self.token_embedding_dimension,))
                embedding_matrix[i] = random_word
                self.token_to_embedding_mapping[word] = random_word
            else:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


class LabelTokenSequenceEmbedding(TokenSequenceEmbedding):

    def _construct_embedding_matrix_from_token_encoding(self):
        return {}


class GazetteerTokenSequenceEmbedding(TokenSequenceEmbedding):

    def __init__(self,
                 sequence_padding_length,
                 default_token,
                 token_embedding_dimension):

        self.token_embedding_dimension = token_embedding_dimension
        self.sequence_padding_length = sequence_padding_length

        self.token_to_encoding_mapping = {}
        self.encoding_to_token_mapping = {}
        self.token_to_encoding_mapping_extracted = {}
        self.encoding_to_token_mapping_extracted = {}

        self.available_token_encoding = 0
        self.available_token_encoding_extracted = 0

        self.default_token = default_token
        self.token_encoding_to_embedding_matrix = \
            np.zeros((len(self.token_to_encoding_mapping.keys()),
                      self.token_embedding_dimension))

    def _construct_embedding_matrix_from_token_encoding(self):
        """
        Constructs the encoding matrix of gaz encoding to gaz embedding

        Returns:
            Embedding matrix ndarray
        """
        gaz_dim = self.token_embedding_dimension
        num_entites = len(self.token_to_encoding_mapping.keys())
        embedding_matrix_gaz = np.zeros((num_entites, gaz_dim))

        for word, i in self.token_to_encoding_mapping.items():
            in_vec = np.zeros(gaz_dim)
            for entity in word.split(","):
                in_vec[self.token_to_encoding_mapping_extracted[entity]] = 1
            embedding_matrix_gaz[i] = in_vec

        return embedding_matrix_gaz

    def _encode_token(self, token):
        if token not in self.token_to_encoding_mapping:
            gaz_indices = set(token.split(","))
            for i in gaz_indices:
                if i not in self.token_to_encoding_mapping_extracted:
                    self.token_to_encoding_mapping_extracted[i] = \
                        self.available_token_encoding_extracted
                    self.token_to_encoding_mapping_extracted[
                        self.available_token_encoding_extracted] = i
                    self.available_token_encoding_extracted += 1

            self.token_to_encoding_mapping[token] = self.available_token_encoding
            self.encoding_to_token_mapping[self.available_token_encoding] = token
            self.available_token_encoding += 1


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

    def transform_example(self, list_of_tokens):
        """ Transforms input query into an encoded query using integer tokens
        with the appropriate padding

        Args:
            list_of_tokens (list): A list of tokens

        Returns:
            A list of the encoded tokens
        """

        # Encode DEFAULT_LABEL if it has not already
        self._word_encoding_transform(DEFAULT_LABEL)
        encoded_query = [self.word_to_encoding[DEFAULT_LABEL]] * self.padding_length

        for idx, token in enumerate(list_of_tokens):
            if idx >= self.padding_length:
                break

            self._word_encoding_transform(token)
            encoded_query[idx] = self.word_to_encoding[token]

        return encoded_query

    def transform_gaz_query(self, list_of_gaz_tokens):
        """
        Transforms a list of gaz tokens to binary encodings with padding

        Args:
            list_of_gaz_tokens (list): A list of gaz tokens

        Returns:
            A list of the binary encodings of the gaz tokens
        """
        # Encode DEFAULT_GAZ_LABEL if it has not already
        self._gaz_encoding_transform(DEFAULT_GAZ_LABEL)
        encoded_query = [self.gaz_word_to_encoding[DEFAULT_GAZ_LABEL]] * self.padding_length

        for idx, token in enumerate(list_of_gaz_tokens):
            if idx >= self.padding_length:
                break

            self._gaz_encoding_transform(token)
            encoded_query[idx] = self.gaz_word_to_encoding[token]

        return encoded_query

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
        transformed_labels = []
        for query_label in labels:
            # We pad the query to the padding length size
            if len(query_label) > self.padding_length:
                query_label = query_label[:self.padding_length]
            else:
                diff = self.padding_length - len(query_label)
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

    def _gaz_encoding_transform(self, token):
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

    def _word_encoding_transform(self, token):
        if token not in self.word_to_encoding:
            self.word_to_encoding[token] = self.next_available_token
            self.encoding_to_word[self.next_available_token] = token
            self.next_available_token += 1
from ...path import WORKBENCH_ROOT
from tqdm import tqdm
import numpy as np
import os
import sys
import zipfile
import logging

logger = logging.getLogger(__name__)

DEFAULT_LABEL = 'B|UNK'
DEFAULT_PADDED_TOKEN = '<UNK>'
DEFAULT_GAZ_LABEL = 'O'

GLOVE_DOWNLOAD_LINK = 'http://nlp.stanford.edu/data/glove.6B.zip'
EMBEDDINGS_LOCAL_DIR = 'data/glove.6B.zip'
EMBEDDING_FILE_TEMPLATE = 'glove.6B.{}d.txt'


class TqdmUpTo(tqdm):
    """Provides `update_to(n)` which uses `tqdm.update(delta_n)`."""
    def update_to(self, b=1, bsize=1, tsize=None):
        """
        b  : int, optional
            Number of blocks transferred so far [default: 1].
        bsize  : int, optional
            Size of each block (in tqdm units) [default: 1].
        tsize  : int, optional
            Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


class GloVeEmbeddingDict:
    def __init__(self,
                 token_dimension=300,
                 token_pretrained_embedding_filepath=None):

        self.token_pretrained_embedding_filepath = \
            token_pretrained_embedding_filepath

        allowed_dims = "50,100,200,300"

        if str(token_dimension) not in allowed_dims:
            logger.info("Token dimension {} not supported, "
                        "chose from these dimensions: {}. "
                        "Selected 300 by default".format(token_dimension,
                                                         allowed_dims))
            self.token_dimension = 300

        self.token_dimension = token_dimension
        self.word_to_embedding = {}
        self._extract_embeddings()

    def get_pretrained_word_to_embeddings_dict(self):
        return self.word_to_embedding

    def _download_embeddings_amd_return_zip_handle(self):
        if sys.version_info[0] >= 3:
            from urllib.request import urlretrieve
        else:
            # Python 2 support
            from urllib import urlretrieve

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                      desc=GLOVE_DOWNLOAD_LINK) as t:

            try:
                file_handle, status = urlretrieve(GLOVE_DOWNLOAD_LINK,
                            os.path.join(WORKBENCH_ROOT, EMBEDDINGS_LOCAL_DIR),
                            reporthook=t.update_to)
            except Exception:
                logger.error("There was an issue downloading this "
                             "link".format(GLOVE_DOWNLOAD_LINK))
                return

            file_name = EMBEDDING_FILE_TEMPLATE.format(self.token_dimension)
            zip_file_object = zipfile.ZipFile(file_handle, 'r')

            if file_name not in zip_file_object.namelist():
                logger.info("Embedding file with {} dimensions "
                            "not found".format(self.token_dimension))
                return

            return zip_file_object

    def _extract_and_map(self, glove_file):
        for line in glove_file:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            self.word_to_embedding[word] = coefs

    def _extract_embeddings(self):
        file_location = self.token_pretrained_embedding_filepath
        logger.info("Attempting to extract embeddings from {}".format(file_location))
        if file_location and os.path.isfile(file_location):
            logger.info("{} is valid".format(file_location))
            with open(file_location) as embedding_file:
                self._extract_and_map(embedding_file)
            return

        file_name = EMBEDDING_FILE_TEMPLATE.format(self.token_dimension)

        zip_folder_location = os.path.join(WORKBENCH_ROOT, EMBEDDINGS_LOCAL_DIR)
        logger.info("Attempting to extract embeddings from {}".format(zip_folder_location))
        if os.path.isfile(zip_folder_location):
            zip_file_object = zipfile.ZipFile(zip_folder_location, 'r')
            with zip_file_object.open(file_name) as embedding_file:
                self._extract_and_map(embedding_file)
            return

        logger.info("Attempting to download embeddings")
        zip_file_object = self._download_embeddings_amd_return_zip_handle()

        if not zip_file_object:
            logger.error("Failed to download embeddings")
            return

        with zip_file_object.open(file_name) as embedding_file:
            self._extract_and_map(embedding_file)
        return


class TokenSequenceEmbedding(object):
    def __init__(self,
                 sequence_padding_length,
                 default_token,
                 token_embedding_dimension=None,
                 token_pretrained_embedding_filepath=None):

        self.token_pretrained_embedding_filepath = token_pretrained_embedding_filepath
        self.token_embedding_dimension = token_embedding_dimension
        self.sequence_padding_length = sequence_padding_length

        self.token_to_embedding_mapping = \
            GloVeEmbeddingDict(
                token_embedding_dimension,
                token_pretrained_embedding_filepath).get_pretrained_word_to_embeddings_dict()

        self.token_to_encoding_mapping = {}
        self.encoding_to_token_mapping = {}

        self.available_token_encoding = 0
        self.default_token = default_token

        if self.token_embedding_dimension:
            self.token_encoding_to_embedding_matrix = \
                np.zeros((len(self.token_to_encoding_mapping.keys()),
                          self.token_embedding_dimension))

    def encode_sequence_of_tokens(self, token_sequence):
        self._encode_token(self.default_token)
        encoded_query = [self.token_to_encoding_mapping[self.default_token]] * self.sequence_padding_length

        for idx, token in enumerate(token_sequence):
            if idx >= self.sequence_padding_length:
                break

            self._encode_token(token)
            encoded_query[idx] = self.token_to_encoding_mapping[token]

        return encoded_query

    def _construct_embedding_matrix_from_token_encoding(self):
        raise NotImplementedError

    def _encode_token(self, token):
        if token not in self.token_to_encoding_mapping:
            self.token_to_encoding_mapping[token] = self.available_token_encoding
            self.encoding_to_token_mapping[self.available_token_encoding] = token
            self.available_token_encoding += 1

    def get_embeddings_from_encodings(self, encoded_sequences):
        """Transform the encoded examples to its respective embeddings based on the
        embeddings matrix. The encoded examples could be queries or gazetteers

        Args:
            encoded_examples (ndarray): encoded examples
            embeddings_matrix (ndarray): embedding matrix

        Returns:
            transformed embedding matrix
        """
        self.token_encoding_to_embedding_matrix = self._construct_embedding_matrix_from_token_encoding()

        examples_shape = np.shape(encoded_sequences)
        final_dimension = np.shape(self.token_encoding_to_embedding_matrix)[1]

        sequence_embeddings = np.zeros((examples_shape[0], examples_shape[1], final_dimension))

        for query_index in range(len(encoded_sequences)):
            for word_index in range(len(sequence_embeddings[query_index])):
                sequence_embeddings[query_index][word_index] = \
                    self.token_encoding_to_embedding_matrix[encoded_sequences[query_index][word_index]]

        return sequence_embeddings


class WordTokenSequenceEmbedding(TokenSequenceEmbedding):

    def _construct_embedding_matrix_from_token_encoding(self):
        """
        Constructs the encoding matrix of word encoding to word embedding

        Returns:
            Embedding matrix ndarray
        """
        num_words = len(self.token_to_encoding_mapping.keys())
        embedding_matrix = np.zeros((num_words, self.token_embedding_dimension))
        for word, i in self.token_to_encoding_mapping.items():
            embedding_vector = self.token_to_embedding_mapping.get(word)
            if embedding_vector is None:
                random_word = np.random.uniform(-1, 1, size=(self.token_embedding_dimension,))
                embedding_matrix[i] = random_word
                self.token_to_embedding_mapping[word] = random_word
            else:
                embedding_matrix[i] = embedding_vector
        return embedding_matrix


class LabelTokenSequenceEmbedding(TokenSequenceEmbedding):

    def _construct_embedding_matrix_from_token_encoding(self):
        num_words = len(self.token_to_encoding_mapping.keys())
        embedding_matrix = np.zeros((num_words, num_words))

        for word, i in self.token_to_encoding_mapping.items():
            in_vec = np.zeros(num_words)
            in_vec[self.token_to_encoding_mapping[word]] = 1
            embedding_matrix[i] = in_vec
        return embedding_matrix


class GazetteerTokenSequenceEmbedding(TokenSequenceEmbedding):

    def __init__(self,
                 sequence_padding_length,
                 default_token,
                 token_embedding_dimension):

        self.token_embedding_dimension = token_embedding_dimension
        self.sequence_padding_length = sequence_padding_length

        self.token_to_encoding_mapping = {}
        self.encoding_to_token_mapping = {}
        self.token_to_encoding_mapping_extracted = {}
        self.encoding_to_token_mapping_extracted = {}

        self.available_token_encoding = 0
        self.available_token_encoding_extracted = 0

        self.default_token = default_token
        self.token_encoding_to_embedding_matrix = \
            np.zeros((len(self.token_to_encoding_mapping.keys()),
                      self.token_embedding_dimension))

    def _construct_embedding_matrix_from_token_encoding(self):
        """
        Constructs the encoding matrix of gaz encoding to gaz embedding

        Returns:
            Embedding matrix ndarray
        """
        gaz_dim = self.token_embedding_dimension
        num_entites = len(self.token_to_encoding_mapping.keys())
        embedding_matrix_gaz = np.zeros((num_entites, gaz_dim))

        for word, i in self.token_to_encoding_mapping.items():
            in_vec = np.zeros(gaz_dim)
            for entity in word.split(","):
                in_vec[self.token_to_encoding_mapping_extracted[entity]] = 1
            embedding_matrix_gaz[i] = in_vec

        return embedding_matrix_gaz

    def _encode_token(self, token):
        if token not in self.token_to_encoding_mapping:
            gaz_indices = set(token.split(","))
            for i in gaz_indices:
                if i not in self.token_to_encoding_mapping_extracted:
                    self.token_to_encoding_mapping_extracted[i] = \
                        self.available_token_encoding_extracted
                    self.token_to_encoding_mapping_extracted[
                        self.available_token_encoding_extracted] = i
                    self.available_token_encoding_extracted += 1

            self.token_to_encoding_mapping[token] = self.available_token_encoding
            self.encoding_to_token_mapping[self.available_token_encoding] = token
            self.available_token_encoding += 1
