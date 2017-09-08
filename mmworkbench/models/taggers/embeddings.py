import os
import sys
import zipfile
import logging

from tqdm import tqdm
import numpy as np

from ...path import WORKBENCH_ROOT
from ...exceptions import PretrainedEmbeddingDownloadError

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
        """Reports update statistics on the download progress

        Args:
            b (int): Number of blocks transferred so far [default: 1]
            bsize (int): Size of each block (in tqdm units) [default: 1]
            tsize (int): Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)  # will also set self.n = b * bsize


class GloVeEmbeddingDict:
    """This class is responsible for the downloading, extraction and storing of
    word embeddings based on the GloVe format"""

    def __init__(self, token_dimension=300, token_pretrained_embedding_filepath=None):

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
        """Returns the word to embedding dict

        Returns:
            (dict): word to embedding mapping
        """
        return self.word_to_embedding

    def _download_embeddings_and_return_zip_handle(self):

        logger.info("Downloading embedding from {}".format(GLOVE_DOWNLOAD_LINK))

        # urllibretrieve is called differently between python 2 and 3.
        if sys.version_info[0] >= 3:
            from urllib.request import urlretrieve
        else:
            # Python 2 support
            from urllib import urlretrieve

        # Make the folder that will contain the embeddings
        if not os.path.exists('data'):
            os.makedirs('data')

        with TqdmUpTo(unit='B', unit_scale=True, miniters=1,
                      desc=GLOVE_DOWNLOAD_LINK) as t:

            try:
                file_handle, _ = urlretrieve(
                    GLOVE_DOWNLOAD_LINK,
                    os.path.join(WORKBENCH_ROOT, EMBEDDINGS_LOCAL_DIR),
                    reporthook=t.update_to)

            except Exception as e:
                logger.error("There was an issue downloading from this "
                             "link {} with the following error: "
                             "{}".format(GLOVE_DOWNLOAD_LINK, e))
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

        if file_location and os.path.isfile(file_location):
            logger.info("Extracting embeddings from provided "
                        "file location {}".format(str(file_location)))
            with open(file_location) as embedding_file:
                self._extract_and_map(embedding_file)
            return

        logger.info("Provided file location {} does not exist".format(str(file_location)))

        file_name = EMBEDDING_FILE_TEMPLATE.format(self.token_dimension)

        zip_folder_location = os.path.join(WORKBENCH_ROOT, EMBEDDINGS_LOCAL_DIR)
        if os.path.isfile(zip_folder_location):
            logger.info("Extracting embeddings from default folder "
                        "location {}".format(zip_folder_location))
            zip_file_object = zipfile.ZipFile(zip_folder_location, 'r')
            with zip_file_object.open(file_name) as embedding_file:
                self._extract_and_map(embedding_file)
            return

        logger.info("Default folder location {} does not exist".format(zip_folder_location))

        zip_file_object = self._download_embeddings_and_return_zip_handle()

        if not zip_file_object:
            raise PretrainedEmbeddingDownloadError("Failed to download embeddings")

        with zip_file_object.open(file_name) as embedding_file:
            self._extract_and_map(embedding_file)
        return


class TokenSequenceEmbedding(object):
    """Base class for encoding a sequence of tokens and transforming the encodings
    into one-hot or word vector based embeddings
    """

    def __init__(self,
                 sequence_padding_length,
                 default_token,
                 use_pretrained_embeddings=False,
                 token_embedding_dimension=None,
                 token_pretrained_embedding_filepath=None):

        self.token_pretrained_embedding_filepath = token_pretrained_embedding_filepath
        self.token_embedding_dimension = token_embedding_dimension
        self.sequence_padding_length = sequence_padding_length

        if use_pretrained_embeddings:
            self.token_to_embedding_mapping = \
                GloVeEmbeddingDict(
                    token_embedding_dimension,
                    token_pretrained_embedding_filepath).get_pretrained_word_to_embeddings_dict()
        else:
            self.token_to_embedding_mapping = {}

        self.token_to_encoding_mapping = {}
        self.encoding_to_token_mapping = {}

        self.available_token_encoding = 0
        self.default_token = default_token

        self.token_encoding_to_embedding_matrix = {}

    def encode_sequence_of_tokens(self, token_sequence):
        """Encodes a sequence of tokens using a simple integer token based approach

        Args:
            token_sequence (list): A sequence of tokens

        Returns:
            (list): Encoded sequence of tokens
        """
        self._encode_token(self.default_token)

        default_encoding = self.token_to_encoding_mapping[self.default_token]
        encoded_query = [default_encoding] * self.sequence_padding_length

        for idx, token in enumerate(token_sequence):
            if idx >= self.sequence_padding_length:
                break

            self._encode_token(token)
            encoded_query[idx] = self.token_to_encoding_mapping[token]

        return encoded_query

    def get_embeddings_from_encodings(self, encoded_sequences):
        """Transform the encoded sequence to its respective embeddings based on the
        embeddings matrix and get it.

        Args:
            encoded_sequences (ndarray): encoded examples

        Returns:
            (ndarray): transformed embedding matrix
        """
        self.token_encoding_to_embedding_matrix = \
            self._construct_embedding_matrix_from_token_encoding()

        examples_shape = np.shape(encoded_sequences)
        final_dimension = np.shape(self.token_encoding_to_embedding_matrix)[1]

        sequence_embeddings = np.zeros((examples_shape[0], examples_shape[1], final_dimension))

        for query_index in range(len(encoded_sequences)):
            for word_index in range(len(sequence_embeddings[query_index])):
                token_encoding = encoded_sequences[query_index][word_index]

                sequence_embeddings[query_index][word_index] = \
                    self.token_encoding_to_embedding_matrix[token_encoding]

        return sequence_embeddings

    def _construct_embedding_matrix_from_token_encoding(self):
        """Constructs the encoding matrix of word encoding to word embedding

        Returns:
            (ndarray): Embedding matrix ndarray
        """
        raise NotImplementedError

    def _encode_token(self, token):
        """Encodes a token to a basic integer based encoding

        Args:
            token (str): Individual token
        """
        if token not in self.token_to_encoding_mapping:
            self.token_to_encoding_mapping[token] = self.available_token_encoding
            self.encoding_to_token_mapping[self.available_token_encoding] = token
            self.available_token_encoding += 1


class WordTokenSequenceEmbedding(TokenSequenceEmbedding):

    def _construct_embedding_matrix_from_token_encoding(self):
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
        self.token_to_gaz_entity_mapping = {}
        self.gaz_entity_to_token_mapping = {}

        self.available_token_encoding = 0
        self.available_token_for_gaz_entity_encoding = 0

        self.default_token = default_token
        self.token_encoding_to_embedding_matrix = {}

    def _construct_embedding_matrix_from_token_encoding(self):
        gaz_dim = self.token_embedding_dimension
        num_entites = len(self.token_to_encoding_mapping.keys())
        embedding_matrix_gaz = np.zeros((num_entites, gaz_dim))

        for word, i in self.token_to_encoding_mapping.items():
            in_vec = np.zeros(gaz_dim)
            for entity in word.split(","):
                in_vec[self.token_to_gaz_entity_mapping[entity]] = 1
            embedding_matrix_gaz[i] = in_vec

        return embedding_matrix_gaz

    def _encode_token(self, token):
        if token not in self.token_to_encoding_mapping:
            individual_gaz_entities = set(token.split(","))
            for gaz_entity in individual_gaz_entities:
                if gaz_entity not in self.token_to_gaz_entity_mapping:
                    self.token_to_gaz_entity_mapping[gaz_entity] = \
                        self.available_token_for_gaz_entity_encoding
                    self.gaz_entity_to_token_mapping[
                        self.available_token_for_gaz_entity_encoding] = gaz_entity
                    self.available_token_for_gaz_entity_encoding += 1

            self.token_to_encoding_mapping[token] = self.available_token_encoding
            self.encoding_to_token_mapping[self.available_token_encoding] = token
            self.available_token_encoding += 1
