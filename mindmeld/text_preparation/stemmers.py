import logging
from abc import ABC, abstractmethod

import nltk
import pycountry

from ..components._config import ENGLISH_LANGUAGE_CODE

logger = logging.getLogger(__name__)


class Stemmer(ABC):

    def __init__(self):
        """Creates a Stemmer instance."""

    @abstractmethod
    def stem_word(self, word):
        """
        Gets the stem of a word. For example, the stem of the word 'fishing' is 'fish'.

        Args:
            word (str): The word to stem

        Returns:
            stemmed_word (str): A stemmed version of the word
        """
        raise NotImplementedError

    def tojson(self):
        """
        Method defined to obtain recursive JSON representation of a TextPreparationPipeline.

        Args:
            None.

        Returns:
            JSON representation of TextPreparationPipeline (dict) .
        """
        return {self.__class__.__name__: None}


class NoOpStemmer(Stemmer):

    def stem_word(self, word):
        return word


class EnglishNLTKStemmer(Stemmer):
    @property
    def _stemmer(self):
        # lazy init the stemmer
        if not hasattr(self, "__stemmer"):
            setattr(self, "__stemmer", nltk.stem.PorterStemmer())
        return getattr(self, "__stemmer")

    def stem_word(self, word):
        stem = word.lower()

        if (
            self._stemmer.mode == self._stemmer.NLTK_EXTENSIONS
            and word in self._stemmer.pool
        ):
            return self._stemmer.pool[word]

        if self._stemmer.mode != self._stemmer.ORIGINAL_ALGORITHM and len(word) <= 2:
            # With this line, strings of length 1 or 2 don't go through
            # the stemming process, although no mention is made of this
            # in the published algorithm.
            return word

        stem = self._stemmer._step1a(stem)
        stem = self._stemmer._step1b(stem)
        stem = self._stemmer._step1c(stem)
        stem = self._stemmer._step5b(stem)
        return word if stem == "" else stem


class SnowballNLTKStemmer(Stemmer):

    def __init__(self, language=None):
        self.language = language

    @property
    def _stemmer(self):
        # lazy init the stemmer
        if not hasattr(self, "__stemmer"):
            setattr(self, "__stemmer", nltk.stem.SnowballStemmer(self.language))
        return getattr(self, "__stemmer")

    def stem_word(self, word):
        stem = word.lower()
        stem = self._stemmer.stem(stem)
        return word if stem == "" else stem


class StemmerFactory:
    """Stemmer Factory Class"""

    @staticmethod
    def get_stemmer(stemmer: str):
        """A static method to get a stemmer.

        Args:
            stemmer (str): Name of the desired translator class
        Returns:
            (Stemmer): Stemmer Class
        """

        stemmer_classes = {
            EnglishNLTKStemmer.__name__: EnglishNLTKStemmer,
            SnowballNLTKStemmer.__name__: SnowballNLTKStemmer,
        }
        stemmer_class = stemmer_classes.get(stemmer)
        if not stemmer_class:
            raise TypeError(f"{stemmer} is not a valid Stemmer type.")
        return stemmer_class()

    @staticmethod
    def get_stemmer_by_language(language_code):

        if not language_code:
            return NoOpStemmer()

        language_code = language_code.lower()

        if language_code == ENGLISH_LANGUAGE_CODE:
            return EnglishNLTKStemmer()

        language = StemmerFactory.get_language_from_language_code(language_code)

        if not language:
            logger.warning(
                'Language code "%s" is not supported for stemming.', language_code
            )
            return NoOpStemmer()

        language_name = language.name.lower()
        if language_name in nltk.stem.SnowballStemmer.languages:
            return SnowballNLTKStemmer(language_name)

        logger.warning(
            'Language code "%s" is not supported for stemming.', language_code
        )
        return NoOpStemmer()

    @staticmethod
    def get_language_from_language_code(language_code):
        language = None
        if len(language_code) == 2:
            language = pycountry.languages.get(alpha_2=language_code)
        elif len(language_code) == 3:
            language = pycountry.languages.get(alpha_3=language_code)
        return language
