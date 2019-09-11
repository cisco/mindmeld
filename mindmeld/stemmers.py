from abc import ABC, abstractmethod, abstractproperty
import nltk


class Stemmer(ABC):

    @abstractproperty
    def _stemmer(self):
        raise NotImplementedError

    @abstractmethod
    def stem_word(self, word):
        """
        Gets the stem of a word. For example, the stem of the word 'fishing' is 'fish'.

        Args:
            word (str): The word to stem

        Returns:
            stemmed word (str): A stemmed version of the word
        """
        raise NotImplementedError


class EnglishNLTKStemmer(Stemmer):

    @property
    def _stemmer(self):
        # lazy init the stemmer
        if not hasattr(self, '__stemmer'):
            setattr(self, '__stemmer', nltk.stem.PorterStemmer())
        return getattr(self, '__stemmer')

    def stem_word(self, word):
        stem = word.lower()

        if self._stemmer.mode == self._stemmer.NLTK_EXTENSIONS and word in self._stemmer.pool:
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
        return word if stem == '' else stem


class SpanishNLTKStemmer(Stemmer):

    @property
    def _stemmer(self):
        # lazy init the stemmer
        if not hasattr(self, '__stemmer'):
            setattr(self, '__stemmer', nltk.stem.SnowballStemmer('spanish'))
        return getattr(self, '__stemmer')

    def stem_word(self, word):
        stem = word.lower()
        stem = self._stemmer.stem(stem)
        return word if stem == '' else stem
