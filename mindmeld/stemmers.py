from abc import ABC, abstractmethod
import nltk


class Stemmer(ABC):

    def __init__(self):
        self._stemmer = None

    def set_stemmer(self):
        """
        This function sets the underlying stemmer. It is lazily initialized
        """
        raise NotImplemented

    @abstractmethod
    def stem_word(self, word):
        """
        Gets the stem of a word. For example, the stem of the word 'fishing' is 'fish'.

        Args:
            word (str): The word to stem

        Returns:
            list(str): A list of the stemmed version of a word.
        """
        raise NotImplemented


class EnglishNLTKStemmer(Stemmer):

    def set_stemmer(self):
        self._stemmer = nltk.stem.PorterStemmer()

    def stem_word(self, word):
        # lazy init the stemmer
        if not self._stemmer:
            self.set_stemmer()

        stem = word.lower()

        if self._stemmer.mode == self._stemmer.NLTK_EXTENSIONS and word in self._stemmer.pool:
            return [self._stemmer.pool[word]]

        if self._stemmer.mode != self._stemmer.ORIGINAL_ALGORITHM and len(word) <= 2:
            # With this line, strings of length 1 or 2 don't go through
            # the stemming process, although no mention is made of this
            # in the published algorithm.
            return [word]

        stem = self._stemmer._step1a(stem)
        stem = self._stemmer._step1b(stem)
        stem = self._stemmer._step1c(stem)
        stem = self._stemmer._step5b(stem)

        # if the stemmed cleaves off the whole token, just return the original one
        if stem == '':
            return [word]
        else:
            return [stem]


class SpanishNLTKStemmer(Stemmer):

    def set_stemmer(self):
        self._stemmer = nltk.stem.SnowballStemmer('spanish')

    def stem_word(self, word):
        # lazy init the stemmer
        if not self._stemmer:
            self.set_stemmer()

        stem = word.lower()
        stem = self._stemmer.stem(stem)

        # if the stemmed cleaves off the whole token, just return the original one
        if stem in (word, ''):
            return [word]
        else:
            word = word.split(stem)[1]
            return [stem + '+', '+' + word]
