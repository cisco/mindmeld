Working with the Preprocessor
=============================

The Preprocessor

 - is an optional module that is run before any component in the MindMeld pipeline
 - uses app-specific logic defined by the developer to perform arbitrary text modifications on the user query before it is sent to the :doc:`natural language processor <nlp>`

Examples of some common preprocessing tasks include spelling correction, punctuation removal, handling special characters, sentence segmentation, and other kinds of application-specific text normalization.


Implement the preprocessor
--------------------------

You can use the :class:`Preprocessor` abstract class provided by MindMeld as a template for defining your own custom preprocessing logic.

The base class contains two methods:

 - :meth:`process`: takes in a text string and returns the processed string
 - :meth:`get_char_index_map`: generates the character mapping from the processed string to the original text string.

While implementing the :meth:`process` method is mandatory and key to defining the functionality of your preprocessor, implementation of the :meth:`get_char_index_map` is optional. However, to ensure the proper functioning of downstream NLP components, it is essential that you implement this function whenever the length (character count) of the input query is changed by the :meth:`process` method.


Example: Stemming as a preprocessing step
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In this example, we define a basic preprocessor that does `stemming <https://en.wikipedia.org/wiki/Stemming>`_ using the `Porter Stemmer <https://tartarus.org/martin/PorterStemmer/>`_ implementation from `NLTK <https://www.nltk.org/>`_. Stemming reduces each word to a base or root form called the word stem. E.g., the words "fish", "fishes" and "fishing" all have the same stem, "fish". This kind of preprocessing is widely used in information retrieval systems as a `query expansion <https://en.wikipedia.org/wiki/Query_expansion>`_ technique to improve search engine recall.

You will need to install the NLTK package to try out the example below. Add the following code to a file named ``stem_processor.py`` and put it in your application folder.

.. code:: python
  :caption: stem_processor.py

    from mindmeld.components import Preprocessor
    from nltk.stem import PorterStemmer


    class StemProcessor(Preprocessor):
        """
        An example or preprocessor for stemming
        """
        def __init__(self):
            self.stemmer = PorterStemmer()

        def process(self, text):
            """ StemProcessor will take in a query text and process it to return the stemmed version of the query.
            Args:
              text (str)

            Returns:
              (str)
            """
            return self.stemmer.stem(text)

        def get_char_index_map(self, raw_text, processed_text):
            """ This function is used to map between the raw text and the processed text, making certain assumptions between the raw text and the processed text.
            We return the 1-1 mapping between the two strings, which the Dialogue Manager uses to compute the mapping between the entity spans in the processed text to the entity spans in the raw text.

            Args:
              raw_text (str)
              processed_text (str)

            Returns:
              (dict)
            """

            raw_tokens = raw_text.split(' ')
            processed_tokens = processed_text.split(' ')

            if len(raw_tokens) != len(processed_tokens):
                raise Exception('Stemming should not change the number of tokens!')

            forward = {}
            raw_index = 0
            processed_index = 0
            for i, raw_token in enumerate(raw_tokens):
                processed_token = processed_tokens[i]
                for character_count in range(len(raw_token)):
                    forward[raw_index + character_count] = min(processed_index + character_count,
                                                               processed_index + len(processed_token) - 1)

                if raw_index + len(raw_token) < len(raw_text):
                    forward[raw_index + len(raw_token)] = processed_index + len(processed_token)
                    raw_index += (len(raw_token) + 1)
                    processed_index += (len(processed_token) + 1)

            backward = {}
            for character_index in forward:
                if forward[character_index] not in backward:
                    backward[forward[character_index]] = character_index

            return forward, backward


Use the preprocessor
--------------------

To use your custom preprocessing logic within your MindMeld application, pass in an instance of your implemented preprocessor class when initializing the :class:`Application` object in the application container file, ``__init__.py``.

.. code:: python
  :caption: __init__.py

  from mindmeld import Application
  from .stem_processor import StemProcessor

  preprocessor = StemProcessor()
  app = Application(__name__, preprocessor=preprocessor)
