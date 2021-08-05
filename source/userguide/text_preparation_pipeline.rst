Working with the Text Preparation Pipeline
==========================================

The Text Preparation Pipeline

 - is MindMeld's text processing module which handles the preprocessing, tokenization, normalization, and stemming of raw queries
 - has extensible components that can be configured based on language-specific requirements
 - offers components that integrate functionality from Spacy, NLTK and Regex


The :attr:`TextPreparationPipeline` processes text in the following order:

.. image:: /images/text_preparation_pipeline.png
    :align: center
    :name: text_preparation_pipeline


The diagram below is a visual example of a :attr:`TextPreparationPipeline` that uses Spacy for tokenization, Regex for normalization and NLTK for stemming.

.. image:: /images/text_preparation_pipeline_sample_breakdown.png
    :align: center
    :name: text_preparation_pipeline_sample_breakdown

TextPreparationPipeline Configuration
-------------------------------------

The :attr:`DEFAULT_TEXT_PREPARATION_CONFIG` is shown below. Observe that various normalization classes
have been pre-selected by default. To modify the selected components (or to use a subset of the normalization steps), duplicate the
default config and rename it to :attr:`TEXT_PREPARATION_CONFIG`. Place this custom config in :attr:`config.py`. 
If a custom configuration is not defined, the default is used.

.. code-block:: python

    DEFAULT_TEXT_PREPARATION_CONFIG = {
        "preprocessors": [],
        "tokenizer": "WhiteSpaceTokenizer",
        "normalizers": [
            'RemoveAposAtEndOfPossesiveForm',
            'RemoveAdjacentAposAndSpace',
            'RemoveBeginningSpace',
            'RemoveTrailingSpace',
            'ReplaceSpacesWithSpace',
            'ReplaceUnderscoreWithSpace',
            'SeparateAposS',
            'ReplacePunctuationAtWordStartWithSpace',
            'ReplacePunctuationAtWordEndWithSpace',
            'ReplaceSpecialCharsBetweenLettersAndDigitsWithSpace',
            'ReplaceSpecialCharsBetweenDigitsAndLettersWithSpace',
            'ReplaceSpecialCharsBetweenLettersWithSpace',
            'Lowercase',
            'ASCIIFold'
        ],
        "regex_norm_rules": [],
        "stemmer": "EnglishNLTKStemmer"
    }


Let's define the the parameters in the TextPreparationPipeline config:

``'preprocessors'`` (:class:`List[str]`): The preprocessor class to use. (Mindmeld does not currently offer default preprocessors.) 

``'tokenizer'`` (:class:`str`): The tokenization method to split raw queries.

``'normalizers'`` (:class:`List[str]`): List of normalization classes. The text will be normalized sequentially given the order of the normalizers specified.

``'keep_special_chars'`` (:class:`str`): String containing characters to be skipped when normalizing/filtering special characters. This only applies for a subset of default MindMeld normalization rules.

``'regex_norm_rules'`` (:class:`List[Dict]`): Regex normalization rules represented as dictionaries. Each rule should have the key "pattern" and "replacement" which map to a
regex pattern (str) and replacement string, respectively. For example, { "pattern": "_", "replacement": " " }.

``'stemmer`` (:class:`str`): The stemmer class to use.


.. note::

    A Regex normalization rule when added will not overwrite existing normalization rules. To do that, place the key in the config.


Preprocessing
--------------
Preprocessing


Tokenization
-------------


White Space Tokenizer
^^^^^^^^^^^^^^^^^^^^^
The :attr:`WhiteSpaceTokenizer` splits up a sentence by whitespace characters. For example, we can run:

.. code:: python

    from mindmeld.text_preparation.tokenizers import WhiteSpaceTokenizer
    
    sentence = "MindMeld is a Conversational AI Platform."
    white_space_tokenizer = WhiteSpaceTokenizer()
    tokens = white_space_tokenizer.tokenize(sentence)
    print([t['text'] for t in tokens])

We find that the resulting tokens are split by whitespace as expected.

.. code:: python

    ['MindMeld', 'is', 'a', 'Conversational', 'AI', 'Platform.']


Character Tokenizer
^^^^^^^^^^^^^^^^^^^
The :attr:`CharacterTokenizer` splits up a sentence by the individual characters. This can be helpful for languages such as Japanese. Let's break apart the Japanese translation for the phrase "The tall man":

.. code:: python

    from mindmeld.text_preparation.tokenizers import CharacterTokenizer
    
    sentence_ja = "背の高い男性"
    character_tokenizer = CharacterTokenizer()
    tokens = character_tokenizer.tokenize(sentence_ja)
    print([t['text'] for t in tokens])

We see that the original text is split at the character level.

.. code:: python

    ['背', 'の', '高', 'い', '男', '性']


Letter Tokenizer
^^^^^^^^^^^^^^^^^^^
The :attr:`LetterTokenizer` splits text into a separate token if the character proceeds a space, is a
non-latin character, or is a different unicode category than the previous character.

This can be helpful to keep characters of the same type together. Let's look at an example with numbers in a Japanese sentence, "1年は365日". This sentence translates to "One year has 365 days". 

.. code:: python

    from mindmeld.text_preparation.tokenizers import LetterTokenizer

    sentence_ja = "1年は365日"
    letter_tokenizer = LetterTokenizer()
    tokens = letter_tokenizer.tokenize(sentence_ja)
    print([t['text'] for t in tokens])

We see that the original text is split at the character level for non-latin characters but the number "365" remains as an unsegmented token.

.. code:: python

    ['1', '年', 'は', '365', '日']


Spacy Tokenizer
^^^^^^^^^^^^^^^
The :attr:`SpacyTokenizer` splits up a sentence using `Spacy's language models <https://spacy.io/models>`_.
Supported languages include English (en), Spanish (es), French (fr), German (de), Danish (da), Greek (el), Portuguese (pt), Lithuanian (lt), Norwegian Bokmal (nb), Romanian (ro), Polish (pl), Italian (it), Japanese (ja), Chinese (zh), Dutch (nl).
If the required Spacy model is not already present it will automatically downloaded during runtime. 
Let's use the :attr:`SpacyTokenizer` to tokenize the Japanese translation of "The gentleman is gone, no one knows why it happened!": 

.. code:: python

    from mindmeld.text_preparation.tokenizers import SpacyTokenizer
    
    sentence_ja = "紳士が過ぎ去った、 なぜそれが起こったのか誰にも分かりません！"
    spacy_tokenizer_ja = SpacyTokenizer(language="ja", spacy_model_size="lg")
    tokens = spacy_tokenizer_ja.tokenize(sentence_ja)

We see that the original text is split semantically and not simply by whitespace.

.. code:: python

    ['紳士', 'が', '過ぎ', '去っ', 'た', '、', 'なぜ', 'それ', 'が', '起こっ', 'た', 'の', 'か', '誰', 'に', 'も', '分かり', 'ませ', 'ん', '！']


Normalization
--------------

Default Regex Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Talk about default regex_normalization.
As a default in MindMeld, the Tokenizer retains the following special characters in addition to alphanumeric characters and spaces:

1. All currency symbols in UNICODE.
2. Entity annotation symbols ``{, }, |``.
3. Decimal point in numeric values (e.g. ``124.45``).
4. Apostrophe within tokens, such as ``O'Reilly``. Apostrophes at the beginning/end of tokens are removed, say ``Dennis'`` or ``'Tis``.

Setting argument ``keep_special_chars=False`` in the Tokenizer would remove all special characters.

+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| Regex Normalization Rule                            | Description                                                                                      | Example Input                         | Example Output                |
+=====================================================+==================================================================================================+=======================================+===============================+
| RemoveAposAtEndOfPossesiveForm                      | Removes any apostrophe following an 's' at the end of a word.                                    | "dennis' truck"                       | "dennis truck"                |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| RemoveAdjacentAposAndSpace                          | Removes apostrophes followed by a space character and apostrphes that precede a space character. | "havana' "                            | "havana"                      |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| RemoveBeginningSpace                                | Removes extra spaces at the start of a word.                                                     | "      MindMeld"                      | "MindMeld"                    |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| RemoveTrailingSpace                                 | Removes extra spaces at the end of a word.                                                       | "MindMeld       "                     | "MindMeld"                    |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| ReplaceSpacesWithSpace                              | Replaces multiple consecutive spaces with a single space.                                        | "How    are    you?"                  | "How are you?"                |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| ReplaceUnderscoreWithSpace                          | Replaces underscore with a single space.                                                         | "How_are_you?"                        | "How are you?"                |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| SeparateAposS                                       | Adds a space before 's.                                                                          | "mindmeld's code"                     | "mindmeld 's code"            |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| ReplacePunctuationAtWordStartWithSpace              | Replaces special characters infront of words with a space.                                       | "HI %#++=-=SPERO"                     | "HI SPERO"                    |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| ReplacePunctuationAtWordEndWithSpace                | Replaces special characters following words with a space.                                        | "How%+=* are++- you^^%"               | "How are you"                 |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| ReplaceSpecialCharsBetweenLettersAndDigitsWithSpace | Replaces special characters between letters and digits with a space.                             | "Coding^^!#%24 hours#%7 days"         | "Coding 24 hours 7 days"      |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| ReplaceSpecialCharsBetweenDigitsAndLettersWithSpace | Replaces special characters between digits and letters with a space.                             | "Coding 24^^!#%%hours 7##%days"      | "Coding 24 hours 7 days"       |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+
| ReplaceSpecialCharsBetweenLettersWithSpace          | Replaces special characters between letters and letters with a space.                            | "Coding all^^!#%%hours seven##%days" | "Coding all hours seven days"  |
+-----------------------------------------------------+--------------------------------------------------------------------------------------------------+---------------------------------------+-------------------------------+


ASCII Fold Normalization
^^^^^^^^^^^^^^^^^^^^^^^^
The :attr:`ASCIIFold` normalizer converts numeric, symbolic and alphabetic characters which are not in the first 127 ASCII characters (Basic Latin Unicode block) into an ASCII equivalent (if possible).

For example, we can normalize the following Spanish sentence with several accented characters:

.. code:: python

    from mindmeld.text_preparation.normalizers import ASCIIFold
    
    sentence_es = "Ha pasado un caballero, ¡quién sabe por qué pasó!"
    ascii_fold_normalizer = ASCIIFold()
    normalized_text = ascii_fold_normalizer.normalize(sentence_es)
    print(normalized_text)

The accents are removed and the accented characters have been replaced with compatible ASCII equivalents.

.. code:: python

    'Ha pasado un caballero, ¡quien sabe por que paso!'


Unicode Character Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Unicode Character Normalization includes techniques such as :attr:`NFD`, :attr:`NFC`, :attr:`NFKD`, :attr:`NFKC`.
These methods break down characters into their canonical or compatible character equivalents as defined by unicode.
Let's take a look at an example. Say we are trying to normalize the word :attr:`quién` using :attr:`NFKD`.

.. code:: python

    from mindmeld.text_preparation.normalizers import NFKD

    nfd_normalizer = NFKD()
    text = "quién"
    normalized_text = nfd_normalizer.normalize(text)

Interestingly, we find that the normalized text looks identical with the original text, it is not quite the same.

.. code:: python

    >>> print(text, normalized_text)
    >>> quién quién
    >>> print(text == normalized_text)
    >>> False

We can print the character values for each of the texts and observe the the normalization has actually changed the representaation for :attr:`é`.

.. code:: python
    
    >>> print([ord(c) for c in text])
    >>> [113, 117, 105, 233, 110]
    >>> print([ord(c) for c in normalized_text])
    >>> [113, 117, 105, 101, 769, 110]

Stemming
--------
Stemming information.

