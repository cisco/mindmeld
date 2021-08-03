Working with the Text Preparation Pipeline
==========================================

The Preprocessor

 - is an optional module that is run before any component in the MindMeld pipeline
 - uses app-specific logic defined by the developer to perform arbitrary text modifications on the user query before it is sent to the :doc:`natural language processor <nlp>`


MindMeld's tokenizer handles both the tokenization and normalization of raw text in your application. These components are configurable based on language-specific needs.


.. image:: /images/architecture1.png
    :align: center
    :name: architecture_diagram

Tokenizer Configuration
----------------------------

The :attr:`DEFAULT_TOKENIZER_CONFIG` shown below is the default config for the Tokenizer.
A custom config can be included in :attr:`config.py` by duplicating the default config and renaming it to :attr:`TOKENIZER_CONFIG`.
If no custom configuration is defined, the default is used.

.. code-block:: python

    DEFAULT_TOKENIZER_CONFIG = {
        "allowed_patterns": default_allowed_patterns,
        "tokenizer": "WhiteSpaceTokenizer",
        "normalizer": "ASCIIFold",
    }


Let's define the the parameters in the Tokenizer config:

``'allowed_patterns'`` (:class:`str`): Enables defining your custom regular expression patterns in the form of a list of different patterns or combinations.
(If :attr:`allowed_patterns` are not provided, then default values will be used.) This list is combined and compiled internally by MindMeld and the resulting pattern is applied for filtering out the characters from the user input queries. For eg.

.. code:: python

   TOKENIZER_CONFIG = {
        "allowed_patterns": ['\w+'],
    }

will allow the system to capture alphanumeric strings and

.. code:: python

   TOKENIZER_CONFIG = {
        "allowed_patterns": ['(\w+\.)$', '(\w+\?)$'],
    }

allows the system to capture only tokens that end with either a question mark or a period.


``'tokenizer'`` (:class:`str`): The tokenization method to split raw queries. Options include :attr:`WhiteSpaceTokenizer`, :attr:`CharacterTokenizer`, and the :attr:`SpacyTokenizer`.

``'normalizer'`` (:class:`str`): The method to normalize raw queries. Options include :attr:`ASCIIFold` and Unicode Character normalization methods such as :attr:`NFD`, :attr:`NFC`, :attr:`NFKD`, :attr:`NFKC`.
For more information on Unicode Chracter Normalization visit the `Unicode Documentation <https://unicode.org/reports/tr15/>`_. Currently, only one normalizer can be selected at a time. 


Tokenizer Methods
------------------


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


Normalization Methods
---------------------

Default MindMeld Normalization
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
As a default in MindMeld, the Tokenizer retains the following special characters in addition to alphanumeric characters and spaces:

1. All currency symbols in UNICODE.
2. Entity annotation symbols ``{, }, |``.
3. Decimal point in numeric values (e.g. ``124.45``).
4. Apostrophe within tokens, such as ``O'Reilly``. Apostrophes at the beginning/end of tokens are removed, say ``Dennis'`` or ``'Tis``.

Setting argument ``keep_special_chars=False`` in the Tokenizer would remove all special characters.

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
