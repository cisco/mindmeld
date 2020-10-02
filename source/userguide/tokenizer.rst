Working with the Tokenizer
==========================

MindMeld provides the ability to customize and configure the default tokenizer for your application.

The settings for tokenizer can be defined in your application configuration file, ``config.py``. The configuration must be defined as a dictionary with the name :data:`TOKENIZER_CONFIG` to override the tokenizer's default settings. If no custom configuration is defined, the default is used.


Anatomy of the tokenizer configuration
--------------------------------------
The configuration currently has one section: **Allowed Patterns**.

**Allowed Patterns** - Enables defining your custom regular expression patterns in the form of a list of different patterns or combinations. This list is combined and compiled internally by MindMeld and the resulting pattern is applied for filtering out the characters from the user input queries. For eg.

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


Default Tokenizer Configuration
-------------------------------
As a default in MindMeld, the Tokenizer retains the following special characters in addition to alphanumeric characters and spaces:

1. All currency symbols in UNICODE.
2. Entity annotation symbols ``{, }, |``.
3. Decimal point in numeric values (e.g. ``124.45``).
4. Apostrophe within tokens, such as ``O'Reilly``. Apostrophes at the end of tokens are removed, say ``dennis'``.

Setting argument ``keep_special_chars=False`` in the Tokenizer would remove all special characters.
