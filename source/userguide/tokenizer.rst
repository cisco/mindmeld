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