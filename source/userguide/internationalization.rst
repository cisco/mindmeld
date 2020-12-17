Internationalization support
============================

MindMeld supports most languages that can be tokenized like English. If the language does not use spaces between
words or has many non-English-like punctuation marks, pre-process the data to remove punctuations and add spaces between words.

Apart from tokenization, there are two optional MindMeld components, Stemming and System entity resolution, that only support a subset of languages.
The limitations of these two components are discussed below.

.. _language_config:

Setting up language configuration
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

MindMeld supports `ISO 639-1 and ISO 639-2 language codes <https://en.wikipedia.org/wiki/List_of_ISO_639-1_codes>`_ and
`ISO 3166-2 locale codes <https://www.iso.org/obp/ui/#search/code/>`_. Locale codes are represented as ISO 639-1 language code
and ISO3166 alpha 2 country code separated by an underscore character, for example, `en_US`.

To use a particular language or locale in MindMeld, the ``config.py`` file needs to configured as follows:

.. code:: python

    LANGUAGE_CONFIG = {
        'language': 'en',
        'locale': 'en_CA'
    }

If the language and locale codes are not configured in ``config.py``, MindMeld uses this default:

.. code:: python

    LANGUAGE_CONFIG = {
        'language': 'en',
        'locale': 'en_US'
    }


Language stemming
^^^^^^^^^^^^^^^^^

Stemming is an important, language-dependent NLP process that transforms a word to an approximation of its root form. Stemming can be
useful for some languages like English but not for others like Vietnamese. MindMeld supports the following ISO 639-1 language codes for stemming:
[EN, DA, NL, AR, FR, DE, HU, IT, NO, PT, RU, RO, ES, SV, FI].

System entity resolution
^^^^^^^^^^^^^^^^^^^^^^^^

For :ref:`system entity resolution <system-entities>`, the following ISO 639-1 language codes are currently supported: [AR, BG, BN, CS, DA,
DE, EL, EN, ES, ET, FI, FR, GA, HE, HI, HR, HU, ID, IS, IT, JA, KA, KN, KM, KO, LO, ML, MN, MY, NB, NE, NL, PL, PT, RO,
RU, SV, SW, TA, TH, TR, UK, VI, ZH].

Moreover, the following ISO 3166-2 locale codes are supported per language:

1. EN: [AU, BZ, CA, GB, IN, IE, JM, NZ, PH, ZA, TT, US]
2. NL: [BE, NL]
3. ZH: [CN, HK, MO, TW]

.. _specify_language:

Locale-based resolution
^^^^^^^^^^^^^^^^^^^^^^^

For languages supported by system entity resolution, one can change the resolution of system entities like `sys_time` entities by varying the
locale in the `process` function call. In the example below, the time entity is resolved differently based on if the locale is `en_CA` or `en_US`.

.. code:: python

    nlp.process('Is the Main Street location open for Thanksgiving', locale='en_CA')

.. code-block:: console

       { 'domain': 'store_info',
         'intent': 'find_nearest_store',
         'entities': [ { 'role': None,
                         'span': {'end': 49, 'start': 37},
                         'text': 'Thanksgiving',
                         'type': 'sys_time',
                         'value': [ { 'grain': 'day',
                                      'value': '2020-10-12T00:00:00.000-07:00'}]}],
         'text': 'Is the Main Street location open for Thanksgiving?'
       }

.. code:: python

    nlp.process('Is the Main Street location open for Thanksgiving', locale='en_US')

.. code-block:: console

       { 'domain': 'store_info',
         'intent': 'find_nearest_store',
         'entities': [ { 'role': None,
                         'span': {'end': 49, 'start': 37},
                         'text': 'Thanksgiving',
                         'type': 'sys_time',
                         'value': [ { 'grain': 'day',
                                      'value': '2020-11-26T00:00:00.000-08:00'}]}],
         'text': 'Is the Main Street location open for Thanksgiving?'
       }
