Data Augmentation in MindMeld
=============================

When starting out to build a new conversational application in a custom domain, it is often the case that we have very limited training data to train models. In such scenarios, one way of bootstrapping these models is by using data augmentation techniques to increase the size of our training set. MindMeld now provides the ability to augment data via automatic paraphrasing in English and other languages to help make applications more robust.

Quick Start
-----------
This section briefly explains the use of the ``augment`` command.

The command below can be used to augment data across all training and testing files in the application, given a language code (specified using the ``--language`` or ``-l`` flags).

Command-line:

.. code-block:: console

    mindmeld augment --app_path <app_path> --language <lang_code>

The following section explains this in more detail.

Using the Augmentors
--------------------

The data augmentation tool in MindMeld is a command line functionality. We demonstrate below the use-cases and configurations that
can be defined to get the best augmentation results based on the application.

Currently, we support data augmentation through paraphrasing for the following languages (with codes in ISO 639-1 format): English (en), French (fr), Italian (it), Portugese (pt), Romanian (ro), Spanish (es). This is done with the use of two models: the English paraphraser and the multi-lingual paraphraser.

First, we will discuss the configuration needed to initiate an augmentor and then follow it with detailed descriptions of the paraphrasers.

.. note::

    Install the additional dependencies for augmentors.

    .. code-block:: console

        pip install mindmeld[augment]

    or in a zsh shell:

    .. code-block:: console

        pip install mindmeld"[augment]"

Augmentation Configuration
--------------------------

The :attr:`DEFAULT_AUGMENTATION_CONFIG` shown below is the default config for an Augmentor.
A custom config can be included in :attr:`config.py` by duplicating the default config and renaming it to :attr:`AUGMENTATION_CONFIG`.

.. code-block:: python

    DEFAULT_AUGMENTATION_CONFIG = {
        "augmentor_class": "EnglishParaphraser",
        "batch_size": 8,
        "retain_entities": False,
        "paths": [
            {
                "domains": ".*",
                "intents": ".*",
                "files": ".*",
            }
        ],
        "path_suffix": "-augmented.txt"
    }

The following values can be updated in the config to customize it:

``'augmentor_class'`` (:class:`str`, default: :class:`EnglishParaphraser`): The augmentor class that can specified for deciding on the augmentation tool to be used. Currently, we recommend using :class:`EnglishParaphraser` for English language applications and :class:`MultiLingualParaphraser` for all other supported languages.

``'batch_size'`` (:class:`int`, default: 8): The number of queries to be batched and processed together by the augmentation models. This size can be modified according to system memory constraints.

``'retain_entities'`` (:class:`bool`, default: ``False``): Whether entity values and annotations should be retained. This is only applicable for the :class:`EnglishParaphraser`.

``'paths'`` (:class:`list`): List of regex based path rules to select files to be augmented. These files become the baseline for the augmentors and are used to generate new data. By default, all training and testing files are augmented.

``'path_suffix'`` (:class:`str`): The default suffix that is appended to the name of the original file being augmented to generate new files with the augmented data.

English Paraphraser
-------------------

The English paraphraser uses a state-of-the-art text summarization model `PEGASUS <https://ai.googleblog.com/2020/06/pegasus-state-of-art-model-for.html>`_ for generating paraphrases. This model has been fine-tuned for the task of paraphrasing.

Usage
^^^^^

.. code-block:: console

    mindmeld augment --app_path <app_path> --language "en"

In the config for this paraphraser class, the ``'augmentor_class'`` should be set to :class:`EnglishParaphraser`.

Retaining Entity Annotations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The default English model ignores entity annotations and paraphrases the full query (along with entities). If you wish to retain the original query's entity values and annotations in the paraphrases, you can set the ``'retain_entities'`` key to ``True`` in the config.
In addition, if a gazetteer is provided for an entity type, the system will automatically replace entity values with randomly sampled entities from the gazetteer. This is recommended to improve the n-gram diversity in your queries.

.. note::

    The default setting uses the Pegasus model fine-tuned for the paraphrasing task from `Hugging Face <https://huggingface.co/tuner007/pegasus_paraphrase>`_.
    A `custom model <https://mindmeld-binaries.s3.amazonaws.com/paraphraser/paraphrase_retain_entities.zip>`_ is used when ``'retain_entities'`` is set to ``True``.

Multi-Lingual Paraphraser
-------------------------

The multi-lingual paraphraser in MindMeld uses machine-translation as the underlying concept to generate paraphrases. Given an application in one of the supported languages, the forward model translates the current set of queries to English, generating a number of English translations. Next, the reverse model translates each of the English translations into one or more queries in the original language. This results in a paraphrased set of queries in the original language.

Currently, we support the following languages with their codes in the ISO 639-1 format:

+--------------+-------+
| Language     | Code  |
+==============+=======+
| French       | fr    |
+--------------+-------+
| Italian      | it    |
+--------------+-------+
| Portugese    | pt    |
+--------------+-------+
| Romanian     | ro    |
+--------------+-------+
| Spanish      | es    |
+--------------+-------+


Usage
^^^^^

.. code-block:: console

    mindmeld augment --app_path <app_path> --language "code"

In the config for this paraphraser class, the ``'augmentor_class'`` should be set to :class:`MultiLingualParaphraser`.

.. note::

    We use both `forward <https://huggingface.co/Helsinki-NLP/opus-mt-ROMANCE-en>`_ and `reverse <https://huggingface.co/Helsinki-NLP/opus-mt-en-ROMANCE>`_ machine-translation models from Hugging Face.
