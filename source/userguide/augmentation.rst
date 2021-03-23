Data Augmentation in MindMeld
=============================

MindMeld provides the ability to augment data to collect more diverse queries and make the applications more robust.

Quick Start
-----------
This section briefly explains the use of the ``augment`` command.

The command below can be used to augment data across all training and testing files in the application, given a language code.

Command-line:

.. code-block:: console

    mindmeld augment --app-path <app_path> --lang <lang_code> --num-augmentations <int>

The following section explains this in more detail.

Using the Augmentors
--------------------

The data augmentation tool in MindMeld is a command line functionality. We demonstrate below the use-cases and configurations that
can be defined to get the best augmentation results based on the application.

Currently, we support data augmentation through paraphrasing for the following languages: English (en), French (fr), Italian (it), Portugese (pt),
Romanian (ro), Spanish (es). This is done with the use of two models: the English paraphraser and the multi-lingual paraphraser.

First, we will discuss the configuration needed to initiate an augmentor and then follow it with detailed descriptions of the paraphrasers.

Augmentation Configuration
--------------------------

The :attr:`DEFAULT_AUGMENTATION_CONFIG` shown below is the default config for an Augmentor.
A custom config can be included in :attr:`config.py` by duplicating the default config and renaming it to :attr:`AUGMENTATION_CONFIG`.

.. code-block:: python

    DEFAULT_AUGMENTATION_CONFIG = {
        "augmentor_class": "EnglishParaphraser",
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

``'paths'`` (:class:`list`): List of regex based path rules to select files to be augmented. These files become the baseline for the augmentors and are used to generate new data. By default, all training and testing files are augmented.

``'path_suffix'`` (:class:`str`): The default suffix that is appended to the name of the original file being augmented to generate new files with the augmented data.

English Paraphraser
-------------------

The English paraphraser uses a state-of-the-art text summarization model PEGASUS for generating paraphrases.

Usage
^^^^^

.. code-block:: console

    mindmeld augment --app-path <app_path> --lang "en" --num-augmentations "10"

In the config for this paraphraser class, the ``'augmentor_class'`` should be set to :class:`EnglishParaphraser`.


Multi-Lingual Paraphraser
-------------------------

The multi-lingual paraphraser in MindMeld uses back-translation as the underlying concept to generate paraphrases. Given an application in one of the supported languages, the forward model translates the current set of queries to English, generating a number of English translations. Next, the reverse model translates each of the English translations into one or more queries in the original language. This results in a paraphrased set of queries in the original language.

Currently, we support the following languages:

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

    mindmeld augment --app-path <app_path> --lang "code" --num-augmentations "10"

In the config for this paraphraser class, the ``'augmentor_class'`` should be set to :class:`MultiLingualParaphraser`.
