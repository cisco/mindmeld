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
        "params": {
            "fwd_params": {},
            "reverse_params": {},
            "batch_size": 8,
        },
        "path_suffix": "-augmented.txt"
    }

The following values can be updated in the config to customize it:

``'augmentor_class'`` (:class:`str`, default: :class:`EnglishParaphraser`): The augmentor class that can specified for deciding on the augmentation tool to be used. Currently, we recommend using :class:`EnglishParaphraser` for English language applications and :class:`MultiLingualParaphraser` for all other supported languages.

``'paths'`` (:class:`list`): List of regex based path rules to select files to be augmented. These files become the baseline for the augmentors and are used to generate new data. By default, all training and testing files are augmented.

``'params'`` (:class:`dict`): This configuration setting can be used to customize the model settings by providing relevant parameters under each of the following categories: ``fwd_params``, ``reverse_params`` and ``batch_size``. ``batch_size`` (:class:`int`, default: 8) is the size of queries that are processed at once for augmentation. We explain the ``fwd_params`` and ``reverse_params`` in model specific configurations later.

``'path_suffix'`` (:class:`str`): The default suffix that is appended to the name of the original file being augmented to generate new files with the augmented data.

English Paraphraser
-------------------

The English paraphraser uses a state-of-the-art text summarization model PEGASUS for generating paraphrases.

Usage
^^^^^

.. code-block:: console

    mindmeld augment --app-path <app_path> --lang "en" --num-augmentations "10"

For this paraphraser class, the config can be customized for the following values:

``'augmentor_class'`` should be set to :class:`EnglishParaphraser`

In ``'params'``, the ``fwd_params`` sub-field can be provided with additional keys to customize the generation parameters for the model.

``num_return_sequences`` (:class:`int`, max: 10, default: 10): Maximum number of paraphrases to be generated per original query. This creates an upperbound to control the total number of paraphrases that the system generates. (Note: ``num_return_sequences`` <= ``num_beams``).

``num_beams`` (:class:`int`, default: 10): Number of generation beams to search through when using beam search. (Note: ``num_return_sequences`` <= ``num_beams``).

``max_length`` (:class:`int`, default: 60): Maximum output length for the generated paraphrases.

``do_sample`` (:class:`bool`, default: False):  This flag can be set to true to activate sampling instead of beam searh. Sampling means randomly picking the next word in the generation process according to its conditional probability distribution.

``top_k`` (:class:`int`, default: 0): Most useful with sampling, this setting selects the K most likely next words, filters them out and redistributes the probability mass among only these words. This reduces the search space for the next word and makes a better prediction.

``top_p`` (:class:`int`, default: 1): Here, the sampling method filters out the smallest set of words possible whose collective probability exceeds the probability p. This reduces the search space for the next word and makes a better prediction.

``temperature`` (:class:`int`, default: 1.5): This setting allows for setting a temperature to the softmax (probability prediction) output and is useful for decreasing sesitivity towards low probability candidates for the next word. This again, improves the search space for the next word.


.. note::

    For a more detailed understanding on using these parameteres for beam search, sampling and their variants of Natural Language Generation techniques, refer to this `blog <https://huggingface.co/blog/how-to-generate>`_

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

For this paraphraser class, the config can be customized for the following values:

``'augmentor_class'`` should be set to :class:`MultiLingualParaphraser`

In ``'params'``, the multi-lingual model uses both the forward model and the reverse model. Hence, both the ``fwd_params`` and ``reverse_params`` sub-fields can be customized for the forward and reverse models respectively.

For both models, the following parameter keys can be modified:

``num_return_sequences`` (:class:`int`, max: 10, default: 10): Maximum number of paraphrases to be generated per original query. This creates an upperbound to control the total number of paraphrases that the system generates. (Note: ``num_return_sequences`` <= ``num_beams``).

``num_beams`` (:class:`int`, default: 10): Number of generation beams to search through when using beam search. (Note: ``num_return_sequences`` <= ``num_beams``).

``max_length`` (:class:`int`, default: 60): Maximum output length for the generated paraphrases.

``do_sample`` (:class:`bool`, default: False):  This flag can be set to true to activate sampling instead of beam searh. Sampling means randomly picking the next word in the generation process according to its conditional probability distribution.

``top_k`` (:class:`int`, default: 0): Most useful with sampling, this setting selects the K most likely next words, filters them out and redistributes the probability mass among only these words. This reduces the search space for the next word and makes a better prediction.

``top_p`` (:class:`int`, default: 1): Here, the sampling method filters out the smallest set of words possible whose collective probability exceeds the probability p. This reduces the search space for the next word and makes a better prediction.

``temperature`` (:class:`int`, default: 1.0): This setting allows for setting a temperature to the softmax (probability prediction) output and is useful for decreasing sesitivity towards low probability candidates for the next word. This again, improves the search space for the next word.

.. note::

    For a more detailed understanding on using these parameteres for beam search, sampling and their variants of Natural Language Generation techniques, refer to this `blog <https://huggingface.co/blog/how-to-generate>`_

