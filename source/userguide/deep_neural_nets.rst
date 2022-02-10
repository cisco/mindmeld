Deep Neural Networks in MindMeld
================================

The advent of `Deep learning <https://en.wikipedia.org/wiki/Deep_learning>`_ has led to substantial developments in the area of Conversational AI as well as Natural Langauge Processing (NLP) in general.
Specifically, the proliferation of deep neural models based on :wiki_api:`Convolutional neural networks (CNN) <Convolutional_neural_network#Natural_language_processing>`, :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` and :wiki_api:`Transformers <Transformer_(machine_learning_model)>`, and their rapid adaptation to wide range of NLP tasks has been a great success.
Thus extending the suite of traditional machine learning models (e.g. :sk_guide:`Logistic regression <linear_model.html#logistic-regression>`, :sk_guide:`Decision tree <tree.html#tree>`, etc.), MindMeld provides the option for users to train and use a variety of deep neural models with an array of configurable parameters.

The deep neural models are available both for text classification aka. sequence classification (:ref:`domain classification <domain_classification>` and :ref:`intent classification <intent_classification>`) as well as for token classification (:ref:`entity recognition <entity_recognition>`) tasks.
The models are implemented using `Pytorch <https://pytorch.org/>`_ framework and thus need extra installations before starting to use them in your chatbot application.

.. note::

   Please make sure to install the Pytorch requirement by running in the shell: :code:`pip install mindmeld[torch]`.

MindMeld supports the use of pretrained transformer models such as :wiki_api:`BERT <BERT_(language_model)>` through the popular `Huggingface Transformers <https://huggingface.co/docs/transformers/index>`_ library.
Several pretrained models from their `Models Hub <https://huggingface.co/models>`_ that can be used for sequence classification or token classification can be employed in your chatbot application.

.. note::

   Install the extra *transformers* requirement by running in the shell: :code:`pip install mindmeld[transformers]`.

Before proceeding to use the deep neural models, consider the following possible advantages and disadvantages of using them in place of traditional machine learning models.

Some advantages are:

- The deep models generally outperform traditional machine learning models on training sets with at least several hundreds or few thousands of queries when training from scratch, and with at least few hundreds if fine-tuning from a pretrained checkpoint.
- The deep models require little or no feature engineering work than traditional machine learning models, because they learn the input features (such as word embeddings) compared to several hundred engineered features (n-grams, system entities, and so on) of the latter.
- On GPU-enabled devices, the deep networks can achieve training time comparable to some of the traditional models in MindMeld.

Some disadvantages are:

- Training time on CPU-only machines is a lot slower than for traditional machine learning models.
- No automated hyperparameter tuning methods like :sk_api:`sklearn.model_selection.GridSearchCV <sklearn.model_selection.GridSearchCV.html>` are available for deep neural models.
- The deep neural models generally occupy similar or more disk storage space compared to their traditional counterparts.

Parameter tuning for deep neural models is more involved than for traditional machine learning models.
A good starting point for understanding this subject is Andrej Karpathy's `course notes <https://cs231n.github.io/neural-networks-3/#baby>`_ from the Convolutional Neural Networks for Visual Recognition course at Stanford University.

.. note::

   To use deep neural networks instead of traditional machine learning models, simply make few modifications to the classifier configuration dictionaries for all or selected classifiers in your app's ``config.py``.

In the following sections, different model architectures and their configurable parameters are outlined.

Domain and Intent classification
--------------------------------

.. _dnns_sequence_classification:

Recall from :ref:`Working with the Domain Classifier <domain_classifier_configuration>` and :ref:`Working with the Intent Classifier <intent_classifier_configuration>` sections that a text classifier configuration consists of the keys

- ``'features'``,
- ``'param_selection'``,
- ``'model_settings'``, and
- ``'params'``

amongst other keys that do not have distinction between traditional models or deep neural models.
When working with deep neural models, the ``'features'`` and ``'param_selection'`` keys in the classifier configuration are redundant as we neither have to handcraft any feature sets for modeling nor there is an automated hyperparameter tuning.
Thus, the only relevant keys to be configured when using deep neural models are ``'model_settings'`` and ``'params'``.

The ``'model_settings'`` is a :class:`dict` with the single key ``'classifier_type'``, whose value specifies the machine learning model to use.
The allowed values of ``'classifier_type'`` that are backed by deep neural nets and are meant for sequence classification are:

+----------------+----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+
| Value          | Classifier                                                                                                                 | Reference for configurable parameters                                     |
+================+============================================================================================================================+===========================================================================+
| ``'embedder'`` | Pooled :wiki_api:`Token Embeddings <Word_embedding>` or :wiki_api:`Deep Contextualized Embeddings <BERT_(language_model)>` | :ref:`Embedder parameters <dnns_sequence_classification_models_embedder>` |
+----------------+----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+
| ``'cnn'``      | :wiki_api:`Convolutional neural networks (CNN) <Convolutional_neural_network#Natural_language_processing>`                 | :ref:`CNN parameters <dnns_sequence_classification_models_cnn>`           |
+----------------+----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+
| ``'lstm'``     | :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>`                                                | :ref:`LSTM parameters <dnns_sequence_classification_models_lstm>`         |
+----------------+----------------------------------------------------------------------------------------------------------------------------+---------------------------------------------------------------------------+

The ``'params'`` is also a :class:`dict` with several configurable keys, some of which are specific to the choice of classifier type and others common across all the above classifier types.
In the following section, the list of allowed parameters related to each choice of classifier type are outlined.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of configurable params that are not just specific to any classifier type but are common across all the classifier types.

1. ``'embedder'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_sequence_classification_models_embedder:

This classifier type includes neural models that are based on either an embedding lookup table or a deep contextualized embedder, along with a pooling operation on top of those embeddings before passing through a classification layer.
For the former type of embeddings, an embedding lookup table is created depending upon the set of tokens found in training data, with tokens being derived based on a chosen tokenization strategy-- word-level, sub-word-level, or character-level tokenization (see :ref:`Tokenization Choices <choices_for_tokenization>` section below for more details).
The lookup table by default is randomly initialized but can instead be initialized to a pretrained checkpoint (such as `GloVe <https://nlp.stanford.edu/projects/glove/>`_) when using the word-level tokenization strategy.

On the other hand, a deep contextualized embedder is a pretrained embedder such as :wiki_api:`BERT <BERT_(language_model)>`, which consists of its own tokenization strategy and neural embedding process.
In any case, all the underlying weights can be tuned to the training data provided, or can be kept frozen during the training process.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

.. note::

   Specify the embedding choice using the param ``embedder_type``. Set it to ``None``, ``'glove'`` or ``'bert'`` to use with desired embeddings-- based on randomly initialized embedding lookup table, based on lookup table initialized with GloVe pretrained embeddings or a BERT-like transformers architecture based deep contextualized embedder, respectively.

Following are the different optional params that are configurable along with the chosen choice of ``embedder_type`` param.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of additional configurable params that are common across classifiers.

1.1 Based on Embedding Lookup Table (``embedder_type``: ``None``)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``emb_dim``                             | Number of dimensions for each token's embedding.                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``256``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``tokenizer_type``                      | The choice of tokenization strategy to extract tokens from the training data.                                                                                                                                                                    |
|                                         | See `Tokenization Choices <choices_for_tokenization>`_ section below for more details.                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'whitespace-tokenizer'``                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: See `Tokenization Choices <choices_for_tokenization>`_                                                                                                                                                                                  |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``add_terminals``                       | If set to ``True``, terminal tokens (a start and an end token) are added at the beginning and ending for each input before applying any padding. If left unset or                                                                                |
|                                         | set to ``None``, the value will be set to ``True`` if the input text encoders (based on the choice of tokenization) require it to be so.                                                                                                         |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[bool, None]                                                                                                                                                                                                                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``True``, ``False``                                                                                                                                                                                                           |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_pooling_type``        | Specifies the manner in which a query's token-wise embeddings are to be collated into a single embedding before passing through classification layer.                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'mean'``                                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``1.0``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a sequence classifier configuration for classifier based on embedding lookup table:

.. code-block:: python

   {
    'model_type': 'text',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'embedder'},
    'params': {
        'embedder_type': None,
        'emb_dim': 256,
    },
   }


1.2 Based on Pretrained Embedding Lookup Table (``embedder_type``: ``glove``)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``token_dimension``                     | Specifies the dimension of the `GloVe-6B <https://nlp.stanford.edu/projects/glove/>`_ pretrained word vectors. This key is only valid when using ``embedder_type`` as ``'glove'``.                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``300``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``50``, ``100``, ``200``, ``300``                                                                                                                                                                                                       |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``token_pretrained_embedding_filepath`` | Specifies a local file path for pretrained embedding file. This key is only valid when using ``embedder_type`` as ``'glove'``.                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[str, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``None``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: File path to a valid GloVe-style embeddings file                                                                                                                                                                                        |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``add_terminals``                       | If set to ``True``, terminal tokens (a start and an end token) are added at the beginning and ending for each input before applying any padding. If left unset or                                                                                |
|                                         | set to ``None``, the value will be set to ``True`` if the input text encoders (based on the choice of tokenization) require it to be so.                                                                                                         |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[bool, None]                                                                                                                                                                                                                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``True``, ``False``                                                                                                                                                                                                           |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_pooling_type``        | Specifies the manner in which a query's token-wise embeddings are to be collated into a single embedding before passing through classification layer.                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'mean'``                                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``1.0``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a sequence classifier configuration for classifier based on pretrained-initialized embedding lookup table:

.. code-block:: python

   {
    'model_type': 'text',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'embedder'},
    'params': {
        'embedder_type': 'glove',
        'update_embeddings': True,
    },
   }

1.3 Based on Deep Contextualized Embeddings (``embedder_type``: ``bert``)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``pretrained_model_name_or_path``       | Specifies a pretrained checkpoint's name or a valid file path to load a bert-like embedder. This key is only valid when using ``embedder_type`` as ``'bert'``.                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'bert-base-uncased'``                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any valid name from `Huggingface Models Hub <https://huggingface.co/models>`_ or a valid folder path where the model's weights as well as its tokenizer's resources are present.                                                        |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_pooling_type``        | Specifies the manner in which a query's token-wise embeddings are to be collated into a single embedding before passing through classification layer.                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'mean'``                                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``1.0``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``save_frozen_bert_weights``            | If set to ``False``, the weights of the underlying bert-like embedder that are not being tuned are not dumped to disk upon calling a classifier's .dump() method. This boolean key is only valid when ``update_embeddings`` is set to ``False``. |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a sequence classifier configuration for classifier based on BERT embedder:

.. code-block:: python

   {
    'model_type': 'text',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'embedder'},
    'params': {
        'embedder_type': 'bert',
        'pretrained_model_name_or_path': 'distilbert-base-uncased',
        'update_embeddings': True,
    },
   }

2. ``'cnn'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_sequence_classification_models_cnn:

:wiki_api:`Convolutional neural networks (CNN) <Convolutional_neural_network#Natural_language_processing>` based text classifiers are light-weight neural classifiers that have achieved remarkably strong performance on the practically important task of sentence classification.
In its typical architecture for text classification, the first layer embeds the sequence of textual tokens obtained from input text into low-dimensional vectors using an embedding lookup table.
The subsequent layer performs convolutions over the embedded word vectors using kernels (aka. filters); kernels of different lengths capture different patterns from the input text.
For each chosen length, several kernels are used to capture different patterns at the same receptive range leading to several feature maps- one per kernel.
Each feature map is reduced to the maximum value observed in that map and maximum values from all maps are combined to form a long feature vector.
This vector is analogous to a an ``'embedder'`` classifier's pooled output, which is then passed through a classification layer.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

Following are the different optional params that are configurable with the ``'cnn'`` classifier type.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of additional configurable params that are common across classifiers.

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``embedder_type``                       | The choice of embeddings to be used. Specifying ``None`` randomly initializes an embeddings lookup table whereas specifying ``'glove'`` initializes the table with pretrained GloVe embeddings.                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[str, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: None                                                                                                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``'glove'``                                                                                                                                                                                                                   |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``emb_dim``                             | Number of dimensions for each token's embedding. This key is only valid when not using a pretrained embedder.                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``256``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``tokenizer_type``                      | The choice of tokenization strategy to extract tokens from the training data.                                                                                                                                                                    |
|                                         | See `Tokenization Choices <choices_for_tokenization>`_ section below for more details.                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'whitespace-tokenizer'``                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: See `Tokenization Choices <choices_for_tokenization>`_                                                                                                                                                                                  |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``add_terminals``                       | If set to ``True``, terminal tokens (a start and an end token) are added at the beginning and ending for each input before applying any padding. If left unset or                                                                                |
|                                         | set to ``None``, the value will be set to ``True`` if the input text encoders (based on the choice of tokenization) require it to be so.                                                                                                         |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[bool, None]                                                                                                                                                                                                                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``True``, ``False``                                                                                                                                                                                                           |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``window_sizes``                        | The lengths of 1D CNN kernels to be used for convolution on top of embeddings.                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: List[int]                                                                                                                                                                                                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``[3,4,5]``                                                                                                                                                                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A list of positive integers                                                                                                                                                                                                             |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``number_of_windows``                   | The number of kernels per each specified length of 1D CNN kernels.                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: List[int]                                                                                                                                                                                                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ```[100,100,100]``                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A list of positive integers; same length as ``window_sizes``                                                                                                                                                                            |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a sequence classifier configuration for classifier based on CNNs:

.. code-block:: python

   {
    'model_type': 'text',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'cnn'},
    'params': {
        'embedder_type': 'glove',
        'window_sizes': [3,4,5],
        'number_of_windows': [100,100,100],
    },
   }

3. ``'lstm'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_sequence_classification_models_lstm:

:wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` based text classifiers are classifiers that utilize recurrent feedback connections to be able to learn temporal dependencies in sequential data.
In its typical architecture for text classification, the first layer embeds the sequence of textual tokens obtained from input text into low-dimensional vectors using an embedding lookup table.
The capacity of an LSTM in maintaining the temporal information is generally dependant on its *hidden* dimension.
Further, several LSTM layers can be stacked one-after-another and each layer can process the text from just beginning-to-end or both ways.
The first layer's output embedding sequence is passed through the stacked LSTMs, which finally produces one vector per token of the input text.
To obtain a single vector per input text, the vectors for each token can be pooled or the last vector in the sequence can simply be used as representative vector.
This vector is analogous to a an ``'embedder'`` classifier's pooled output, which is then passed through a classification layer.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

Following are the different optional params that are configurable with the ``'lstm'`` classifier type.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of additional configurable params that are common across classifiers.

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``embedder_type``                       | The choice of embeddings to be used. Specifying ``None`` randomly initializes an embeddings lookup table whereas specifying ``'glove'`` initializes the table with pretrained GloVe embeddings.                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[str, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: None                                                                                                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``'glove'``                                                                                                                                                                                                                   |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``emb_dim``                             | Number of dimensions for each token's embedding. This key is only valid when not using a pretrained embedder.                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``256``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``tokenizer_type``                      | The choice of tokenization strategy to extract tokens from the training data.                                                                                                                                                                    |
|                                         | See `Tokenization Choices <choices_for_tokenization>`_ section below for more details.                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'whitespace-tokenizer'``                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: See `Tokenization Choices <choices_for_tokenization>`_                                                                                                                                                                                  |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``add_terminals``                       | If set to ``True``, terminal tokens (a start and an end token) are added at the beginning and ending for each input before applying any padding. If left unset or                                                                                |
|                                         | set to ``None``, the value will be set to ``True`` if the input text encoders (based on the choice of tokenization) require it to be so.                                                                                                         |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[bool, None]                                                                                                                                                                                                                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``True``, ``False``                                                                                                                                                                                                           |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_hidden_dim``                     | Number of states per each LSTM layer.                                                                                                                                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``128``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_num_layers``                     | The number of LSTM layers that are to be stacked sequentially.                                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``2``                                                                                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_keep_prob``                      | Keep probability for the nodes that constitute the outputs of each LSTM layer except the last LSTM layer.                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_bidirectional``                  | If ``True``, the LSTM layers will be bidirectional.                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_output_pooling_type``            | Specifies the manner in which a query's token-wise embeddings are to be collated into a single embedding before passing through classification layer.                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'last'``                                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a sequence classifier configuration for classifier based on LSTMs:

.. code-block:: python

   {
    'model_type': 'text',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'cnn'},
    'params': {
        'embedder_type': 'glove',
        'lstm_hidden_dim': 128,
        'lstm_bidirectional': True,
    },
   }

Entity recognition
------------------

.. _dnns_token_classification:

Recall from :ref:`Working with the Entity Recognizer <entity_recognizer_configuration>` section that a recognizer's configuration consists of the keys

- ``'features'``,
- ``'param_selection'``,
- ``'model_settings'``, and
- ``'params'``

amongst other keys that do not have distinction between traditional models or deep neural models.
When working with deep neural models, the ``'features'`` and ``'param_selection'`` keys in the classifier configuration are redundant as we neither have to handcraft any feature sets for modeling nor there is an automated hyperparameter tuning.
Thus, the only relevant keys to be configured when using deep neural models are ``'model_settings'`` and ``'params'``.

The ``'model_settings'`` is a :class:`dict` with the single key ``'classifier_type'``, whose value specifies the machine learning model to use.
The allowed values of ``'classifier_type'`` that are backed by deep neural nets and are meant for token classification are:

+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| Value               | Classifier                                                                                                                                                                                                                                            | Reference for configurable parameters                                            |
+=====================+=======================================================================================================================================================================================================================================================+==================================================================================+
| ``'embedder'``      | Pooled :wiki_api:`Token Embeddings <Word_embedding>` or :wiki_api:`Deep Contextualized Embeddings <BERT_(language_model)>`                                                                                                                            | :ref:`Embedder parameters <dnns_tokens_classification_models_embedder>`          |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| ``'lstm-pytorch'``  | :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>`                                                                                                                                                                           | :ref:`LSTM-PYTORCH parameters <dnns_token_classification_models_lstm_pytorch>`   |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| ``'cnn-lstm'``      | Character-level :wiki_api:`Convolutional neural networks (CNN) <Convolutional_neural_network#Natural_language_processing>` followed by word-level :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>`                         | :ref:`CNN-LSTM parameters <dnns_token_classification_models_cnn_lstm>`           |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| ``'lstm-lstm'``     | Character-level :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` followed by word-level :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>`                                                        | :ref:`LSTM-LSTM parameters <dnns_token_classification_models_lstm_lstm>`         |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+
| ``'lstm'``          | :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` coupled with gazetteer encodings and backed by `Tensorflow <https://www.tensorflow.org/>`_                                                                                | :ref:`LSTM parameters <dnns_token_classification_models_lstm_tensorflow>`        |
+---------------------+-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+----------------------------------------------------------------------------------+

The ``'params'`` is also a :class:`dict` with several configurable keys, some of which are specific to the choice of classifier type and others common across all the above classifier types.
In the following section, the list of allowed parameters related to each choice of classifier type are outlined.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of configurable params that are not just specific to any classifier type but are common across all the classifier types.

1. ``'embedder'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_tokens_classification_models_embedder:

This classifier type includes neural models that are based on either an embedding lookup table or a deep contextualized embedder, the outputs of which are then passed through a `Conditional Random Field (CRF) <https://en.wikipedia.org/wiki/Conditional_random_field>`_ or a `Softmax layer <https://en.wikipedia.org/wiki/Softmax_function>`_  which labels the target word as a particular entity.
For the former type of embeddings, an embedding lookup table is created depending upon the set of tokens found in training data, with tokens being derived based on a chosen tokenization strategy-- word-level, sub-word-level, or character-level tokenization (see :ref:`Tokenization Choices <choices_for_tokenization>` section below for more details).
The lookup table by default is randomly initialized but can instead be initialized to a pretrained checkpoint (such as `GloVe <https://nlp.stanford.edu/projects/glove/>`_) when using the word-level tokenization strategy.

On the other hand, a deep contextualized embedder is a pretrained embedder such as :wiki_api:`BERT <BERT_(language_model)>`, which consists of its own tokenization strategy and neural embedding process.
In any case, all the underlying weights can be tuned to the training data provided, or can be kept frozen during the training process.

To obtain a single vector per word per input text, the vectors of all tokens corresponding to each word (for which an entity tag is to be ascertained) are pooled.
This is unlike sequence classification models where all tokens of all words are pooled together.
The pooled outputs are then passed through a classification layer.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

.. note::

   Specify the embedding choice using the param ``embedder_type``. Set it to ``None``, ``'glove'`` or ``'bert'`` to use with desired embeddings-- based on randomly initialized embedding lookup table, based on lookup table initialized with GloVe pretrained embeddings or a BERT-like transformers architecture based deep contextualized embedder, respectively.

Following are the different optional params that are configurable along with the chosen choice of ``embedder_type`` param.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of additional configurable params that are common across classifiers.

1.1 Based on Embedding Lookup Table (``embedder_type``: ``None``)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``emb_dim``                             | Number of dimensions for each token's embedding.                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``256``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``tokenizer_type``                      | The choice of tokenization strategy to extract tokens from the training data.                                                                                                                                                                    |
|                                         | See `Tokenization Choices <choices_for_tokenization>`_ section below for more details.                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'whitespace-tokenizer'``                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: See `Tokenization Choices <choices_for_tokenization>`_                                                                                                                                                                                  |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``add_terminals``                       | If set to ``True``, terminal tokens (a start and an end token) are added at the beginning and ending for each input before applying any padding. If left unset or                                                                                |
|                                         | set to ``None``, the value will be set to ``True`` if the input text encoders (based on the choice of tokenization) require it to be so.                                                                                                         |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[bool, None]                                                                                                                                                                                                                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``None``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``True``, ``False``                                                                                                                                                                                                           |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``1.0``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``token_spans_pooling_type``            | Specifies the manner in which a word's token-wise embeddings are to be collated into a single embedding before passing through entity classification layer.                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'first'``                                                                                                                                                                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``use_crf_layer``                       | If set to ``True``, a CRF layer is used for entity classification instead of a softmax layer.                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a token classifier configuration for classifier based on embedding lookup table:

.. code-block:: python

   {
    'model_type': 'tagger',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'embedder'},
    'params': {
        'embedder_type': None,
        'emb_dim': 256,
    },
   }

1.2 Based on Pretrained Embedding Lookup Table (``embedder_type``: ``glove``)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``token_dimension``                     | Specifies the dimension of the `GloVe-6B <https://nlp.stanford.edu/projects/glove/>`_ pretrained word vectors. This key is only valid when using ``embedder_type`` as ``'glove'``.                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``300``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``50``, ``100``, ``200``, ``300``                                                                                                                                                                                                       |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``token_pretrained_embedding_filepath`` | Specifies a local file path for pretrained embedding file. This key is only valid when using ``embedder_type`` as ``'glove'``.                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[str, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``None``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: File path to a valid GloVe-style embeddings file                                                                                                                                                                                        |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``add_terminals``                       | If set to ``True``, terminal tokens (a start and an end token) are added at the beginning and ending for each input before applying any padding. If left unset or                                                                                |
|                                         | set to ``None``, the value will be set to ``True`` if the input text encoders (based on the choice of tokenization) require it to be so.                                                                                                         |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[bool, None]                                                                                                                                                                                                                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``None``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``True``, ``False``                                                                                                                                                                                                           |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``1.0``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``token_spans_pooling_type``            | Specifies the manner in which a word's token-wise embeddings are to be collated into a single embedding before passing through entity classification layer.                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'first'``                                                                                                                                                                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``use_crf_layer``                       | If set to ``True``, a CRF layer is used for entity classification instead of a softmax layer.                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a token classifier configuration for classifier based on pretrained-initialized embedding lookup table:

.. code-block:: python

   {
    'model_type': 'tagger',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'embedder'},
    'params': {
        'embedder_type': 'glove',
        'update_embeddings': True,
    },
   }

1.3 Based on Deep Contextualized Embeddings (``embedder_type``: ``bert``)
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``pretrained_model_name_or_path``       | Specifies a pretrained checkpoint's name or a valid file path to load a bert-like embedder. This key is only valid when using ``embedder_type`` as ``'bert'``.                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'bert-base-uncased'``                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any valid name from `Huggingface Models Hub <https://huggingface.co/models>`_ or a valid folder path where the model's weights as well as its tokenizer's resources are present.                                                        |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``1.0``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``save_frozen_bert_weights``            | If set to ``False``, the weights of the underlying bert-like embedder that are not being tuned are not dumped to disk upon calling a classifier's .dump() method. This boolean key is only valid when ``update_embeddings`` is set to ``False``. |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``token_spans_pooling_type``            | Specifies the manner in which a word's token-wise embeddings are to be collated into a single embedding before passing through entity classification layer.                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'first'``                                                                                                                                                                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``use_crf_layer``                       | If set to ``True``, a CRF layer is used for entity classification instead of a softmax layer.                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a token classifier configuration for classifier based on BERT embedder:

.. code-block:: python

   {
    'model_type': 'tagger',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'embedder'},
    'params': {
        'embedder_type': 'bert',
        'pretrained_model_name_or_path': 'distilbert-base-uncased',
        'update_embeddings': True,
    },
   }

2. ``'lstm-pytorch'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_token_classification_models_lstm_pytorch:

:wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` based text classifiers are classifiers that utilize recurrent feedback connections to be able to learn temporal dependencies in sequential data.
In its typical architecture for entity recognition, the first layer embeds the sequence of textual tokens obtained from input text into low-dimensional vectors using an embedding lookup table.
The capacity of an LSTM in maintaining the temporal information is generally dependant on its *hidden* dimension.
Further, several LSTM layers can be stacked one-after-another and each layer can process the text from just beginning-to-end or both ways.
The first layer's output embedding sequence is passed through the stacked LSTMs, which finally produces one vector per token of the input text.

To obtain a single vector per word per input text, the vectors of all tokens corresponding to each word (for which an entity tag is to be ascertained) are pooled.
This is unlike sequence classification models where all tokens of all words are pooled together.
The pooled outputs are then passed through a classification layer.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

Following are the different optional params that are configurable with the ``'lstm'`` classifier type.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of additional configurable params that are common across classifiers.

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``embedder_type``                       | The choice of embeddings to be used. Specifying ``None`` randomly initializes an embeddings lookup table whereas specifying ``'glove'`` initializes the table with pretrained GloVe embeddings.                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[str, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: None                                                                                                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``'glove'``                                                                                                                                                                                                                   |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``emb_dim``                             | Number of dimensions for each token's embedding. This key is only valid when not using a pretrained embedder.                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``256``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``tokenizer_type``                      | The choice of tokenization strategy to extract tokens from the training data.                                                                                                                                                                    |
|                                         | See `Tokenization Choices <choices_for_tokenization>`_ section below for more details.                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'whitespace-tokenizer'``                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: See `Tokenization Choices <choices_for_tokenization>`_                                                                                                                                                                                  |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``add_terminals``                       | If set to ``True``, terminal tokens (a start and an end token) are added at the beginning and ending for each input before applying any padding. If left unset or                                                                                |
|                                         | set to ``None``, the value will be set to ``True`` if the input text encoders (based on the choice of tokenization) require it to be so.                                                                                                         |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[bool, None]                                                                                                                                                                                                                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``None``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``True``, ``False``                                                                                                                                                                                                           |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_hidden_dim``                     | Number of states per each LSTM layer.                                                                                                                                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``128``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_num_layers``                     | The number of LSTM layers that are to be stacked sequentially.                                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``2``                                                                                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_keep_prob``                      | Keep probability for the nodes that constitute the outputs of each LSTM layer except the last LSTM layer.                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_bidirectional``                  | If ``True``, the LSTM layers will be bidirectional.                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``token_spans_pooling_type``            | Specifies the manner in which a word's token-wise embeddings are to be collated into a single embedding before passing through entity classification layer.                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'first'``                                                                                                                                                                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``use_crf_layer``                       | If set to ``True``, a CRF layer is used for entity classification instead of a softmax layer.                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a token classifier configuration for classifier based on LSTMs:

.. code-block:: python

   {
    'model_type': 'tagger',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'cnn'},
    'params': {
        'embedder_type': 'glove',
        'lstm_hidden_dim': 128,
        'lstm_bidirectional': True,
    },
   }

3. ``'cnn-lstm'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_token_classification_models_cnn_lstm:

:wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` based text classifiers are classifiers that utilize recurrent feedback connections to be able to learn temporal dependencies in sequential data.
When coupled with :wiki_api:`Convolutional neural networks (CNN) <Convolutional_neural_network#Natural_language_processing>` for extracting character-level features from input text, the overall architecture can better model the textual data as well as it is more robust to variations in the spellings.
In its typical architecture for entity recognition, the first layer embeds the sequence of words obtained from input text into low-dimensional vectors using an embedding lookup table,
concatenated with outputs of convolutions at character-level (for each word individually) using kernels of different lengths to capture different patterns.
The capacity of an LSTM in maintaining the temporal information is generally dependant on its *hidden* dimension.
Further, several LSTM layers can be stacked one-after-another and each layer can process the text from just beginning-to-end or both ways.
The first layer's output embedding sequence is passed through the stacked LSTMs, which finally produces one vector per token of the input text.

To obtain a single vector per word per input text, the vectors of all tokens corresponding to each word (for which an entity tag is to be ascertained) are pooled.
This is unlike sequence classification models where all tokens of all words are pooled together.
The pooled outputs are then passed through a classification layer.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

Following are the different optional params that are configurable with the ``'lstm'`` classifier type.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of additional configurable params that are common across classifiers.

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``embedder_type``                       | The choice of embeddings to be used. Specifying ``None`` randomly initializes an embeddings lookup table whereas specifying ``'glove'`` initializes the table with pretrained GloVe embeddings.                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[str, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: None                                                                                                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``'glove'``                                                                                                                                                                                                                   |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``emb_dim``                             | Number of dimensions for each token's embedding. This key is only valid when not using a pretrained embedder.                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``256``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_hidden_dim``                     | Number of states per each LSTM layer.                                                                                                                                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``128``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_num_layers``                     | The number of LSTM layers that are to be stacked sequentially.                                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``2``                                                                                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_keep_prob``                      | Keep probability for the nodes that constitute the outputs of each LSTM layer except the last LSTM layer.                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_bidirectional``                  | If ``True``, the LSTM layers will be bidirectional.                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_emb_dim``                        | Number of dimensions for each character's embedding.                                                                                                                                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``50``                                                                                                                                                                                                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_window_sizes``                   | The lengths of 1D CNN kernels to be used for character-level convolution on top of character embeddings.                                                                                                                                         |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: List[int]                                                                                                                                                                                                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``[3,4,5]``                                                                                                                                                                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A list of positive integers                                                                                                                                                                                                             |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_number_of_windows``              | The number of kernels per each specified length of 1D CNN kernels in ``char_window_sizes``.                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: List[int]                                                                                                                                                                                                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ```[100,100,100]``                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A list of positive integers; same length as ``char_window_sizes``                                                                                                                                                                       |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_cnn_output_keep_prob``           | Keep probability for the dropout layer placed on top of character CNN's output. Dropout helps in regularization and reduces over-fitting.                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_proj_dim``                       | The final dimension of each character after it is transformed by the character-level network.                                                                                                                                                    |
|                                         | Usually greater than the ``char_emb_dim`` since it encodes more information about orthography and semantics.                                                                                                                                     |
|                                         | If unspecified or ``None``, the dimension is same as the ``char_emb_dim``.                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[int, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``None``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer, ``None``                                                                                                                                                                                                          |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_padding_length``                 | The maximum number of characters allowed per word.                                                                                                                                                                                               |
|                                         | If a word has more characters than ``char_padding_length``, the surplus characters are discarded.                                                                                                                                                |
|                                         | If specified as ``None``, the ``char_padding_length`` in a mini-batch of queries is simply the maximum length of                                                                                                                                 |
|                                         | all words in that mini-batch.                                                                                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[int, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``None``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer, ``None``                                                                                                                                                                                                          |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_add_terminals``                  | If set to ``True``, terminal character tokens (a start and an end character token) are added at the beginning and ending for each                                                                                                                |
|                                         | word before applying any padding, while preparing inputs to the underlying character-level network.                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``use_crf_layer``                       | If set to ``True``, a CRF layer is used for entity classification instead of a softmax layer.                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a token classifier configuration for classifier based on CNN-LSTM:

.. code-block:: python

   {
    'model_type': 'tagger',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'cnn-lstm'},
    'params': {
        'embedder_type': 'glove',
        'lstm_hidden_dim': 128,
        'lstm_bidirectional': True,
        'char_emb_dim': 32
    },
   }

4. ``'lstm-lstm'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_token_classification_models_lstm_lstm:

:wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` based text classifiers are classifiers that utilize recurrent feedback connections to be able to learn temporal dependencies in sequential data.
When coupled with :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` for extracting character-level features from input text, the overall architecture can better model the textual data as well as it is more robust to variations in the spellings.
In its typical architecture for entity recognition, the first layer embeds the sequence of words obtained from input text into low-dimensional vectors using an embedding lookup table,
concatenated with outputs of character-level LSTM (for each word individually) to capture different patterns.
The capacity of an LSTM in maintaining the temporal information is generally dependant on its *hidden* dimension.
Further, several LSTM layers can be stacked one-after-another and each layer can process the text from just beginning-to-end or both ways.
The first layer's output embedding sequence is passed through the stacked LSTMs, which finally produces one vector per token of the input text.

To obtain a single vector per word per input text, the vectors of all tokens corresponding to each word (for which an entity tag is to be ascertained) are pooled.
This is unlike sequence classification models where all tokens of all words are pooled together.
The pooled outputs are then passed through a classification layer.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

Following are the different optional params that are configurable with the ``'lstm'`` classifier type.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of additional configurable params that are common across classifiers.

+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Configuration Key                       | Description                                                                                                                                                                                                                                      |
+=========================================+==================================================================================================================================================================================================================================================+
| ``embedder_type``                       | The choice of embeddings to be used. Specifying ``None`` randomly initializes an embeddings lookup table whereas specifying ``'glove'`` initializes the table with pretrained GloVe embeddings.                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[str, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: None                                                                                                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``None``, ``'glove'``                                                                                                                                                                                                                   |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``emb_dim``                             | Number of dimensions for each token's embedding. This key is only valid when not using a pretrained embedder.                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``256``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``update_embeddings``                   | If set to ``False``, the weights of embedding table or the deep contextualized embedder will not be updated during back-propogation of gradients. This boolean key is only valid when using a pretrained embedder type.                          |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``embedder_output_keep_prob``           | Keep probability for the dropout layer placed on top of embeddings. Dropout helps in regularization and reduces over-fitting.                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``output_keep_prob``                    | Keep probability for the dropout layer placed on top of classifier's penultimate layer (i.e the layer before logits are computed). Dropout helps in regularization and reduces over-fitting.                                                     |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_hidden_dim``                     | Number of states per each LSTM layer.                                                                                                                                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``128``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_num_layers``                     | The number of LSTM layers that are to be stacked sequentially.                                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``2``                                                                                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_keep_prob``                      | Keep probability for the nodes that constitute the outputs of each LSTM layer except the last LSTM layer.                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``lstm_bidirectional``                  | If ``True``, the LSTM layers will be bidirectional.                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_emb_dim``                        | Number of dimensions for each character's embedding.                                                                                                                                                                                             |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``50``                                                                                                                                                                                                                                  |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_lstm_hidden_dim``                | Number of states per each character-level LSTM layer.                                                                                                                                                                                            |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``128``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_lstm_num_layers``                | The number of character-level LSTM layers that are to be stacked sequentially.                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: int                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``2``                                                                                                                                                                                                                                   |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer                                                                                                                                                                                                                    |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_lstm_keep_prob``                 | Keep probability for the nodes that constitute the outputs of each character-level LSTM layer except the last layer in the stack.                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: float                                                                                                                                                                                                                                      |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``0.7``                                                                                                                                                                                                                                 |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: A float between 0 and 1                                                                                                                                                                                                                 |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_lstm_bidirectional``             | If ``True``, the character-level LSTM layers will be bidirectional.                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``True``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_lstm_output_pooling_type``       | Specifies the manner in which a word's character-level embeddings are to be collated into a single embedding before passing to subsequent layers.                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: str                                                                                                                                                                                                                                        |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``'last'``                                                                                                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``'first'``, ``'last'``, ``'max'``, ``'mean'``, ``'mean_sqrt'``                                                                                                                                                                         |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_proj_dim``                       | The final dimension of each character after it is transformed by the character-level network.                                                                                                                                                    |
|                                         | Usually greater than the ``char_emb_dim`` since it encodes more information about orthography and semantics.                                                                                                                                     |
|                                         | If unspecified or ``None``, the dimension is same as the ``char_emb_dim``.                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[int, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``None``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer, ``None``                                                                                                                                                                                                          |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_padding_length``                 | The maximum number of characters allowed per word.                                                                                                                                                                                               |
|                                         | If a word has more characters than ``char_padding_length``, the surplus characters are discarded.                                                                                                                                                |
|                                         | If specified as ``None``, the ``char_padding_length`` in a mini-batch of queries is simply the maximum length of                                                                                                                                 |
|                                         | all words in that mini-batch.                                                                                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: Union[int, None]                                                                                                                                                                                                                           |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``None``                                                                                                                                                                                                                                |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: Any positive integer, ``None``                                                                                                                                                                                                          |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``char_add_terminals``                  | If set to ``True``, terminal character tokens (a start and an end character token) are added at the beginning and ending for each                                                                                                                |
|                                         | word before applying any padding, while preparing inputs to the underlying character-level network.                                                                                                                                              |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| ``use_crf_layer``                       | If set to ``True``, a CRF layer is used for entity classification instead of a softmax layer.                                                                                                                                                    |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimalistic example of a token classifier configuration for classifier based on LSTM-LSTM:

.. code-block:: python

   {
    'model_type': 'tagger',
    'train_label_set': 'train.*\.txt',
    'test_label_set': 'test.*\.txt',
    'model_settings': {'classifier_type': 'lstm-lstm'},
    'params': {
        'embedder_type': 'glove',
        'lstm_hidden_dim': 128,
        'lstm_bidirectional': True,
        'char_emb_dim': 32,
    },
   }

5. ``'lstm'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_token_classification_models_lstm_tensorflow:

A `Tensorflow <https://www.tensorflow.org/>`_ backed implementation of `Bi-Directional Long Short-Term Memory (LSTM) Network <https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks>`_.

.. note::

   To use this classifier type, please make sure to install the Tensorflow requirement by running in the shell: :code:`pip install mindmeld[tensorflow]`.

The MindMeld Bi-Directional LSTM network

 - encodes words as pre-trained word embeddings using Stanford's `GloVe representation <https://nlp.stanford.edu/projects/glove/>`_
 - encodes characters using a convolutional network trained on the training data
 - concatenates the word and character embeddings together and feeds them into the bi-directional LSTM
 - couples the forget and input gates of the LSTM using a peephole connection, to improve overall accuracies on downstream NLP tasks
 - feeds the output of the LSTM into a `linear chain Conditional Random Field <https://en.wikipedia.org/wiki/Conditional_random_field>`_ (CRF) or `Softmax layer <https://en.wikipedia.org/wiki/Softmax_function>`_  which labels the target word as a particular entity

Following are the different optional params that are configurable with the ``'lstm'`` classifier type.
Unlike other classifier types, this classifier type **does not** share any common additional configurable params.

+-----------------------------------------+------------------------------------------------------------------------------------------------+
| Parameter name                          | Description                                                                                    |
+=========================================+================================================================================================+
| ``padding_length``                      | The sequence model treats this as the maximum number of words in a query.                      |
|                                         | If a query has more words than ``padding_length``, the surplus words are discarded.            |
|                                         |                                                                                                |
|                                         | Typically set to the maximum word length of query expected both at train and predict time.     |
|                                         |                                                                                                |
|                                         | Default: ``20``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'padding_length': 20}``                                                                     |
|                                         |  - a query can have a maximum of twenty words                                                  |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``batch_size``                          | Size of each batch of training data to feed into the network (which uses mini-batch learning). |
|                                         |                                                                                                |
|                                         | Default: ``20``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'batch_size': 20}``                                                                         |
|                                         |  - feed twenty training queries to the network for each learning step                          |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``display_epoch``                       | The network displays training accuracy statistics at this interval, measured in epochs.        |
|                                         |                                                                                                |
|                                         | Default: ``5``                                                                                 |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'display_epoch': 5}``                                                                       |
|                                         |  - display accuracy statistics every five epochs                                               |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``number_of_epochs``                    | Total number of complete iterations of the training data to feed into the network.             |
|                                         | In each iteration, the data is shuffled to break any prior sequence patterns.                  |
|                                         |                                                                                                |
|                                         | Default: ``20``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'number_of_epochs': 20}``                                                                   |
|                                         |  - iterate through the training data twenty times                                              |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``optimizer``                           | Optimizer to use to minimize the network's stochastic objective function.                      |
|                                         |                                                                                                |
|                                         | Default: ``'adam'``                                                                            |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'optimizer': 'adam'}``                                                                      |
|                                         |  - use the Adam optimizer to minimize the objective function                                   |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``learning_rate``                       | Parameter to control the size of weight and bias changes                                       |
|                                         | of the training algorithm as it learns.                                                        |
|                                         |                                                                                                |
|                                         | `This <https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Error-Correction_Learning>`_   |
|                                         | article explains Learning Rate in technical terms.                                             |
|                                         |                                                                                                |
|                                         | Default: ``0.005``                                                                             |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'learning_rate': 0.005}``                                                                   |
|                                         |  - set learning rate to 0.005                                                                  |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``dense_keep_prob``                     | In the context of the ''dropout'' technique (a regularization method to prevent overfitting),  |
|                                         | keep probability specifies the proportion of nodes to "keep"—that is, to exempt from dropout   |
|                                         | during the network's learning phase.                                                           |
|                                         |                                                                                                |
|                                         | The ``dense_keep_prob`` parameter sets the keep probability of the nodes                       |
|                                         | in the dense network layer that connects the output of the LSTM layer                          |
|                                         | to the nodes that predict the named entities.                                                  |
|                                         |                                                                                                |
|                                         | Default: ``0.5``                                                                               |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'dense_keep_prob': 0.5}``                                                                   |
|                                         |  - 50% of the nodes in the dense layer will not be turned off by dropout                       |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``lstm_input_keep_prob``                | Keep probability for the nodes that constitute the inputs to the LSTM cell.                    |
|                                         |                                                                                                |
|                                         | Default: ``0.5``                                                                               |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'lstm_input_keep_prob': 0.5}``                                                              |
|                                         |  - 50% of the nodes that are inputs to the LSTM cell will not be turned off by dropout         |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``lstm_output_keep_prob``               | Keep probability for the nodes that constitute the outputs of the LSTM cell.                   |
|                                         |                                                                                                |
|                                         | Default: ``0.5``                                                                               |
|                                         |                                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'lstm_output_keep_prob': 0.5}``                                                             |
|                                         |  - 50% of the nodes that are outputs of the LSTM cell will not be turned off by dropout        |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``token_lstm_hidden_state_dimension``   | Number of states per LSTM cell.                                                                |
|                                         |                                                                                                |
|                                         | Default: ``300``                                                                               |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'token_lstm_hidden_state_dimension': 300}``                                                 |
|                                         |  - an LSTM cell will have 300 states                                                           |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``token_embedding_dimension``           | Number of dimensions for word embeddings.                                                      |
|                                         |                                                                                                |
|                                         | Allowed values: [50, 100, 200, 300].                                                           |
|                                         |                                                                                                |
|                                         | Default: ``300``                                                                               |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'token_embedding_dimension': 300}``                                                         |
|                                         |  - each word embedding will have 300 dimensions                                                |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``gaz_encoding_dimension``              | Number of nodes to connect to the gazetteer encodings in a fully-connected network.            |
|                                         |                                                                                                |
|                                         | Default: ``100``                                                                               |
|                                         |                                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'gaz_encoding_dimension': 100}``                                                            |
|                                         |  - 100 nodes will be connected to the gazetteer encodings in a fully-connected network         |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``max_char_per_word``                   | The sequence model treats this as the maximum number of characters in a word.                  |
|                                         | If a word has more characters than ``max_char_per_word``, the surplus characters are discarded.|
|                                         |                                                                                                |
|                                         | Usually set to the size of the longest word in the training and test sets.                     |
|                                         |                                                                                                |
|                                         | Default: ``20``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'max_char_per_word': 20}``                                                                  |
|                                         |  - a word can have a maximum of twenty characters                                              |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``use_crf_layer``                       | If set to ``True``, use a linear chain Conditional Random Field layer for the final layer,     |
|                                         | which predicts sequence tags.                                                                  |
|                                         |                                                                                                |
|                                         | If set to ``False``, use a softmax layer to predict sequence tags.                             |
|                                         |                                                                                                |
|                                         | Default: ``False``                                                                             |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'use_crf_layer': True}``                                                                    |
|                                         |  - use the CRF layer                                                                           |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``use_character_embeddings``            | If set to ``True``, use the character embedding trained on the training data                   |
|                                         | using a convolutional network.                                                                 |
|                                         |                                                                                                |
|                                         | If set to ``False``, do not use character embeddings.                                          |
|                                         |                                                                                                |
|                                         | Note: Using character embedding significantly increases training time                          |
|                                         | compared to vanilla word embeddings only.                                                      |
|                                         |                                                                                                |
|                                         | Default: ``False``                                                                             |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'use_character_embeddings': True}``                                                         |
|                                         |  - use character embeddings                                                                    |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``char_window_sizes``                   | List of window sizes for convolutions that the network should use                              |
|                                         | to build the character embeddings.                                                             |
|                                         | Usually in decreasing numerical order.                                                         |
|                                         |                                                                                                |
|                                         | Note: This parameter is needed only if ``use_character_embeddings`` is set to ``True``.        |
|                                         |                                                                                                |
|                                         | Default: ``[5]``                                                                               |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'char_window_sizes': [5, 3]}``                                                              |
|                                         |  - first, use a convolution of size 5                                                          |
|                                         |  - next, feed the output of that convolution through a convolution of size 3                   |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``character_embedding_dimension``       | Initial dimension of each character before it is fed into the convolutional network.           |
|                                         |                                                                                                |
|                                         | Note: This parameter is needed only if ``use_character_embeddings`` is set to ``True``.        |
|                                         |                                                                                                |
|                                         | Default: ``10``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'character_embedding_dimension': 10}``                                                      |
|                                         |  - initialize the dimension of each character to ten                                           |
+-----------------------------------------+------------------------------------------------------------------------------------------------+
| ``word_level_character_embedding_size`` | The final dimension of each character after it is transformed                                  |
|                                         | by the convolutional network.                                                                  |
|                                         |                                                                                                |
|                                         | Usually greater than ``character_embedding_dimension`` since it encodes                        |
|                                         | more information about orthography and semantics.                                              |
|                                         |                                                                                                |
|                                         | Note: This parameter is needed only if ``use_character_embeddings`` is set to ``True``.        |
|                                         |                                                                                                |
|                                         | Default: ``40``                                                                                |
|                                         |                                                                                                |
|                                         | Example:                                                                                       |
|                                         |                                                                                                |
|                                         | ``{'word_level_character_embedding_size': 40}``                                                |
|                                         |  - each character should have dimension of forty, after convolutional network training         |
+-----------------------------------------+------------------------------------------------------------------------------------------------+


Addendum
--------

Common Configurable Params
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _common_configurable_params:

The following are some params that are commonly configurable across all the classifier types described above, both for domain/intent classification as well as for entity recognition.

+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| Parameter name                  | Description                                                                                                                     |
+=================================+=================================================================================================================================+
| ``device``                      | Name of the device on which torch tensors will be allocated. The ``'cuda'`` choice is to be specified only when                 |
|                                 | building models in a `GPU <https://en.wikipedia.org/wiki/Graphics_processing_unit>`_ environment.                               |
|                                 |                                                                                                                                 |
|                                 | Type: str                                                                                                                       |
|                                 |                                                                                                                                 |
|                                 | Default: ``'cuda'`` if torch.cuda.is_available() else ``'cpu'``                                                                 |
|                                 |                                                                                                                                 |
|                                 | Choices: ``'cuda'``, ``'cpu'``                                                                                                  |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| ``number_of_epochs``            | The total number of complete iterations of the training data to feed into the network. In each iteration, the                   |
|                                 | data is shuffled to break any prior sequence patterns.                                                                          |
|                                 |                                                                                                                                 |
|                                 | Type: int                                                                                                                       |
|                                 |                                                                                                                                 |
|                                 | Default: ``100`` unless specified otherwise for the selected classifier type.                                                   |
|                                 |                                                                                                                                 |
|                                 | Choices: Any positive integer                                                                                                   |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| ``patience``                    | The number of epochs to wait without any improvement on the validation metric before terminating training.                      |
|                                 |                                                                                                                                 |
|                                 | Type: int                                                                                                                       |
|                                 |                                                                                                                                 |
|                                 | Default: ``10`` if token classification else ``7``, unless specified otherwise for the selected classifier type.                |
|                                 |                                                                                                                                 |
|                                 | Choices: Any positive integer                                                                                                   |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| ``batch_size``                  | Size of each batch of training data to feed into the network (which uses mini-batch learning).                                  |
|                                 |                                                                                                                                 |
|                                 | Type: int                                                                                                                       |
|                                 |                                                                                                                                 |
|                                 | Default: ``32`` unless specified otherwise for the selected classifier type.                                                    |
|                                 |                                                                                                                                 |
|                                 | Choices: Any positive integer                                                                                                   |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| ``gradient_accumulation_steps`` | Number of consecutive mini-batches for which gradients will be averaged and accumulated before updating the                     |
|                                 | weights of the network.                                                                                                         |
|                                 |                                                                                                                                 |
|                                 | Type: int                                                                                                                       |
|                                 |                                                                                                                                 |
|                                 | Default: ``1`` unless specified otherwise for the selected classifier type.                                                     |
|                                 |                                                                                                                                 |
|                                 | Choices: Any positive integer                                                                                                   |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| ``max_grad_norm``               | Maximum norm to which the accumulated gradients' norm is to be clipped.                                                         |
|                                 |                                                                                                                                 |
|                                 | Type: Union[float, None]                                                                                                        |
|                                 |                                                                                                                                 |
|                                 | Default: ``None``                                                                                                               |
|                                 |                                                                                                                                 |
|                                 | Choices: Any positive float, ``None``                                                                                           |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| ``optimizer``                   | Optimizer to use to minimize the network's stochastic objective function.                                                       |
|                                 |                                                                                                                                 |
|                                 | Type: str                                                                                                                       |
|                                 |                                                                                                                                 |
|                                 | Default: ``'Adam'``                                                                                                             |
|                                 |                                                                                                                                 |
|                                 | Choices: A valid name from `Pytorch optimizers <https://pytorch.org/docs/stable/optim.html#algorithms>`_                        |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| ``learning_rate``               | Parameter to control the size of weight and bias changes of the training algorithm as it learns.                                |
|                                 | `This <https://en.wikibooks.org/wiki/Artificial_Neural_Networks/Error-Correction_Learning>`_ article explains                   |
|                                 | Learning Rate in technical terms.                                                                                               |
|                                 |                                                                                                                                 |
|                                 | Type: float                                                                                                                     |
|                                 |                                                                                                                                 |
|                                 | Default: ``0.001``                                                                                                              |
|                                 |                                                                                                                                 |
|                                 | Choices: Any positive float                                                                                                     |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| ``validation_metric``           | The metric used to track model improvements on the validation data split.                                                       |
|                                 |                                                                                                                                 |
|                                 | Type: str                                                                                                                       |
|                                 |                                                                                                                                 |
|                                 | Default: ``'accuracy'`` for sequence classification and ``'f1'`` for token classification                                       |
|                                 |                                                                                                                                 |
|                                 | Choices: ``'accuracy'``, ``'f1'``                                                                                               |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| ``dev_split_ratio``             | The fraction of samples in the training data that are to be used for validation; sampled randomly.                              |
|                                 |                                                                                                                                 |
|                                 | Type: float                                                                                                                     |
|                                 |                                                                                                                                 |
|                                 | Default: ``0.2``                                                                                                                |
|                                 |                                                                                                                                 |
|                                 | Choices: A float between 0 and 1                                                                                                |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| ``padding_length``              | The maximum number of tokens (words, sub-words, or characters) allowed in a query. If a query has                               |
|                                 | more tokens than ``padding_length``, the surplus words are discarded. If specified as ``None``, the                             |
|                                 | ``padding_length`` of a mini-batch is simply the maximum length of inputs to that mini-batch.                                   |
|                                 |                                                                                                                                 |
|                                 | Type: Union[int, None]                                                                                                          |
|                                 |                                                                                                                                 |
|                                 | Default: ``None``                                                                                                               |
|                                 |                                                                                                                                 |
|                                 | Choices: Any positive integer, ``None``                                                                                         |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+
| ``avoid_whitespace_splitting``  | If specified ``True``, the classifier's input encoder does not distinguish whitespace with other characters.                    |
|                                 | Setting this to ``True`` is useful in the following two scenarios:                                                              |
|                                 |                                                                                                                                 |
|                                 | - When working with a language that has a non-whitespace script, e.g. Japanese, Chinese, etc.                                   |
|                                 |                                                                                                                                 |
|                                 | - When using a Huggingface pretrained model whose underlying pretrained tokenizer does not delimit words using                  |
|                                 | whitespace e.g.                                                                                                                 |
|                                 | `Byte-level BPE, SentencePiece, etc. <https://huggingface.co/docs/transformers/tokenizer_summary#summary-of-the-tokenizers>`_.  |
|                                 | Examples of such models are *distilroberta-base*, etc.                                                                          |
|                                 |                                                                                                                                 |
|                                 | Leaving this value as ``None`` assumes any value necessitated by the input encoder.                                             |
|                                 |                                                                                                                                 |
|                                 | Type: Union[bool, None]                                                                                                         |
|                                 |                                                                                                                                 |
|                                 | Default: ``None``                                                                                                               |
|                                 |                                                                                                                                 |
|                                 | Choices: ``None``, ``True``, ``False``                                                                                          |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+

Tokenization Choices
^^^^^^^^^^^^^^^^^^^^

.. _choices_for_tokenization:

A noteworthy distinction between the traditional suite of models versus the deep neural models suite is the way the inputs are prepared for the underlying model.
While the inputs for the former are prepared based on the specifications provided in the ``'features'`` key of the classifier's config, those of the latter are naive in the sense that they are simply a sequence of tokens in the input query;
the deep neural models do the heavy-lifting of discovering patters to classify the text.
Broadly, the tokens can be created with individual characters or group of characters (aka. sub-words) or words split at whitespace.
Based on the choice of tokenization, a sequence of tokens are obtained from the input queries which are then converted into a sequence of ids for the neural model.
The neural suite has the following choices of tokenizations to prepare inputs for neural models.

.. note::

   - To use a specific tokenization strategy, simply set the ``tokenizer_type`` param to one of the following choices, e.g. {``tokenizer_type``: ``'whitespace-tokenizer'``}.
   - Note that some of strategies are specific to the choice of embedder being used in the classifier.

.. warning::

   The choices of tokenization presented here shouldn't be confused with the :ref:`Tokenizers in text preparation pipeline <tokenization_text_preparation_pipeline>`. The latter are used to preprocess the text while the former are used to prepare sequence of tokens for the neural models.

1. ``'whitespace-tokenizer'``
"""""""""""""""""""""""""""""

A Whitespace tokenizer tokenizes a query into a sequence of tokens by splitting it at whitespaces.
Therefore the tokens are simply the words present in the query.
This tokenization strategy is state-less and the sequence of tokens produced will be same irrespective of the queries present in the training data.

2. ``'char-tokenizer'``
"""""""""""""""""""""""

A Character tokenizer tokenizes a query into a sequence of characters present in it.
This tokenization strategy is state-less and the sequence of tokens produced will be same irrespective of the queries present in the training data.

3. ``'bpe-tokenizer'``
""""""""""""""""""""""

A :ref:`Byte-Pair Encoding (BPE) tokenizer <https://huggingface.co/docs/transformers/tokenizer_summary#bytepair-encoding-bpe>` tokenizes a query into a sequence of sub-words based on a vocabulary created from all of the queries in the training data.
This tokenization strategy is state-ful and the sequence of tokens produced might not be same if the queries present in the training data change.
This tokenizer is implemented using the :ref: Hugginface's Tokenizer library <https://huggingface.co/docs/tokenizers/python/latest/index.html>`.

4. ``'wordpiece-tokenizer'``
""""""""""""""""""""""""""""

A :ref:`Word-Piece tokenizer <https://huggingface.co/docs/transformers/tokenizer_summary#wordpiece>` tokenizes a query into a sequence of sub-words based on a vocabulary created from all of the queries in the training data.
This tokenization strategy is state-ful and the sequence of tokens produced might not be same if the queries present in the training data change.
This tokenizer is implemented using the :ref: Hugginface's Tokenizer library <https://huggingface.co/docs/tokenizers/python/latest/index.html>`.

5. ``'huggingface_pretrained-tokenizer'``
"""""""""""""""""""""""""""""""""""""""""

A tokenizer pretrained and available as part of :ref:`Huggingface transformers <https://huggingface.co/docs/transformers/index>` library.
Although this tokenization strategy is state-ful due to its pretraining, the sequence of tokens produced will be same irrespective of the queries present in the training data.
To use this tokenizer, set the ``tokenizer_type`` and ``pretrained_model_name_or_path`` keys appropriately as follows: {``tokenizer_type``: ``'huggingface_pretrained-tokenizer'``, ``pretrained_model_name_or_path``: ``'distilbert-base-uncased'``}.
