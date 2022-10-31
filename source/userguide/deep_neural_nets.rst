Deep Neural Networks in MindMeld
================================

Conversational AI and Natural Language Processing more generally have seen a boost in performance at a variety of tasks through the use of `Deep learning <https://en.wikipedia.org/wiki/Deep_learning>`_. In particular, deep neural models based on :wiki_api:`Convolutional neural networks (CNNs) <Convolutional_neural_network#Natural_language_processing>`, :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` and :wiki_api:`Transformers <Transformer_(machine_learning_model)>` architectures have been widely adopted over more traditional approaches to NLP to great success. MindMeld now extends its suite of traditional machine learning models (e.g. :sk_guide:`Logistic regression <linear_model.html#logistic-regression>`, :sk_guide:`Decision tree <tree.html#tree>`, etc.) with a variety of deep neural models and an array of configurable parameters.

Users can now train and use deep neural models for :ref:`domain classification <domain_classification>` and :ref:`intent classification <intent_classification>` (aka. sequence classification) as well as for :ref:`entity recognition <entity_recognition>` (or token classification) tasks.

.. note::

   These models are implemented using `Pytorch <https://pytorch.org/>`_ framework and thus need extra installations before starting to use them in your chatbot application. Please make sure to install the Pytorch requirement by running in the shell:

   :code:`pip install mindmeld[torch]`

MindMeld supports the use of pretrained transformer models such as :wiki_api:`BERT <BERT_(language_model)>` through the popular `Huggingface Transformers <https://huggingface.co/docs/transformers/index>`_ library.
Several pretrained models from their `Models Hub <https://huggingface.co/models>`_ that can be used for sequence classification or token classification can be employed in your chatbot application.

.. note::

   To use pretrained transformer models, install the extra *transformers* requirement by running in the shell:

   :code:`pip install mindmeld[transformers]`

Before proceeding to use the deep neural models, consider the following possible advantages and disadvantages of using them in place of traditional machine learning models.

- **Better overall performance for larger training sets.** Deep models generally outperform traditional machine learning models on training sets with several hundreds or thousands of queries when training from scratch, and with at least few hundred if fine-tuning from a pretrained checkpoint.
- **Slower training and inference times on CPU devices but faster on GPU devices.** Training and inference times for deep models on CPU-only machines can take longer than traditional machine learning models. However, on `GPU-enabled devices <https://developer.nvidia.com/deep-learning>`_, the run times of the deep networks can be comparable to some of the traditional models in MindMeld.
- **Minimal feature engineering work but manual hyperparameter tuning.** Unlike traditional machine learning models, deep models require little or no feature engineering work because they infer input features (such as word embeddings). Traditional models must take into account several hundred engineered features (n-grams, system entities, and so on), which requires fine-grained tuning. On the flip side, Mindmeld's deep models don't have automated hyperparameter tuning methods like :sk_api:`sklearn.model_selection.GridSearchCV <sklearn.model_selection.GridSearchCV.html>`, which are available for their traditional counterparts. While the default hyperparameters for MindMeld's deep neural models work well across datasets, you can further tune them and a good starting point to understand this subject better is Andrej Karpathy's `course notes <https://cs231n.github.io/neural-networks-3/#baby>`_ from the Convolutional Neural Networks for Visual Recognition course at Stanford University.
- **Larger disk storage required.** While deep neural models can have a similar disk storage footprint to their traditional counterparts, depending on your data, it is not uncommon for them to require more disk storage space.

.. note::

   - To use deep neural networks instead of traditional machine learning models in your MindMeld application, simply make few modifications to the classifier configuration dictionaries for all or selected classifiers in your app's ``config.py``.
   - To make modifications to selected domains or intents, recollect that you can implement the :ref:`get_intent_classifier_config() <get_intent_classifier_config>` and :ref:`get_entity_recognizer_config() <get_entity_recognizer_config>` functions respectively in your app's ``config.py`` for a finer-grained control.

In the following sections, different model architectures and their configurable parameters are outlined.

Domain and Intent classification
--------------------------------

.. _dnns_sequence_classification:

Using MindMeld’s deep neural models requires configuring only two keys in your classifier configuration dictionaries: ``'model_settings'`` and ``'params'``.
When working with the deep models, the ``'features'`` and ``'param_selection'`` keys in the classifier configuration are redundant, as we neither have to handcraft any feature sets for modeling, nor is there automated hyperparameter tuning.

This is a departure from other documentation on :ref:`Working with the Domain Classifier <domain_classifier_configuration>` and :ref:`Working with the Intent Classifier <intent_classifier_configuration>`, which outlines that text classifier configuration requires an additional two keys (``'features'`` and ``'param_selection'``).

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

Mindmeld's ``'embedder'`` classifier type uses a pooling operation on top of model embeddings, which are based on either a lookup table or a deep neural model:

- **Lookup table embeddings** can be derived based on a user-defined tokenization strategy-- word-level, sub-word-level, or character-level tokenization (see :ref:`Tokenization Choices <choices_for_tokenization>` below for more details). By default, the lookup table is randomly initialized, but it can instead be initialized to a pretrained checkpoint when using a word-level tokenization strategy (such as `GloVe <https://nlp.stanford.edu/projects/glove/>`_) .

- **Deep contextualized embedders** are pretrained embedders in the style of :wiki_api:`BERT <BERT_(language_model)>`, which consists of its own tokenization strategy and neural embedding process.

In either case, all the underlying weights can be tuned to the training data provided, or can be kept frozen during the training process.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

.. note::

   Specify the embedding choice using the param ``embedder_type``. Set it to ``None``, ``'glove'`` or ``'bert'`` to use with desired embedding styles-- based on a randomly initialized embedding lookup table, based on lookup table initialized with GloVe (or GloVe-like formatted) pretrained embeddings or a BERT-like pretrained transformer based deep contextualized embedder, respectively.

The following are the different optional params that are configurable along with the chosen choice of ``embedder_type`` param.
See :ref:`Common Configurable Params <common_configurable_params>` section for list of additional configurable params that are common across classifiers.

1.1 Embedding Lookup Table (``embedder_type``: ``None``)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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

Below is a minimal working example of a sequence classifier configuration for a classifier based on an embedding lookup table:

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


1.2 Pretrained Embedding Lookup Table (``embedder_type``: ``glove``)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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

Below is a minimal working example of a sequence classifier configuration for a classifier based on a pretrained-initialized embedding lookup table:

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

1.3 Deep Contextualized Embeddings (``embedder_type``: ``bert``)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
| ``save_frozen_embedder``                | If set to ``False``, the weights of the underlying bert-like embedder that are not being tuned are not dumped to disk upon calling a classifier's .dump() method. This boolean key is only valid when ``update_embeddings`` is set to ``False``. |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Type: bool                                                                                                                                                                                                                                       |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Default: ``False``                                                                                                                                                                                                                               |
|                                         |                                                                                                                                                                                                                                                  |
|                                         | Choices: ``True``, ``False``                                                                                                                                                                                                                     |
+-----------------------------------------+--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

Below is a minimal working example of a sequence classifier configuration for a classifier based on a BERT-like embedder:

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

Using a sequence of textual tokens extracted from the input text, the first layer of this classifier type embeds those sequences into low-dimensional vectors using an embedding lookup table.
The subsequent layer performs convolutions over the sequence of embedded word vectors using kernels (also called *filters*); kernels of different lengths capture different *n*-gram patterns from the input text.
For each chosen length, several kernels are used to capture different patterns at the same receptive range.
Finally, each kernel leads to one feature map.

Each feature map is reduced to the maximum value observed in that map, and maximum values from all maps are combined to form a long feature vector.
This vector is analogous to an ``'embedder'`` classifier's pooled output, which is then passed through a classification layer.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

The following are the different optional params that are configurable with the ``'cnn'`` classifier type.
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

Below is a minimal working example of a sequence classifier configuration for a classifier based on CNNs:

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

:wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` based text classifiers utilize recurrent feedback connections to be able to learn temporal dependencies in sequential data.

Using a sequence of textual tokens extracted from the input text, the first layer of this classifier type embeds those sequences into low-dimensional vectors using an embedding lookup table.
The subsequent layer applies LSTM over the sequence of embedded word vectors.
An LSTM's ability to maintain temporal information is generally dependent on its *hidden* dimension.
The LSTM processes the text from left-to-right or in the case of a bi-directional LSTM (bi-LSTM), it can process the text both ways, from left-to-right and right-to-left.
This yields an output sequence of one vector per token of the input text.
Optionally, several LSTMs can then be stacked, with the output of one serving as the input to another.

To obtain a single vector per input text, the vectors for each token can be pooled or the last vector in the sequence can simply be used as the representative vector.
This vector is analogous to an ``'embedder'`` classifier's pooled output, which is then passed through a classification layer.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

The following are the different optional params that are configurable with the ``'lstm'`` classifier type.
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

Below is a minimal working example of a sequence classifier configuration for a classifier based on LSTMs:

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

.. _deep_ner:

Entity recognition
------------------

.. _dnns_token_classification:

Using MindMeld’s deep neural models requires configuring only two keys in your classifier configuration dictionaries: ``'model_settings'`` and ``'params'``.
When working with the deep models, the ``'features'`` and ``'param_selection'`` keys in the classifier configuration are redundant, as we neither have to handcraft any feature sets for modeling, nor is there automated hyperparameter tuning.

This is a departure from other documentation on :ref:`Working with the Entity Recognizer <entity_recognizer_configuration>`, which outlines that text classifier configuration requires an additional two keys (``'features'`` and ``'param_selection'``).

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
See :ref:`Common Configurable Params <common_configurable_params>` section for a list of configurable params that are common across all classifier types.

1. ``'embedder'`` classifier type
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_tokens_classification_models_embedder:

This classifier type includes neural models that are based on either an embedding lookup table or a deep contextualized embedder, the outputs of which are then passed through a `Conditional Random Field (CRF) <https://en.wikipedia.org/wiki/Conditional_random_field>`_ or a `Softmax layer <https://en.wikipedia.org/wiki/Softmax_function>`_  which labels target word as a particular entity.

- **Lookup table embeddings** can be derived based on a user-defined tokenization strategy-- word-level, sub-word-level, or character-level tokenization (see :ref:`Tokenization Choices <choices_for_tokenization>` below for more details). By default, the lookup table is randomly initialized, but it can instead be initialized to a pretrained checkpoint when using a word-level tokenization strategy (such as `GloVe <https://nlp.stanford.edu/projects/glove/>`_) .

- **Deep contextualized embedders** are pretrained embedders in the style of :wiki_api:`BERT <BERT_(language_model)>`, which consists of its own tokenization strategy and neural embedding process.

In either case, all the underlying weights can be tuned to the training data provided, or can be kept frozen during the training process.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

The ``'embedder'`` classifier type pools the vectors of all tokens corresponding to a word that has been assigned an entity tag, so as to obtain a single vector per word in an input text.
This is unlike sequence classification models, where all tokens of all words are pooled together, and then passed through a classification layer.

.. note::

   Specify the embedding choice using the param ``embedder_type``. Set it to ``None``, ``'glove'`` or ``'bert'`` to use with desired embedding styles-- based on a randomly initialized embedding lookup table, based on lookup table initialized with GloVe (or GloVe-like formatted) pretrained embeddings or a BERT-like pretrained transformer based deep contextualized embedder, respectively.

The following are the different optional params that are configurable along with the chosen choice of ``embedder_type`` param.
See :ref:`Common Configurable Params <common_configurable_params>` for list of additional configurable params that are common across classifiers.

1.1 Embedding Lookup Table (``embedder_type``: ``None``)
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

Below is a minimal working example of a token classifier configuration for a classifier based on an embedding lookup table:

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

1.2 Pretrained Embedding Lookup Table (``embedder_type``: ``glove``)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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

Below is a minimal working example of a token classifier configuration for a classifier based on a pretrained-initialized embedding lookup table:

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

1.3 Deep Contextualized Embeddings (``embedder_type``: ``bert``)
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

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
| ``save_frozen_embedder``                | If set to ``False``, the weights of the underlying bert-like embedder that are not being tuned are not dumped to disk upon calling a classifier's .dump() method. This boolean key is only valid when ``update_embeddings`` is set to ``False``. |
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

Below is a minimal working example of a token classifier configuration for a classifier based on a BERT embedder:

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

:wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` based text classifiers utilize recurrent feedback connections to be able to learn temporal dependencies in sequential data.

Using a sequence of textual tokens extracted from the input text, the first layer of this classifier type embeds those sequences into low-dimensional vectors using an embedding lookup table.
The subsequent layer applies LSTM over the sequence of embedded word vectors.
An LSTM's ability to maintain temporal information is generally dependent on its *hidden* dimension.
The LSTM processes the text from left-to-right or in the case of a bi-directional LSTM (bi-LSTM), it can process the text both ways, from left-to-right and right-to-left.
This yields an output sequence of one vector per token of the input text.
Optionally, several LSTMs can then be stacked, with the output of one serving as the input to another.

To obtain a single vector per word per input text, the vectors of all tokens corresponding to each word (for which an entity tag is to be ascertained) are pooled.
This vector is analogous to an ``'embedder'`` classifier's output, which is then passed through a classification layer.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

The following are the different optional params that are configurable with the ``'lstm'`` classifier type.
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

Below is a minimal working example of a token classifier configuration for a classifier based on LSTMs:

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

:wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` based text classifiers utilize recurrent feedback connections to be able to learn temporal dependencies in sequential data.
When coupled with :wiki_api:`Convolutional neural networks (CNN) <Convolutional_neural_network#Natural_language_processing>` for extracting character-level features from input text, the overall architecture can better model the textual data as well as it is more robust to variations in the spellings.

Using a sequence of textual tokens extracted from the input text, the first layer of this classifier type embeds those sequences into low-dimensional vectors using an embedding lookup table.
This is then concatenated with the outputs of each word's convolutions at the character-level using kernels of different lengths to capture different patterns.
These convolutions are similar to those of :ref:`CNN classifier type <dnns_sequence_classification_models_cnn>` except they are applied for each word in the input text separately to obtain one representation for each word.

The subsequent layer applies LSTM over the sequence of concatenated word vectors.
An LSTM's ability to maintain temporal information is generally dependent on its *hidden* dimension.
The LSTM processes the text from left-to-right or in the case of a bi-directional LSTM (bi-LSTM), it can process the text both ways, from left-to-right and right-to-left.
This yields an output sequence of one vector per token of the input text.
Optionally, several LSTMs can then be stacked, with the output of one serving as the input to another.

The  ``'cnn-lstm'`` classifier type pools the vectors of all tokens corresponding to words that have been assigned an entity tag so as to obtain a single vector per word in an input text.
This vector is analogous to an ``'embedder'`` classifier's output, which is then passed through a classification layer.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

The following are the different optional params that are configurable with the ``'lstm'`` classifier type.
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

Below is a minimal working example of a token classifier configuration for a classifier based on CNN-LSTM:

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

:wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` based text classifiers utilize recurrent feedback connections to be able to learn temporal dependencies in sequential data.
When coupled with :wiki_api:`Long short-term memory networks (LSTM) <Long_short-term_memory>` for extracting character-level features from input text, the overall architecture can better model the textual data as well as it is more robust to variations in the spellings.

Using a sequence of textual tokens extracted from the input text, the first layer of this classifier type embeds those sequences into low-dimensional vectors using an embedding lookup table, and
concatenates them with the outputs of a character-level bi-LSTM (for each word individually) to capture different character-level patterns.
This is then concatenated with the outputs of each word's convolutions at the character-level using kernels of different lengths to capture different patterns.
These convolutions are similar to those of :ref:`CNN classifier type <dnns_sequence_classification_models_cnn>` except they are applied for each word in the input text separately to obtain one representation for each word.

The subsequent layer applies LSTM over the sequence of concatenated word vectors.
An LSTM's ability to maintain temporal information is generally dependent on its *hidden* dimension.
The LSTM processes the text from left-to-right or in the case of a bi-directional LSTM (bi-LSTM), it can process the text both ways, from left-to-right and right-to-left.
This yields an output sequence of one vector per token of the input text.
Optionally, several LSTMs can then be stacked, with the output of one serving as the input to another.

The  ``'lstm-lstm'`` classifier type pools the vectors of all tokens corresponding to words that have been assigned an entity tag so as to obtain a single vector per word in an input text.
This vector is analogous to an ``'embedder'`` classifier's output, which is then passed through a classification layer.
Dropout layers are used as regularizers to avoid over-fitting, which is a more common phenomenon when working with small sized datasets.

The following are the different optional params that are configurable with the ``'lstm'`` classifier type.
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

Below is a minimal working example of a token classifier configuration for a classifier based on LSTM-LSTM:

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
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _dnns_token_classification_models_lstm_tensorflow:

A `Tensorflow <https://www.tensorflow.org/>`_ backed implementation of `Bi-Directional Long Short-Term Memory (LSTM) Network <https://en.wikipedia.org/wiki/Bidirectional_recurrent_neural_networks>`_.

.. note::

   To use this classifier type, make sure to install the Tensorflow requirement by running in the shell:

   :code:`pip install mindmeld[tensorflow]`

The MindMeld Bi-Directional LSTM network

 - encodes words as pre-trained word embeddings using Stanford's `GloVe representation <https://nlp.stanford.edu/projects/glove/>`_
 - encodes characters using a convolutional network trained on the training data
 - concatenates the word and character embeddings together and feeds them into the bi-directional LSTM
 - couples the forget and input gates of the LSTM using a peephole connection, to improve overall accuracies on downstream NLP tasks
 - feeds the output of the LSTM into a `linear chain Conditional Random Field <https://en.wikipedia.org/wiki/Conditional_random_field>`_ (CRF) or `Softmax layer <https://en.wikipedia.org/wiki/Softmax_function>`_  which labels the target word as a particular entity

The following are the different optional params that are configurable with the ``'lstm'`` classifier type.
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
| ``query_text_type``             | Determines the choice of text that is fed into the neural model. This param is coupled with the                                 |
|                                 | *Text Preparation Pipeline* when using a choice other than ``'text'``. The following are the                                    |
|                                 | three available choices:                                                                                                        |
|                                 |                                                                                                                                 |
|                                 | ``'text'``: Specifies that the raw text of the queries be used without any processing. This is an                               |
|                                 | appropriate choice if using a Huggingface pretrained model as the pretrained model's tokenizer takes                            |
|                                 | care of any processing, tokenization, and normalizations on top of the raw text. This choice is oblivious                       |
|                                 | to the app's :attr:`TEXT_PREPARATION_CONFIG`.                                                                                   |
|                                 |                                                                                                                                 |
|                                 | ``'processed_text'``: Specifies that processed text of the Text Preparation Pipeline be used as input to the                    |
|                                 | neural model.                                                                                                                   |
|                                 |                                                                                                                                 |
|                                 | ``'normalized_text'``: Specifies that the text upon processing, tokenization, and normalization steps of the                    |
|                                 | Text Preparation Pipeline be used as input to the neural model.                                                                 |
|                                 |                                                                                                                                 |
|                                 | Type: str                                                                                                                       |
|                                 |                                                                                                                                 |
|                                 | Default: ``'processed_text'`` for sequence classification and ``'normalized_text'`` for token classification                    |
|                                 |                                                                                                                                 |
|                                 | Choices: ``'text'``, ``'processed_text'``, ``'normalized_text'``                                                                |
+---------------------------------+---------------------------------------------------------------------------------------------------------------------------------+

Tokenization Choices
^^^^^^^^^^^^^^^^^^^^

.. _choices_for_tokenization:

A noteworthy distinction between the traditional suite of models versus the deep neural models suite is the way the inputs are prepared for the underlying model.
While the inputs for the former are prepared based on the specifications provided in the ``'features'`` key of the classifier's config, inputs of deep neural models are naive in the sense that they are simply a sequence of tokens in the input query;
the deep models do the heavy-lifting of discovering patterns to classify the text.

Broadly, tokens can be extracted from an input text as a sequence of individual characters or group of characters (aka. sub-words) or words itself by simply splitting the input text at whitespaces.
Based on the choice of tokenization, a sequence of tokens are obtained from the input queries which are then converted into a sequence of ids for the neural model.


.. note::

   - To use a specific tokenization strategy, simply set the ``tokenizer_type`` param to one of the following choices (e.g. {``tokenizer_type``: ``'whitespace-tokenizer'``}).
   - Note that some of strategies are specific to the choice of embedder being used in the classifier.

.. warning::

   The choices of tokenization presented here shouldn't be confused with the :ref:`Tokenizers in text preparation pipeline <tokenization_text_preparation_pipeline>`. The tokenizers in text preparation pipeline are used to develop text that is inputted to the neural models while the following are used to prepare sequence of tokens for the underlying embedders.

The neural suite has the following choices of tokenizations to prepare inputs for neural models.

1. ``'whitespace-tokenizer'``
"""""""""""""""""""""""""""""

A Whitespace tokenizer tokenizes a query into a sequence of tokens by splitting it at whitespaces.
The result are tokens that are simply the words present in the query.
This tokenization strategy is state-less and the sequence of tokens produced for an input text will be same irrespective of the queries present in the training data.

2. ``'char-tokenizer'``
"""""""""""""""""""""""

A Character tokenizer tokenizes a query into a sequence of characters present in it.
This tokenization strategy is state-less and the sequence of tokens produced for an input text will be same irrespective of the queries present in the training data.

3. ``'bpe-tokenizer'``
""""""""""""""""""""""

A `Byte-Pair Encoding (BPE) tokenizer <https://huggingface.co/docs/transformers/tokenizer_summary#bytepair-encoding-bpe>`_ tokenizes a query into a sequence of sub-words based on a vocabulary created from all of the queries in the training data.
This tokenization strategy is state-ful and the sequence of tokens produced for an input text might not be same if the queries present in the training data change.
This tokenizer is implemented using the `Huggingface's Tokenizer library <https://huggingface.co/docs/tokenizers/python/latest/index.html>`_.

4. ``'wordpiece-tokenizer'``
""""""""""""""""""""""""""""

A `Word-Piece tokenizer <https://huggingface.co/docs/transformers/tokenizer_summary#wordpiece>`_ tokenizes a query into a sequence of sub-words based on a vocabulary created from all of the queries in the training data.
This tokenization strategy is state-ful and the sequence of tokens produced for an input text might not be same if the queries present in the training data change.
This tokenizer is implemented using the `Huggingface's Tokenizer library <https://huggingface.co/docs/tokenizers/python/latest/index.html>`_.

5. ``'huggingface_pretrained-tokenizer'``
"""""""""""""""""""""""""""""""""""""""""

A tokenizer pretrained and available as part of `Huggingface transformers <https://huggingface.co/docs/transformers/index>`_ library.
Although this tokenization strategy is state-ful (due to its pretraining), the sequence of tokens produced for an input text will be same irrespective of the queries present in the training data.
To use this tokenizer, set the ``tokenizer_type`` and ``pretrained_model_name_or_path`` keys appropriately as follows: {``tokenizer_type``: ``'huggingface_pretrained-tokenizer'``, ``pretrained_model_name_or_path``: ``'distilbert-base-uncased'``}.
