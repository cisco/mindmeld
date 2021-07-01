Active Learning in MindMeld
===========================

Once a conversational application is deployed, capturing user logs can be highly beneficial to understand new trends and diversity in terms of how users are phrasing their queries for different intents. These logs can improve the quality of the training data and overall classifier performance. However, manually analyzing these logs and annotating them with the correct domain, intent and entity labels is costly and time consuming. Moreover, as the number of users for the system increase, the number of logs to be annotated also increases by many folds. This reduces the scalability of the manual annotation process. For such scenarios, we can use Active Learning.

What is Active Learning?
^^^^^^^^^^^^^^^^^^^^^^^^
In this section, we take a closer look at what active learning is and how it works within MindMeld.

Active Learning is an approach to continuously and iteratively pick data that is most informative for the models to train on from a pool of data-points. By choosing these informative data-points, active learning provides a higher chance of improving the accuracy of the model with fewer training examples. If the pool of data-points is unannotated, active learning also greatly reduces the number of queries that need manual annotation. MindMeld provides this inbuilt functionality to select these must-have queries from existing data or additional logs and datasets to get the best out of your conversational applications.

MindMeld's NLP pipeline consists of a heirarchy of classifier and tagger models for domain, intent, entity and role classification. Each classifier takes as input the query text and extracted features to generate a probability distribution for the corresponding classes in the heirarchy. These probability distributions can be strong indicator of the classifier's confidence in its predictions and can be useful in determining whether the query needs further reflection. 

Let's say for a query A from the logs, the selected class (or the highest probability class) in the domain classifier is at 0.95 or 95% and the remaining 5% is distributed amongst the rest of the domains, this could potentially indicate a high confidence by the classifier in its prediction. Whereas a query B has a probability 0.5 for the selected class, possibly indicating lower prediction confidence. Based on these values, the active learning strategies will select query B over query A for adding back to the training data. With query B now in the training data with proper annotation, the system has a higher chance of performing better on similar confusable queries. Later sections explain how different strategies use these confidence scores to select important queries.

There are two phases in MindMeld's active learning pipeline. One for configuring the best hyperparamters (such as optimal strategy) for the pipeline and the other for selecting the most important subset of queries from logs given the configured pipeline. These phases are referred to as ``Strategy Tuning`` and ``Query Selection``.


Quick Reference
^^^^^^^^^^^^^^^
This section is a quick reference on the command-line usage of the active learning Strategy Tuning (``tune``) and Query Selection (``select``) commands. These are detailed in the next sections.

Strategy tuning

.. code-block:: console

    mindmeld active_learning --train --app-path '<PATH>/app_name/' --output_folder '<PATH>'

Query selection

.. code-block:: console

    mindmeld active_learning --select --app-path "<PATH>/app_name/" --output_folder '<PATH>' --unlabeled_logs_path "<PATH>/logs.txt"

.. note::
    
    Running these commands without defining a custom active learning configuration in ``config.py`` would result in the use of a default configuration. The custom configuration settings and MindMeld's default active learning configuration are explained in the next section.

.. note::
    
    Accuracy and query selection results can be found in the directory ``output_folder/results`` in files ``accuracies.json`` and ``selected_queries.json`` respectively. Plots for the tuning results are saved in ``output_folder/plots``.


Defining the AL config
^^^^^^^^^^^^^^^^^^^^^^
The first step towards customizing the active learning pipeline to your application is configuration adjustment. This section details different components in the configuration and provides a default configuration setup which is applied if no active learning config is defined in ``config.py``.

1. **output_folder** - Directory path to save the output of the active learning pipeline.

2. **pre_tuning**

+------------------------+--------+----------------------------------------------------------------------------------+
| Configuration Key      | Type   | Definition and Examples                                                          |
+========================+========+==================================================================================+
| train_pattern          | str    | Regex pattern to match train files. Example for all train files ".*train.*.txt"  |
+------------------------+--------+----------------------------------------------------------------------------------+
| test_pattern           | str    | Regex pattern to match train files. Example for all test files ".*test.*.txt"    |
+------------------------+--------+----------------------------------------------------------------------------------+
| train_seed_pct         | float  | Percentage of training data to use as the initial seed                           |
+------------------------+--------+----------------------------------------------------------------------------------+


3. **tuning**

+------------------------+------------+----------------------------------------------------------------------------------+
| Configuration Key      | Type       | Definition and Examples                                                          |
+========================+============+==================================================================================+
| n_classifiers          | int        | Number of classifier instances to be used by ensemble strategies                 |
+------------------------+------------+----------------------------------------------------------------------------------+
| n_epochs               | int        | Number of turns strategy tuning is run on the complete log data                  |
+------------------------+------------+----------------------------------------------------------------------------------+
| batch_size             | int        | Number of queries to sample at each iteration from the logs                      |
+------------------------+------------+----------------------------------------------------------------------------------+
| tuning_level           | str        | Selecting classifiers to use for tuning, options: "domain" or "intent"           |
+------------------------+------------+----------------------------------------------------------------------------------+
| tuning_strategies      | List[str]  | List of strategies (heuristics) to use for tuning                                |
+------------------------+------------+----------------------------------------------------------------------------------+

4. **tuning_output**

+------------------------+------------+----------------------------------------------------------------------------------+
| Configuration Key      | Type       | Definition and Examples                                                          |
+========================+============+==================================================================================+
| save_sampled_queries   | bool       | Whether to save the queries sampled at each iteration                            |
+------------------------+------------+----------------------------------------------------------------------------------+
| aggregate_statistic    | str        | Metric to record aggregate classifier performance data in the output and plots,  |
|                        |            | options: "accuracy", "f1_weighted", "f1_macro", "f1_micro"                       |
+------------------------+------------+----------------------------------------------------------------------------------+
| class_level_statistic  | str        | Metric to record class/tuning level performance data in the output and plots,    |
|                        |            | options: "f_beta", "percision", "recall"                                         |
+------------------------+------------+----------------------------------------------------------------------------------+

5. **query_selection**

+------------------------+------------+----------------------------------------------------------------------------------+
| Configuration Key      | Type       | Definition and Examples                                                          |
+========================+============+==================================================================================+
| selection_strategy     | str        | Strategy (heuristic) to use for log selection                                    |
+------------------------+------------+----------------------------------------------------------------------------------+
| log_usage_pct          | float      | Percentage of the log data to use for selection                                  |
+------------------------+------------+----------------------------------------------------------------------------------+
| labeled_logs_pattern   | str        | Regex pattern to match log files if already labeled. For example, ".*log.*.txt"  |
+------------------------+------------+----------------------------------------------------------------------------------+
| unlabeled_logs_path    | str        | Path to text file containing unlabeled queries from user logs or other resources |
+------------------------+------------+----------------------------------------------------------------------------------+


If there is no configuration defined in the ``config.py`` file or if fields are missing in the custom configuration, the relevant missing information is obtained from the following default configuration:

.. code-block:: python

    DEFAULT_ACTIVE_LEARNING_CONFIG = {
        "output_folder": None,
        "pre_tuning": {
            "train_pattern": ".*train.*.txt",
            "test_pattern": ".*test.*.txt",
            "train_seed_pct": 0.20,
        },
        "tuning": {
            "n_classifiers": 3,
            "n_epochs": 5,
            "batch_size": 100,
            "tuning_level": "domain",
            "tuning_strategies": [
                "LeastConfidenceSampling",
                "MarginSampling",
                "EntropySampling",
                "RandomSampling",
                "DisagreementSampling",
                "EnsembleSampling",
                "KLDivergenceSampling",
            ],
        },
        "tuning_output": {
            "save_sampled_queries": True,
            "aggregate_statistic": "accuracy",
            "class_level_statistic": "f_beta",
        },
        "query_selection": {
            "selection_strategy": "EntropySampling",
            "log_usage_pct": 1.00,
            "labeled_logs_pattern": None,
            "unlabeled_logs_path": "logs.txt",
        },
    }

Before diving deeper into strategy tuning and query selection, let's take a look at the different strategies and tuning levels. These hyperparameters are studied at the strategy tuning level with the best ones chosen for query selection.

Strategies
----------
The tuning step allows the application to run 7 possible strategies and choose the best performing one. Each strategy is a sampling function that samples the worst performing queries from the latest batch iteration of the training data. The assessment of worst performance comes from the classifiers' confidence in the predictions for that query. All heuristics use this information differently as described next.

+---------------------------+-----------------------------------------------------------------------------------------------+
| Strategy                  | How does it work?                                                                             |
+===========================+===============================================================================================+
| Random Sampling           | Samples the next set of queries at random.                                                    |
+---------------------------+-----------------------------------------------------------------------------------------------+
| Least Confidence Sampling | From the available queries in the batch, this sampling strategy samples queries with the      |
|                           | lowest max confidence score across any class, i.e., queries that the classifier is least      |
|                           | confident about the selected class.                                                           |
+---------------------------+-----------------------------------------------------------------------------------------------+
| Entropy Sampling          | Calculates the entropy score of the classifier confidences per query. Samples the ones with   |
|                           | highest entropy.                                                                              |
+---------------------------+-----------------------------------------------------------------------------------------------+
| Margin Sampling           | From the available queries in the batch, this sampling strategy samples queries that have the |
|                           | lowest confidence score difference between the top two class confidence scores for the query. |
|                           | This difference is referred to as the "margin".                                               |
+---------------------------+-----------------------------------------------------------------------------------------------+
| Disagreement Sampling     | Across n runs of the classifier, this sampling strategy calculates an agreement score for     |
|                           | every query (% of classifiers that voted for the most frequent class). The queries are then   |
|                           | ranked from lowest classifier agreement to highest and the sampled in order.                  |
+---------------------------+-----------------------------------------------------------------------------------------------+
| KL Divergence Sampling    | Across n runs of the classifier, this sampling strategy calculates the KL divergence between  |
|                           | average confidence distribution across all classifiers for a given class and the confidence   |
|                           | distribution for a given query for that class. Queries with higher divergence are sampled.    |
+---------------------------+-----------------------------------------------------------------------------------------------+
| Ensemble Sampling         | Combines ranks from all the above heuristics and samples in order.                            |
+---------------------------+-----------------------------------------------------------------------------------------------+

Tuning Levels
-------------

Since MindMeld defines a hierarchy of domains and intents, the various heuristics can be computed by using the confidence scores or probabilities of either the domain or intent classifiers or both. This level is indicated by the ``tuning_level`` in the config. 

* For the domain level, the domain classifier is run and the probability scores of the classifier are passed to the strategies.
* For the intent level, the intent classifier probability scores across all domains are combined into a single vector and passed on to the strategies.

Strategy Tuning
^^^^^^^^^^^^^^^
The strategy tuning phase in the active learning pipeline is useful in choosing the best tuning strategies for your application. We partition the existing training data in the app as a seed set and unsampled set according to the  ``train_seed_pct`` value mentioned in the config file. Note that the pipeline only uses the data from files that match the ``train_pattern`` regex in the config file in this step.  Next, the classifiers are trained on this seed data and evaluated on the existing test data (files matching ``test_pattern``). Iteratively, data equivalent to the ``batch_size`` is added from the remaining unsampled training data to the initial seed data by using the different sampling strategies discussed earlier. The classifier models are retrained with the new expanded seed set and evaluated against the test set. In this process the train set keeps growing till all training data has been consumed by the model and the final iteration of classifier training has been trained on all possible training data. 

.. image:: /images/strategy_tuning.png
    :align: center
    :name: strategy_tuning_flow

The accuracy, query selection and tuning strategy performance results are saved along with performance plots tracking the classifier performance in tandem with the increasing training data. We repeat the above Tuning process for ``n_epochs`` (as defined in the config) to obtain average active learning performance. Results obtained at the end can be used to quantitatively and visually choose the best tuning approach for your application.

The following command can be used to run tuning using the settings defined in the application configuration:

.. code-block:: console

    mindmeld active_learning --tune --app-path '<PATH>/app_name/' --output_folder '<PATH>'

Flags for application path and output folder are required and overwrite the default configuration settings for active learning. In addition to the aforementioned required flags, the following optional flags can be used - tuning_level, batch_size, n_epochs, train_seed_pct, and plot. These are described in detail in the AL config section.

At the end of the tuning process, results are stored in the ``output_folder``. The ``accuracy.json`` file in the directory ``output_folder/results`` consist of strategy performance on the application's test/evaluation data for every iteration and epoch. ``selected_queries.json`` consists of the same information but instead of evaluation performance, this file records the queries selected at that iteration. the ``output_folder/plots`` directory consists of this quantitative information in a  visual format. The plots record performance of all chosen strategies across data iterations and gives a sense of which strategy is best suited for your application. The same information can be gauged from these results and plots about the best ``tuning_level`` for your application.


Query Selection
^^^^^^^^^^^^^^^

Once the best performing strategy and level is known through tuning, the same set of hyperparameters can be carried over to the query selection step. Here, the active learning pipeline picks the best subset of queries from the logs that can be added to the training files to give the maximum performance boost in terms of accuracy.

.. image:: /images/query_selection.png
    :align: center
    :name: query_selection_flow

The following command can be used to run query selection using the settings defined in the application configuration if the log file or the log files' pattern has been specified in the config:

.. code-block:: console

    mindmeld active_learning --select --app-path '<PATH>/app_name/' --output_folder '<PATH>'


Alternatively, path to unlabeled logs (``unlabeled_logs_path``) can be provided as a flag. 

.. code-block:: console

    mindmeld active_learning --select --app-path "<PATH>/app_name/" --output_folder '<PATH>' --unlabeled_logs_path "<PATH>/logs.txt"


Also, if your log data is labelled and included in your MindMeld application you can specify the pattern for your log data using the following flag:

.. code-block:: console 

    mindmeld active_learning --select --app-path '<PATH>/app_name/' --output_folder '<PATH>' --labeled_logs_pattern ".*log.*.txt"

Optional flags that can be used for selection include: ``batch_size``, ``log_usage_pct``, ``strategy``.


.. note::

    When selecting from labelled logs, ensure that your log pattern (``labeled_logs_pattern``) does not overlap with your train pattern (``train_pattern``).
