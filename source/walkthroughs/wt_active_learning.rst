Step-by-Step Guide to Active Learning with Log Data in MindMeld
===============================================================

Active Learning is an approach to continuously and iteratively pick data that is most informative for the models to train on from a pool of unlabelled data-points. By picking such data-points, active learning provides a higher chance of improving the accuracy of the model with fewer training examples. It also greatly reduces the number of queries to be annotated by humans. MindMeld provides this inbuilt functionality to select these must-have queries from existing data or additional logs and datasets to get the best out of your conversational applications.

The following step-by-step guide will use the HR Assistant blueprint for showcasing MindMeld's active learning functionality and the different customizations involved.

Step 1: Setup a MindMeld App with Log Resources
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first step for running active learning is to set up your MindMeld app with log data configured for the app. This log data can either be in the form of labelled text files (similar to the train and test files in the app), or an unlabelled text file containing raw text queries across all domains and intents.

For the purpose of this tutorial, we will generate 'logs' for the current HR Assistant blueprint using the MindMeld Data Augmentation pipeline. This means additional queries for the app to train on. After we have figured out the best hyperparameters using the tuning step, we'll select the best qureries from the data augmentation logs (files with the patter ``.*augment.txt``. Adding these queries to the train files of the assistant should improve performance of the classifers.

Step 2: Define Active Learning Config
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This section shows the customized active learning part of the configuration file ``config.py`` for the HR Assistant app.

.. code-block:: python

    ACTIVE_LEARNING_CONFIG = {
        "output_folder": "hr_assistant_active_learning",
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
            "labeled_logs_pattern": .*augment.*.txt,
            "unlabeled_logs_path": None,
        },
    }

We will breakdown the different components of this config in their respective steps.

The ``"output_folder"`` here refers to a directory that will house all saved results from the active learning tuning and selection steps.

Step 3: Run Strategy Tuning
^^^^^^^^^^^^^^^^^^^^^^^^^^^

Step 4: Evaluate Best Hyperparameters
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Step 5: Select Best Queries
^^^^^^^^^^^^^^^^^^^^^^^^^^^

