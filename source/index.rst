
About This Playbook
-------------------

This playbook represents a first step toward defining the governing principles and best practices which will enable developers to build great conversational applications. It is the result of several years of practical experience building and deploying dozens of the most advanced conversational applications achievable. Cutting-edge research and state-of-the-art algorithms are not surveyed here; there are many other resources available for that purpose. Instead, this playbook focuses on helping developers and data scientists build real production applications. The detailed instructions, practical advice, and real-world examples provided here should empower developers to improve the quality and variety of conversational experiences of the coming months and years.

The :ref:`intro` provides an overview of the state of the art in building conversational applications and discusses tradeoffs associated with different implementation approaches.

The :ref:`quickstart` illustrates how to build an end-to-end conversational application using MindMeld, the MindMeld Conversational AI toolkit. You can apply it directly while building a conversational application, or just read it to learn the essentials of MindMeld.

The :ref:`blueprints` section explains how to use MindMeld to quickly build and test a fully working conversational application without writing code or collecting training data. You can bootstrap a Blueprint app into a more specialized and powerful conversational application, or just use the Blueprint app as a demo or research tool.

The :ref:`userguide` covers each component of the MindMeld platform in depth and explains how to use MindMeld to build and configure each component for optimal performance. It functions as a technical supplement that details how to improve an application built following either the :ref:`quickstart` or the :ref:`blueprints`. The :ref:`userguide` highlights many common techniques and best practices that contribute to successful conversational experiences today.

The :ref:`versions` section contains a Recent Changes section which you should always consult before installing or upgrading MindMeld.

Readers should be familiar with `Machine Learning <https://www.coursera.org/learn/machine-learning>`_ and the `Python <https://www.python.org/>`_ programming language, including functional knowledge of the `scikit-learn <http://scikit-learn.org/>`_ python package.

For those who wish to use the deep learning models included in MindMeld, familiarity with the `TensorFlow <https://www.tensorflow.org/>`_ library is recommended.

How to Use these Docs
---------------------

The questions and answers below explain how to use this Conversational AI Playbook to support your individual needs as a developer and learner.

----

**Q:** What is Conversational AI? Why does MindMeld exist and what does it do?

**A:** Read the :ref:`intro`. This explains what MindMeld does differently from other platforms and toolkits.

----

**Q:** What is the methodology for building Conversational AI applications with MindMeld?

**A:** Read the :ref:`quickstart`. This and the :ref:`userguide` together form the main body of knowledge for MindMeld developers.

----

**Q:** How do I quickly get a pre-built MindMeld app running, either as a learning exercise or as a baseline for my own application?

**A:** Follow the instructions in :ref:`blueprints`.

----

**Q:** How do I apply the Step-by-Step methodology to my own MindMeld app, either building from scratch or extending a Blueprint app?

**A:** Work through either the :ref:`quickstart` or the section of :ref:`blueprints` that’s appropriate for your use case. Where those docs refer you to the :ref:`userguide`, apply the in-depth material in the :ref:`userguide` to improve your application’s performance or increase its sophistication, as needed.

The :ref:`userguide` and the :ref:`quickstart` together form the main body of knowledge for MindMeld developers.

----


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Introduction
   :name: intro

   intro/introduction_to_conversational_applications
   intro/approaches_for_building_conversational_applications
   intro/anatomy_of_a_conversational_ai_interaction
   intro/introducing_mindmeld
   intro/key_concepts


.. toctree::
   :maxdepth: 2
   :caption: Step-by-Step Guide
   :name: quickstart

   quickstart/00_overview
   quickstart/01_select_the_right_use_case
   quickstart/02_script_interactions
   quickstart/03_define_the_hierarchy
   quickstart/04_define_the_dialogue_handlers
   quickstart/05_create_the_knowledge_base
   quickstart/06_generate_representative_training_data
   quickstart/07_train_the_natural_language_processing_classifiers
   quickstart/08_configure_the_language_parser
   quickstart/09_optimize_question_answering_performance
   quickstart/10_deploy_to_production


.. toctree::
   :maxdepth: 2
   :caption: Blueprint Applications
   :name: blueprints

   blueprints/overview
   blueprints/food_ordering
   blueprints/video_discovery
   blueprints/home_assistant


.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :name: userguide

   userguide/getting_started
   userguide/architecture
   userguide/preprocessor
   userguide/nlp
   userguide/domain_classifier
   userguide/intent_classifier
   userguide/entity_recognizer
   userguide/lstm
   userguide/role_classifier
   userguide/entity_resolver
   userguide/parser
   userguide/custom_features
   userguide/kb
   userguide/dm
   userguide/voice
   integrations/webex_teams


.. toctree::
   :maxdepth: 2
   :caption: API
   :name: internal

   internal/api_reference


.. toctree::
   :maxdepth: 2
   :caption: Versions
   :name: versions

   versions/changes
   versions/history
