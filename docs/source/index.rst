
The Conversational AI Playbook
==============================

Conversational applications have been the subject of AI research for many decades. Only in the past few years, however, have they have gone mainstream. Today, billions of people around the world rely on products like Siri, Alexa, Google Assistant, and Cortana every week. Experts agree that this is just the beginning. Conversational interfaces represent the new frontier of application development, and over the next decade, we will see a wide variety of conversational assistants emerge to help us with many daily tasks.

This is not the first tectonic shift in application development. In the mid 1990s, the arrival of the Web saw traditional packaged software replaced by a new generation of browser-based, web applications. Similarly, with the arrival of the iPhone app store in 2008, native mobile applications supplanted web applications as the predominant application paradigm. Conversational applications are now ushering in a third major transformation in development practices, but this one looks to be even more disruptive than its predecessors.

For nearly three decades, application design has centered around the Graphical User Interface, or GUI. During this time, a generation of developers became well-versed in a set of tools, design patterns, and best practices which streamline the building of professional, GUI-based applications. With conversational experiences, all of this changes, and the GUI becomes de-emphasized or absent altogether. As a result, the previous generation of tools and best practices no longer applies. Furthermore, unlike their predecessors, conversational applications rely heavily on AI technology to understand and respond to human language. Prior to now, mastering AI was a never a prerequisite for good application design.

Today, faced with these unfamiliar challenges, developers and companies struggle to create reliable and useful conversational applications. In fact, the vast majority of attempts have failed. This woeful track record can no doubt be traced to the dearth of tools and best practices companies could use to find the path to success. A new playbook and a new generation of tools is desperately needed to help organizations chart a fruitful course in the new frontier of conversational application design.


About This Guide
----------------

This guide represents a first step toward defining the governing principles and best practices which will enable developers to build great conversational applications. It is the result of several years of practical experience building and deploying dozens of the most advanced conversational applications achievable. This guide does not survey cutting-edge research and state-of-the-art algorithms; there are many other helpful resources already available for that purpose. Instead, it serves as a comprehensive playbook to help developers and data scientists build real production applications. The detailed instructions, practical advice, and real-world examples provided here should empower developers to improve the quality and variety of conversational experiences of the coming months and years.

This guide assumes familiarity with `Machine Learning <https://www.coursera.org/learn/machine-learning>`_ and the `Python <https://www.python.org/>`_ programming language, including functional knowledge of the `scikit-learn <http://scikit-learn.org/>`_ python package. The :ref:`intro` provides an overview of the state-of-the-art in building conversational applications and discusses tradeoffs associated with different implementation approaches. The :ref:`quickstart` illustrates how to build a simple end-to-end conversational application using Workbench, the MindMeld Conversational AI toolkit. The :ref:`userguide` covers conversational applications and MindMeld Workbench in depth. In the process, it highlights many of the common techniques and best practices that are used to build production-quality conversational experiences today.



Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Introduction
   :name: intro

   intro/introduction_to_conversational_applications
   intro/approaches_for_building_conversational_applications
   intro/anatomy_of_a_conversational_ai_interaction
   intro/introducing_mindmeld_workbench


.. toctree::
   :maxdepth: 2
   :caption: Step-by-Step Guide
   :name: quickstart

   quickstart/overview
   quickstart/select_the_right_use_case
   quickstart/script_interactions
   quickstart/define_the_hierarchy
   quickstart/define_the_dialogue_handlers
   quickstart/create_the_knowledge_base
   quickstart/generate_representative_training_data
   quickstart/train_the_natural_language_processing_classifiers
   quickstart/configure_the_language_parser
   quickstart/optimize_question_answering_performance
   quickstart/deploy_to_production


.. toctree::
   :maxdepth: 2
   :caption: Workbench User Manual
   :name: userguide

   userguide/coming_soon
   userguide/getting_started
   userguide/architecture
   userguide/key_concepts
   userguide/directory_structure
   userguide/interface
   userguide/dialogue_manager
   userguide/kb
   userguide/domain_classification
   userguide/intent_classification
   userguide/entity_recognition
   userguide/role_classification
   userguide/entity_resolution
   userguide/language_parsing
   userguide/deployment


.. toctree::
   :maxdepth: 2
   :caption: Internal Documentation
   :name: internal

   internal/api_reference
   internal/history
