
The Conversational AI Playbook
==============================

Conversational applications have been the subject of AI research for many decades. It has only been in the past few years, however, that they have gone mainstream. Today, billions of people around the world rely on products like Siri, Alexa, Google Assistant and Cortana every week. Experts agree that this is just the beginning. Conversational interfaces represent the new frontier of application development, and over the next decade, we will see a wide variety of converational assistants emerge to help us with many daily tasks.

This is not the first tectonic shift in application development. In the mid 1990s, the arrival of the Web saw traditional packaged software replaced by a new generation of browser-based, web applications. Similarly, with the arrival of the iPhone app store in 2008, native mobile applications supplanted web applications as the predominant application paradigm. Conversational applications are now ushering in a third major transformation in development practices, but this one looks to be even more disruptive than its predecessors.

For nearly three decades, application design has centered around the Graphical User Interface, or GUI. Over this time, a generation of developers has become well-versed in a set of tools, design patterns and best practices which streamline the building of professional, GUI-based applications. With conversational experiences, all of this changes, and the GUI becomes de-emphasized or absent altogether. As a result, the previous generation of tools and best practices no longer applies. Furthermore, unlike their predecessors, conversational applications rely heavily on AI technology in order to understand and respond to human language. Prior to now, mastering AI was a never a prerequisite for good application design. 

Faced with these unfamiliar challenges, developers and companies struggle today to create reliable and useful conversational applications. In fact, over the past few years, the vast majority of attempts have failed. This woeful track record can no doubt be traced to the dearth of tools and best practices available to guide companies down the path toward success. A new playbook and new generation of tools is desperately needed to help organizations chart a fruitful course in this new frontier of conversational application design.


About This Guide
----------------

This guide represents a first step toward defining the governing principles and best practices which will enable developers to build great conversational applications. It is the result of several years of practical experience building and deploying dozens of the most advanced conversational applications achievable today. This guide is not intended to survey cutting-edge research and state-of-the-art algorthims; there are many other helpful resources already available for that topic. Instead, this guide serves as a comprehensive playbook to help developers and data scientists build real production applications. It provides detailed instructions, practical advice, and real-world examples. We hope that this material will empower developers with the tools and know-how to improve the quality and variety of conversational experiences which become available in the coming months and years.

This guide assumes familiarity with `Machine Learning <https://www.coursera.org/learn/machine-learning>`_ and the `python <https://www.python.org/>`_ programming language, including functional knowledge of the `scikit-learn <http://scikit-learn.org/>`_ python package. The :ref:`intro` provides an overview of the state-of-the-art in building conversational applications and discusses some of the tradeoffs associated with different implementation approaches. The :ref:`quickstart` illustrates how to build a simple end-to-end conversational application using the MindMeld Conversational AI toolkit, called Workbench. The :ref:`userguide` covers conversational applications and MindMeld Workbench in depth. In the process, it highlights many of the common techniques and best practices that are used to build production-quality conversational experiences today.



Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Introduction
   :name: intro

   introduction_to_conversational_applications
   approaches_for_building_conversational_applications
   anatomy_of_a_conversational_ai_interaction
   introducing_mindmeld_workbench

.. toctree::
   :maxdepth: 2
   :caption: Step-by-Step Guide
   :name: quickstart

   overview
   select_the_right_use_case
   script_interactions
   define_the_hierarchy
   define_the_dialogue_handlers
   create_the_knowledge_base
   generate_representative_training_data
   train_the_natural_language_processing_classifiers
   configure_the_language_parser
   optimize_question_answering_performance
   deploy_to_production

.. toctree::
   :maxdepth: 2
   :caption: Workbench User Manual
   :name: userguide
   
   coming_soon
   
..
   getting_started
   architecture
   key_concepts
   directory_structure
   interface
   dialogue_manager
   kb
   domain_classification
   intent_classification
   entity_recognition
   role_classification
   entity_resolution
   language_parsing
   deployment

.. toctree::
   :maxdepth: 2
   :caption: Other
   :name: other

   history

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
