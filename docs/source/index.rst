Getting Started
===============

1. Obtain a technology license from MindMeld

2. Install MindMeld Workbench — you will be prompted to enter your license key

.. code-block:: text

  pip install git+ssh://git@github.com/expectlabs/mindmeld-workbench.git

3. When writing Python code, use ``import mindmeld`` to import Workbench

.. code-block:: python

  import mindmeld as mm

.. note:: Workbench supports Python 3.5.x.

Welcome to MindMeld Workbench
=============================

MindMeld Workbench

* is a set of Natural Language Processing tools for building Deep Domain Conversational AI systems that intelligently converse with users in natural language
* is optimized for language understanding based on large, custom datasets (rather than pre-trained, generic models)
* supports robust, flexible and scalable Machine Learning modeling constructs to enable Deep-Domain language understanding
* has been demonstrated on **production-grade** conversational apps in several domains (described below)
* enables machine learning engineers and data scientists to quickly build, test and deploy conversational user interfaces across a wide range of platforms

About this Tutorial
*******************

In this tutorial, you will learn

* the role of each component in the Workbench architecture
* how to train, test, and optimize each component to achieve the high accuracy required for user-facing commercial applications
* how to build an intelligent conversational interface with easy-to-use modules

In short, you will learn how to build Conversational AI applications step by step, while gaining insight into best practices.

The code samples in this tutorial reference four applications, each featuring a medium- to large-vocabulary content catalog:

===========  =========================================================================
Application  Use Case
===========  =========================================================================
Barista  Order drinks and food at a coffee shop
Home Assistant  Operate a Smart Home, discover TV shows and movies, navigate TV Guide
Music Assistant  Discover and save favorite songs, albums, artists, playlists
Fashion Assistant  Shop for clothing and accessories
===========  =========================================================================

This tutorial assumes familiarity with `Machine Learning`_ and the Python programming language, including functional knowledge of the `scikit-learn`_ package.

.. _scikit-learn: http://scikit-learn.org/
.. _Machine Learning: https://www.coursera.org/learn/machine-learning


Contents
========

.. toctree::
   :maxdepth: 2
   :caption: Learn
   :name: learn

   anatomy

.. toctree::
   :maxdepth: 2
   :caption: Data
   :name: data

   training_data
   kb

.. toctree::
   :maxdepth: 2
   :caption: Build
   :name: build

   entity_map
   domain_classification
   intent_classification
   entity_recognition
   role_classification
   entity_resolution
   semantic_parsing
   dialogue_manager
   question_answering
   nlg

.. toctree::
   :maxdepth: 2
   :caption: Deploy
   :name: deploy

   deployment

.. toctree::
   :maxdepth: 2
   :caption: Notebook
   :name: notebook

   hello-world
