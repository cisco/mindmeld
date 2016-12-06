Getting Started
===============

To install MindMeld Workbench, please contact our Business Development team to obtain a technology licence. Once you obtain the licence, you can run the following command and enter your licence key.

.. code-block:: text

  pip install git+ssh://git@github.com/expectlabs/mindmeld-workbench.git

Once installed, you will be able to run through the various steps in the tutorial. The preferred python version is Python3.

.. code-block:: python

  import mindmeld as mm

Prerequisites
*************

.. _scikit-learn: http://scikit-learn.org/
.. _Machine Learning: https://www.coursera.org/learn/machine-learning

This tutorial assumes familiarity with `Machine Learning`_ and functional knowledge of the `scikit-learn`_ package. MindMeld Workbench has been optimized towards enabling language understanding on large, custom datasets (rather than pre-trained, generic models). Throughout this tutorial, we provide several code snippets that reference the following Conversational AI applications in no particular order -

* Barista (Coffee-ordering Assistant)
* Home Assistant (Smart Home, TV/Movies Discovery, Times & Dates)
* Music Assistant (Search for songs, albums, artists ...)
* Fashion Assistant (Search for clothing and fashion)


The use of MindMeld Workbench has been demonstrated on **production-grade** conversational apps in all of the above domain areas. Each comes with a medium to large-vocabulary content catalog, and Workbench is geared with robust, flexible and scalable Machine Learning modeling constructs to enable Deep-Domain language understanding.

Welcome to the MindMeld Workbench
=================================

The MindMeld Workbench python package provides a set of Natural Language Processing tools that can be used to build Deep Domain Conversational AI systems, capable of intelligently conversing with users in natural language. The package enables machine learning engineers and data scientists to quickly build, test and deploy conversational user interfaces across a wide range of platforms.

Along with easy-to-use modules for building an intelligent conversational interface, we also provide detailed tutorials explaining the step-by-step process and best practices for building such an application. Our user guide will walk you through the various machine learning components and how to train, test and optimize each of them to achieve the high level of accuracy required for user-facing commercial applications.

Our user guide is organized into three sections:

.. toctree::
   :maxdepth: 2
   :caption: Learn
   :name: learn

   anatomy

.. toctree::
   :maxdepth: 2
   :caption: Build
   :name: build

   training_data
   kb
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
