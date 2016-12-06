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