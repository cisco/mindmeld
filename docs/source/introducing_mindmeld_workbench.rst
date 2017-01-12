Introducing MindMeld Workbench 
==============================

Large-scale supervised learning, when applied effectively, can be used to build very useful and versatile conversational applications. Unfortunately, the vast majority of attempts fail. Over the past few years, most companies that have attempted to build conversational applications have been unsuccessful in creating reliable or useful experiences. This woeful track record can no doubt be traced to the dearth of tools and best practices available to guide companies down the path toward success. This guide and the MindMeld platform were created to enable every company with the tools and techniques required to build highly useful and versatile conversational applications. The MindMeld platform is one of the most advanced AI platforms available today for building and deploying production-quality conversational experiences. This guide provides detailed instructions, best practices and reference applications which will enable any organization to create some of the most advanced conversational experiences possible today. 

MindMeld Workbench is the core machine learning toolkit which powers the MindMeld platform. It is a Python-based framework which encompasses all of the algorithms and utilities required to build modern, production-quality conversational applications. It has evolved over several years during the process of building and deploying dozens of the most advanced conversation experiences achievable today. It has been optimized for the needs of companies that want to build an advanced conversational assistant which demonstrates deep understanding of a particular use case or domain. MindMeld Workbench is a commercial software product which requires a license to use. To inquire about obtaining a license for MindMeld Workbench, please `contact MindMeld sales <mailto:info@mindmeld.com>`_.

MindMeld Workbench is the only Conversational AI platform available today which provides capabilities to handle every step in the workflow for a state-of-the-art conversational application. The architecture of MindMeld Workbench is illustrated below.

.. image:: images/architecture1.png

The functionality available in Workbench includes

  - advanced natural language processing
  
    - domain classification
    - intent classification
    - entity recognition
    - entity role labeling
    - entity resolution
    - language parsing
  - versatile dialogue management
  - custom knowledge base creation
  - advanced question answering
  - training data collection and management support
  - large-scale data analytics

This guide describes the capabilities of MindMeld Workbench in detail and leverages this toolkit to illustrate the standard approaches and best practices for creating production-quality conversational applications. 


The MindMeld Workbench Philosophy
---------------------------------
MindMeld Workbench is a Conversational AI platform that was specifically designed to meet the needs of enterprises that want to build and deploy production-quality conversational applications. To date, it has been used for applications in dozens of different domains by some of the largest global organizations. Over the course of these many different production deployments, MindMeld Workbench has evolved to be ideally suited for building production-quality, large-vocabulary language understanding capabilities for any custom application domain. As such, the architectural philosophy underpinning the Workbench platform has aligned around a set of guiding principles:

===============================  =====
**Power and Versatility**        Unlike GUI-based tools which are often too rigid to accommodate the functionality required by many applications, Workbench provides a collection of powerful command-line utilities which have the flexibility to accommodate nearly any product requirements.
**Algorithms and Data**          Unlike most machine learning toolkits which offer algorithms but little data, Workbench provides not only state-of-the-art algorithms, but it also streamlines the collection and management of large sets of custom training data.
**NLP plus QA and DM**           Conversational AI platforms available today typically provide natural language processing (NLP) support, but few assist with question answering (QA) or dialogue management (DM). Production applications today require all three of these components. MindMeld Workbench is designed to provide end-to-end functionality including advanced NLP as well as QA and DM.
**Knowledge-Driven Learning**    Nearly all production conversational applications rely on a comprehensive knowlege base to enhance intelligence and utility. MindMeld Workbench is the only Conversational AI platform available today which supports custom knowledge base creation. As a result, it is ideally suited for applications which must demonstrate deep understanding of a large product catalog, content library or FAQ database, for example.
**You Own Your Data**            Unlike cloud-based NLP services which require that you forfeit your data to the cloud, MindMeld Workbench was designed from the start to ensure that proprietary training data and models always remain within the control and ownership of your application.
===============================  =====  


About This Guide
----------------

This guide details the functionality offered by MindMeld workbench and illustrates common techniques and best practices for building state-of-the-art production applications. This guide assumes familiarity with `Machine Learning <https://www.coursera.org/learn/machine-learning>`_ and the `Python <https://www.python.org/>`_ programming language, including functional knowledge of the `scikit-learn <http://scikit-learn.org/>`_ Python package.

The :ref:`intro` provides an overview of the state-of-the-art in building conversational applications and discusses some of the tradeoffs associated with different implementation approaches. The :ref:`quickstart` section illustrates how to build a simple end-to-end conversational application using MindMeld Workbench. The :ref:`userguide` covers MindMeld Workbench in depth. In the process, it highlights many of the common techniques and best practices that are used to build production-quality conversational experiences today.

