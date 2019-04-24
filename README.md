# MindMeld Conversational AI Platform

[![CircleCI](https://circleci.com/gh/expectlabs/mindmeld-workbench.svg?style=svg&circle-token=437cf905895688ac1b58b60fe79144c180893372)](https://circleci.com/gh/expectlabs/mindmeld-workbench)

This repository contains the [MindMeld Conversational AI Platform](https://www.mindmeld.com).

## Introducing MindMeld
The MindMeld Conversational AI platform is among the most advanced AI platforms for building production-quality conversational applications. It is a Python-based machine learning framework which encompasses all of the algorithms and utilities required for this purpose. Evolved over several years of building and deploying dozens of the most advanced conversational experiences achievable, MindMeld is optimized for building advanced conversational assistants which demonstrate deep understanding of a particular use case or domain while providing highly useful and versatile conversational experiences.

MindMeld is the only Conversational AI platform available today that provides tools and capabilities for every step in the workflow for a state-of-the-art conversational application. The architecture of MindMeld is illustrated below.

![MindMeld architecture](http://www.mindmeld.com/_images/architecture1.png)

To summarize, the functionality available in MindMeld includes:

  - advanced **Natural Language Processing**, consisting of

    - **Domain Classification**
    - **Intent Classification**
    - **Entity Recognition**
    - **Entity Role Labeling**
    - **Entity Resolution**
    - **Language Parsing**
  - versatile **dialogue management**
  - custom **knowledge base creation**
  - advanced **question answering**
  - **training data collection and management** support
  - **large-scale data analytics**

## The MindMeld Philosophy

MindMeld has been used for applications in dozens of different domains by some of the largest global organizations. Over the course of these production deployments, MindMeld has evolved to be ideally suited for building production-quality, large-vocabulary language understanding capabilities for any custom application domain. This has been achieved by following the architectural philosophy whose guiding principles are expressed in the table below.

|Concept|Description|
|---|---|
|**Power and Versatility**        |Unlike GUI-based tools typically too rigid to accommodate the functionality required by many applications, MindMeld provides powerful command-line utilities with the flexibility to accommodate nearly any product requirements.|
|**Algorithms and Data**          |In contrast to machine learning toolkits which offer algorithms but little data, MindMeld provides not only state-of-the-art algorithms, but also functionality which streamlines the collection and management of large sets of custom training data.|
|**NLP plus QA and DM**           |While conversational AI platforms available today typically provide natural language processing (NLP) support, few assist with question answering (QA) or dialogue management (DM). MindMeld provides end-to-end functionality including advanced NLP, QA, and DM, all three of which are required for production applications today.|
|**Knowledge-Driven Learning**    |Nearly all production conversational applications rely on a comprehensive knowlege base to enhance intelligence and utility. MindMeld is the only Conversational AI platform available today which supports custom knowledge base creation. This makes MindMeld ideally suited for applications which must demonstrate deep understanding of a large product catalog, content library, or FAQ database, for example.|
|**You Own Your Data**            |Differently from cloud-based NLP services, which require that you forfeit your data, MindMeld was designed from the start to ensure that proprietary training data and models always remain within the control and ownership of your application.|

## Quick Start

Assuming you have pip installed with Python 3.4, Python 3.5 or Python 3.6 and Elasticsearch running in the background:

```
pip install mindmeld
mindmeld blueprint home_assistant
python -m home_assistant build
python -m home_assistant converse
```

For detailed installation instructions, see [Getting Started](http://www.mindmeld.com/docs/userguide/getting_started.html). To start with pre-built sample applications, see [MindMeld Blueprints](http://www.mindmeld.com/docs/blueprints/overview.html).

## Want to learn more about MindMeld?

Visit the [MindMeld website](https://www.mindmeld.com/).

## Feedback or Support Questions

Please contact us at [mindmeld@cisco.com](mailto:mindmeld@cisco.com).
