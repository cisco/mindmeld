# MindMeld Conversational AI Platform

[![CircleCI](https://circleci.com/gh/expectlabs/mindmeld-workbench.svg?style=svg&circle-token=437cf905895688ac1b58b60fe79144c180893372)](https://circleci.com/gh/expectlabs/mindmeld-workbench)

This repository contains the MindMeld Conversational AI Platform.

## Introducing MindMeld

The MindMeld Conversational AI platform is among the most advanced AI platforms for building production-quality conversational applications. It is a Python-based machine learning framework which encompasses all of the algorithms and utilities required for this purpose. Evolved over several years of building and deploying dozens of the most advanced conversational experiences achievable, MindMeld is optimized for building advanced conversational assistants which demonstrate deep understanding of a particular use case or domain while providing highly useful and versatile conversational experiences.

MindMeld is the only Conversational AI platform available today that provides tools and capabilities for every step in the workflow for a state-of-the-art conversational application. The architecture of MindMeld is illustrated below.

![MindMeld architecture](images/architecture1.png)

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

## TLDR: Quick Start

Assuming you have pyenv installed.
```
cd mindmeld/  # go to the root of the workbench repo
pyenv install 3.6.0  # install an appropriate version of python
pyenv virtualenv 3.6.0 mindmeld-env  # create a virtualenv
pyenv local mindmeld-env  # set the virtualenv for the repo
pip install -r dev-requirements.txt  # install development dependencies
```

## Releasing a Version to MindMeld

**Beware! Make sure you read this guide and know what you are doing!**

### Bump the Version

We use a python utility for bumping the version, aptly named `bumpversion`.

This utility will update the version in all the places it should, create a new commit `Bump version: {old-version} → {new-version}`, and create a tag for the new version.

Our versioning format is `{major}.{minor}.{patch}{release}{revision}`. So version part should be any of `major`, `minor`, `patch` `release` or `revision`.

Let's say the current version is `3.0.1dev2`. `bumpversion revision` would bump the version to `3.0.1dev3`.

If this is your first release, it is recommended that you do a dry run first to confirm you are using the correct version part:

```
bumpversion <version-part> --dry-run --verbose
```

#### TLDR; Do The Thing

```
# Bump the Version
bumpversion <version-part> --commit --tag
git push
git push --tags
```

### Building the wheel

Python wheels are the preferred binary package format for python packages. If you care to learn more, read [here](http://pythonwheels.com/) or [here](https://www.python.org/dev/peps/pep-0427/)

#### TLDR; Do The Thing

```
python setup.py bdist_wheel --universal
```
