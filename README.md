# MindMeld Workbench 3.0

[![CircleCI](https://circleci.com/gh/expectlabs/mindmeld-workbench3.svg?style=svg&circle-token=437cf905895688ac1b58b60fe79144c180893372)](https://circleci.com/gh/expectlabs/mindmeld-workbench3)

This repository contains documentation for MindMeld Workbench 3.0.

## TLDR: Quick Start

Assuming you have pyenv installed.
```
cd mindmeld-workbench3/  # go to the root of the workbench repo
pyenv install 3.5.3  # install an appropriate version of python
pyenv virtualenv 3.5.3 workbench3  # create a virtualenv
pyenv local workbench3  # set the virtualenv for the repo
pip install -r dev-requirements.txt  # install development dependencies
```

## Getting Started

Workbench 3.x is developed primarily in Python 3.4+, but has support for Python 2.7 as well. Best practice is to set up a virtualenv using the latest stable version of Python for Workbench development. Steps below describe how to do this, assuming pyenv is installed on your system. If not see (pyenv on the eng-wiki)[https://github.com/expectlabs/eng-wiki/wiki/pyenv]. 

Make sure you're in the root of the repo
`cd mindmeld-workbench3/`

Install an appropriate version of Python, 3.5.3 or 3.6.0
`pyenv install 3.5.3`

Create virtualenv with that Python for Workbench 3
`pyenv virtualenv 3.5.3 workbench3`

Set that virtualenv to be automatically activated for the repo.
`pyenv local workbench3`

Install the development dependencies.
`pip install -r dev-requirements.txt`

## Building Docs

`make docs`

(You might see a warning about mallard.rst. Safe to ignore that for now.)

## Viewing Docs

`open docs/build/html/index.html`
