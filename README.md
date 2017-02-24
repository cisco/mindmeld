# MindMeld Workbench 3.0

[![CircleCI](https://circleci.com/gh/expectlabs/mindmeld-workbench3.svg?style=svg&circle-token=437cf905895688ac1b58b60fe79144c180893372)](https://circleci.com/gh/expectlabs/mindmeld-workbench3)

This repository contains documentation for MindMeld Workbench 3.0.

## Getting Started

Workbench 3.0 is developed primarily in Python 3.3+, but has support for Python 2.7 as well. Set up a virtualenv using the latest stable version of Python for Workbench development


`pip install -r dev-requirements.txt`

`brew install pandoc`

`pip install --upgrade ipykernel`

`python -m ipykernel install --user`

## Building

`make docs`

(You might see a warning about mallard.rst. Safe to ignore that for now.)

## Running

`open docs/build/html/index.html`
