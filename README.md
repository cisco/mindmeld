# MindMeld Workbench 2.0

This repository contains documentation for MindMeld Workbench 2.0.

## Getting Started

We will be using Python3. Make sure you are not in a Python2 virtualenv. Run these commands in sequence -

`pip install -r requirements.txt`

`brew install pandoc`

`python3 -m pip install --upgrade ipykernel`

`python3 -m ipykernel install --user`

## Building

`cd docs`

`make html`

## Running

`open build/html/index.html`
