# MindMeld Workbench

[![CircleCI](https://circleci.com/gh/expectlabs/mindmeld-workbench.svg?style=svg&circle-token=437cf905895688ac1b58b60fe79144c180893372)](https://circleci.com/gh/expectlabs/mindmeld-workbench)

This repository contains MindMeld Workbench.  This repository was renamed to https://github.com/expectlabs/mindmeld-workbench/ from https://github.com/expectlabs/mindmeld-workbench3/ to avoid confusion in relation to the upcoming 4.0.0 release.  Please update any configuration that is still pointed to the mindmeld-workbench3 repo.

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

## Releasing a Version to MindMeld's PyPI

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


### Publishing to PyPI

You can publish to pypi by pushing the .whl to `s3://mindmeld-pypi/<develop||staging||master>/` (the trailing forward slash is important!).

```
aws s3 cp dist/mmworkbench-{new-version}-py2.py3-none-any.whl s3://mindmeld-pypi/<develop||staging||master>/
```

Currently we do not have a deployment process that takes a package from `dev` to `staging` to `master`.

Hence you should deploy to all three environments.

Usually it takes about 5 - 10 minutes to for the PyPI server to update, and you can check at `https://mindmeld.com/packages/`.
