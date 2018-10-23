#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_get_app
----------------------------------

Tests for `get_app` function in `path` module.
"""
import os
import tempfile

import pytest

from mmworkbench import Application
from mmworkbench.path import get_app
from mmworkbench.exceptions import WorkbenchImportError


GOOD_APP_FILE = """
from mmworkbench import Application

app = Application(__name__)


@app.handle(default=True)
def hello(context, responder):
    responder.reply('hello')


if __name__ == '__main__':
    app.cli()

"""

BAD_APP_FILE = """
from mmworkbench import Application

"""


def dump_to_file(path, contents):
    with open(path, 'w') as target:
        target.write(contents)


app_counter = 0


@pytest.fixture
def app_dir():
    global app_counter
    with tempfile.TemporaryDirectory() as temp_dir:

        temp_app_dir = os.path.join(temp_dir, 'wb_app_' + str(app_counter))
        os.mkdir(temp_app_dir)
        yield temp_app_dir

    app_counter += 1


def test_package_app(app_dir):
    """Tests that a package app is correctly loaded"""
    # prep dir
    dump_to_file(os.path.join(app_dir, '__init__.py'), GOOD_APP_FILE)

    app = get_app(app_dir)
    assert app
    assert isinstance(app, Application)


def test_module_app(app_dir):
    """Tests that a package app is correctly loaded"""
    # prep dir
    dump_to_file(os.path.join(app_dir, 'app.py'), GOOD_APP_FILE)

    app = get_app(app_dir)
    assert app
    assert isinstance(app, Application)


def test_no_files(app_dir):
    with pytest.raises(WorkbenchImportError):
        get_app(app_dir)


def test_bad_package(app_dir):
    dump_to_file(os.path.join(app_dir, '__init__.py'), BAD_APP_FILE)

    with pytest.raises(WorkbenchImportError):
        get_app(app_dir)


def test_bad_module(app_dir):
    dump_to_file(os.path.join(app_dir, 'app.py'), BAD_APP_FILE)

    with pytest.raises(WorkbenchImportError):
        get_app(app_dir)


def test_bad_package_good_module(app_dir):
    dump_to_file(os.path.join(app_dir, '__init__.py'), "")
    dump_to_file(os.path.join(app_dir, 'app.py'), GOOD_APP_FILE)

    app = get_app(app_dir)
    assert app
    assert isinstance(app, Application)
