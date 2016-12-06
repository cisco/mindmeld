"""
Contains base test class for nbformat
"""

# Copyright (c) IPython Development Team.
# Distributed under the terms of the Modified BSD License.

import os
import unittest


class TestsBase(unittest.TestCase):
    """Base tests class."""

    def fopen(self, f, mode=u'r'):
        return open(os.path.join(self._get_files_path(), f), mode)


    def _get_files_path(self):
        return os.path.dirname(__file__)
