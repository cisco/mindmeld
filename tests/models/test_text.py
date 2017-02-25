#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_text_classifier
----------------------------------

Tests for `text_classifier` module.
"""
from __future__ import unicode_literals

from mmworkbench.models import TextModel


def test_params_grid():
    param_grid = {
        'fit_intercept': [True, False],
        'C': [0.01, 1, 100, 10000, 1000000],
        'class_bias': [1, 0.7, 0.3, 0]
    }

    configs = list(TextModel.settings_for_params_grid({}, param_grid))

    assert len(configs) == 40
    for key in param_grid.keys():
        assert key in configs[0]
