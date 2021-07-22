#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test PlotManager
----------------------------------

Test for PlotManager in the `active_learning.plot_manager` module.
"""
import pytest
from mindmeld.active_learning.plot_manager import PlotManager

data_dict = {"a": {"b": {"c": 2021}}}


@pytest.mark.extras
@pytest.mark.parametrize(
    "selected_keys, expected_value",
    [
        (["a"], {"b": {"c": 2021}}),
        (["a", "b"], {"c": 2021}),
        (["a", "b", "c"], 2021),
    ],
)
def test_get_nested(selected_keys, expected_value):
    assert PlotManager.get_nested(data_dict, selected_keys) == expected_value
