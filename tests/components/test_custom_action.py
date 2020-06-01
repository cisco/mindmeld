#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_custom_action
----------------------------------

Tests for custom actions.
"""


def test_custom_action_config(kwik_e_mart_app):
    assert kwik_e_mart_app.custom_action_config is not None
    assert "url" in kwik_e_mart_app.custom_action_config
    assert kwik_e_mart_app.custom_action_config["url"] == "http://0.0.0.0:8080/"
