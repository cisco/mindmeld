#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_tagging
----------------------------------

Tests for `tagging` module.
"""
# pylint: disable=locally-disabled,redefined-outer-name
from __future__ import unicode_literals

import os
import pytest

from mmworkbench import NaturalLanguageProcessor
from mmworkbench.models import tagging

APP_NAME = 'kwik_e_mart'
APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), APP_NAME)

# This index is the start index of when the time section of the full time format. For example:
# 2013-02-12T11:30:00.000-02:00, index 8 onwards slices 11:30:00.000-02:00 from the full time
# format.
MINUTE_GRAIN_INDEX = 11


class TestTagging:

    @classmethod
    def setup_class(cls):
        nlp = NaturalLanguageProcessor(APP_PATH)
        nlp.build()
        cls.nlp = nlp

    test_data_1 = [
        ('set alarm for 1130',
         ['O||O|', 'O||O|', 'O||O|', 'O||B|sys_time'],
         "2017-07-31T11:30:00.000-07:00"),
        ('the vikings fought on the year 1130',
         ['O||O|', 'O||O|', 'O||O|', 'O||O|', 'O||O|', 'O||O|', 'O||B|sys_time'],
         "1130-01-01T00:00:00.000-07:00"),
    ]

    @pytest.mark.parametrize("query,tags,expected_time", test_data_1)
    def test_get_entities_from_tags_where_tag_idx_in_sys_candidate(self,
                                                                   query,
                                                                   tags,
                                                                   expected_time):
        """Tests the behavior when the system entity tag index is
        within the system candidates spans"""

        processed_query = self.nlp.create_query(query)
        res_entity = tagging.get_entities_from_tags(processed_query, tags)

        if res_entity[0].to_dict()['value']['grain'] == 'minute':
            assert res_entity[0].to_dict()['value']['value'][MINUTE_GRAIN_INDEX:] == \
                   expected_time[MINUTE_GRAIN_INDEX:]

        if res_entity[0].to_dict()['value']['grain'] == 'year':
            assert res_entity[0].to_dict()['value']['value'] == expected_time

    test_data_2 = [
        ('set alarm for 1130',
         ['O||O|', 'O||O|', 'O||O|', 'O||O|']),
        ('the vikings fought on the year 1130',
         ['O||O|', 'O||O|', 'O||B|sys_time', 'O||O|', 'O||O|', 'O||O|', 'O||O|'])
    ]

    @pytest.mark.parametrize("query,tags", test_data_2)
    def test_get_entities_from_tags_where_tag_idx_not_in_sys_candidate(self,
                                                                       query,
                                                                       tags):
        """Tests the behavior when the system entity tag index is outside
        the system candidates spans"""

        processed_query = self.nlp.create_query(query)
        res_entity = tagging.get_entities_from_tags(processed_query, tags)
        assert res_entity == ()
