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
         ['O|', 'O|', 'O|', 'B|sys_time'],
         ["11:30:00.000-07:00", "23:30:00.000-07:00"]),
        ('the vikings fought on the year 1130',
         ['O|', 'O|', 'O|', 'O|', 'O|', 'O|', 'B|sys_time'],
         ["1130-01-01T00:00:00.000-07:00"]),
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
            assert res_entity[0].to_dict()['value']['value'][MINUTE_GRAIN_INDEX:] in \
                   set(expected_time)

        if res_entity[0].to_dict()['value']['grain'] == 'year':
            assert res_entity[0].to_dict()['value']['value'] in set(expected_time)

    test_data_2 = [
        ('set alarm for 1130',
         ['O|', 'O|', 'O|', 'O|']),
        ('the vikings fought on the year 1130',
         ['O|', 'O|', 'B|sys_time', 'O|', 'O|', 'O|', 'O|'])
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

    test_data_3 = [
        ('order 2 sandwiches',
         ['O|', 'B|sys_number', 'B|dish']),
        ('I would like sandwiches, 3 orders please',
         ['O|', 'O|', 'O|', 'B|dish', 'B|sys_number', 'O|', 'O|']),
        ('I would like eggplant parm, 3 orders please',
         ['O|', 'O|', 'O|', 'B|dish', 'I|dish', 'B|sys_number', 'O|', 'O|'])
    ]

    @pytest.mark.parametrize("query,tags", test_data_3)
    def test_get_entities_from_tags_where_entity_truncated_by_new_entity(self,
                                                                         query,
                                                                         tags):
        """Test the behavior when a new entity is directly after another entity"""

        processed_query = self.nlp.create_query(query)
        res_entity = tagging.get_entities_from_tags(processed_query, tags)
        assert len(res_entity) == 2

    test_data_4 = [
        ('order samosa, 2 naans, and daal',
         ['O|', 'B|dish', 'B|sys_number', 'B|dish', 'O|', 'B|dish'])
    ]

    @pytest.mark.parametrize("query,tags", test_data_4)
    def test_get_entities_from_tags_where_sys_entity_between_entities(self,
                                                                      query,
                                                                      tags):
        """Tests the behavior when a system entity is between two entities"""

        processed_query = self.nlp.create_query(query)
        res_entity = tagging.get_entities_from_tags(processed_query, tags)
        assert len(res_entity) == 4

    test_data_5 = [
        ('order a samosa',
         ['O|', 'O|', 'B|dish']),
        ('set alarm for 6pm',
         ['O|', 'O|', 'O|', 'B|sys_time'])
    ]

    @pytest.mark.parametrize("query,tags", test_data_5)
    def test_get_entities_from_tags_where_entities_end_with_query_end(self,
                                                                      query,
                                                                      tags):
        """Tests the behavior when the entity is at the end of a query"""

        processed_query = self.nlp.create_query(query)
        res_entity = tagging.get_entities_from_tags(processed_query, tags)
        assert len(res_entity) == 1

    test_data_6 = [
        ('order a gluten free burger and a salad',
         ['O|', 'O|', 'B|dish', 'I|dish', 'I|dish', 'O|', 'O|', 'B|dish']),
        ("set alarms for 6 o'clock and 10pm",
         ['O|', 'O|', 'O|', 'B|sys_time', 'I|sys_time', 'O|', 'B|sys_time'])
    ]

    @pytest.mark.parametrize("query,tags", test_data_6)
    def test_get_entities_from_tags_with_multi_token_entities(self,
                                                              query,
                                                              tags):
        """Tests the behavior with multi token entities"""

        processed_query = self.nlp.create_query(query)
        res_entity = tagging.get_entities_from_tags(processed_query, tags)
        assert len(res_entity) == 2
