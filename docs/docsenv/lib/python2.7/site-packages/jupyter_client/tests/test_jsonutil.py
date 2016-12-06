# coding: utf-8
"""Test suite for our JSON utilities."""

# Copyright (c) Jupyter Development Team.
# Distributed under the terms of the Modified BSD License.

import datetime
import json

import nose.tools as nt

from jupyter_client import jsonutil
from ipython_genutils.py3compat import unicode_to_str, str_to_bytes, iteritems


def test_extract_dates():
    timestamps = [
        '2013-07-03T16:34:52.249482',
        '2013-07-03T16:34:52.249482Z',
        '2013-07-03T16:34:52.249482Z-0800',
        '2013-07-03T16:34:52.249482Z+0800',
        '2013-07-03T16:34:52.249482Z+08:00',
        '2013-07-03T16:34:52.249482Z-08:00',
        '2013-07-03T16:34:52.249482-0800',
        '2013-07-03T16:34:52.249482+0800',
        '2013-07-03T16:34:52.249482+08:00',
        '2013-07-03T16:34:52.249482-08:00',
    ]
    extracted = jsonutil.extract_dates(timestamps)
    ref = extracted[0]
    for dt in extracted:
        nt.assert_true(isinstance(dt, datetime.datetime))
        nt.assert_equal(dt, ref)

def test_parse_ms_precision():
    base = '2013-07-03T16:34:52'
    digits = '1234567890'
    
    parsed = jsonutil.parse_date(base)
    nt.assert_is_instance(parsed, datetime.datetime)
    for i in range(len(digits)):
        ts = base + '.' + digits[:i]
        parsed = jsonutil.parse_date(ts)
        if i >= 1 and i <= 6:
            nt.assert_is_instance(parsed, datetime.datetime)
        else:
            nt.assert_is_instance(parsed, str)


ZERO = datetime.timedelta(0)

class tzUTC(datetime.tzinfo):
    """tzinfo object for UTC (zero offset)"""

    def utcoffset(self, d):
        return ZERO

    def dst(self, d):
        return ZERO

UTC = tzUTC()

def test_date_default():
    now = datetime.datetime.now()
    utcnow = now.replace(tzinfo=UTC)
    data = dict(now=now, utcnow=utcnow)
    jsondata = json.dumps(data, default=jsonutil.date_default)
    nt.assert_in("+00", jsondata)
    nt.assert_equal(jsondata.count("+00"), 1)
    extracted = jsonutil.extract_dates(json.loads(jsondata))
    for dt in extracted.values():
        nt.assert_is_instance(dt, datetime.datetime)

