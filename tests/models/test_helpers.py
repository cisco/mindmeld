#!/usr/bin/env python
# -*- coding: utf-8 -*-
from json.decoder import JSONDecodeError
import os
import pytest

from mindmeld.models.helpers import FileBackedList


def test_file_backed_list_data():
    # test that the file backed list saves and retrieves data properly
    test_data = [1, 2, 3, {"a": "a"}, ["1", "2", "3"], "\n", ";", "'", "á", "囧"]

    fb_list = FileBackedList()
    for data in test_data:
        fb_list.append(data)

    # test when using multiple iterators
    for original, saved, saved2, saved3 in zip(test_data, fb_list, fb_list, fb_list):
        assert original == saved
        assert original == saved2
        assert original == saved3


def test_file_backed_list_cleanup():
    test_data = ["a"] * 1000
    fb_list = FileBackedList()
    fname = fb_list.filename
    assert os.path.exists(fname)

    for data in test_data:
        fb_list.append(data)

    iterator = fb_list.__iter__()
    fb_list = None

    # verify that the iterator is still valid after the reference to fb_list is removed
    assert [next(iterator) for _ in range(len(test_data))] == test_data

    iterator = None
    # verify that the file is removed once the reference to the iterator is gone
    assert not os.path.exists(fname)


def test_file_back_list_exception():
    fb_list = FileBackedList()
    fname = fb_list.filename
    for data in ["a"] * 1000:
        fb_list.append(data)
    # flush file writer
    fb_list.file_handle.close()
    fb_list.file_handle = None
    # overwrite the file with bad json data
    with open(fname, "w") as tmp_file:
        tmp_file.write('{"this":"is almost", "valid":"json\n"')
    with pytest.raises(JSONDecodeError):
        for data in fb_list:
            pass
