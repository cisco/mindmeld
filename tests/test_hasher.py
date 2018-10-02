import hashlib
from mock import mock_open, patch

import pytest

from mmworkbench.resource_loader import Hasher


@pytest.fixture
def hasher():
    return Hasher()


def test_hashfile_not_found(hasher):
    assert hasher.hash_file('some file name') == 'da39a3ee5e6b4b0d3255bfef95601890afd80709'


@pytest.mark.parametrize(
    'file_content',
    ['hello world', '', 'hello world\nwe are here'])
def test_hashfile(hasher, file_content):
    file_content = file_content.encode('utf-8')
    with patch('mmworkbench.resource_loader.open', mock_open(read_data=file_content)):
        hash_obj = hashlib.sha1()
        hash_obj.update(file_content)
        assert hasher.hash_file('some file name') == hash_obj.hexdigest()


def test_hashfile_with_large_file(hasher, aeneid_path, aeneid_content):
    hash_obj = hashlib.sha1()
    hash_obj.update(aeneid_content.encode('utf-8'))
    assert hasher.hash_file(aeneid_path) == hash_obj.hexdigest()
