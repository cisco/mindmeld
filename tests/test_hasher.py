import pytest
from mmworkbench.resource_loader import Hasher


@pytest.fixture
def hasher():
    return Hasher()


def test_hashfile_not_found(hasher):
    assert hasher.hash_file('some file name') == 'da39a3ee5e6b4b0d3255bfef95601890afd80709'
