import pkg_resources

from mock import mock_open, patch
import pytest

from mindmeld._version import validate_mindmeld_version


@pytest.mark.parametrize("file_content, version", [
    ("mindmeld==3.3.0", "mindmeld==3.3.0"),
    ("mo1==1.1.0\nmindmeld==3.3.0", "mindmeld==3.3.0"),
    ("mo1==1.1.0\nmindmeld==3.3.0\nmo2", "mindmeld==3.3.0"),
    ("mo1==1.1.0\nmindmeld>=3.3.0\nmo2", "mindmeld>=3.3.0"),
    ("mo1==1.1.0\nmindmeld<=3.3.0\nmo2", "mindmeld<=3.3.0"),
    ("mo1==1.1.0\nmindmeld<=3.3.0,>=1.0.5\nmo2", "mindmeld<=3.3.0,>=1.0.5"),
    ("mo1==1.1.0\nmindmeld~=2.0\nmo2", "mindmeld~=2.0"),
    ("mo1==1.1.0\nmindmeld==rc1.0\nmo2", "mindmeld==rc1.0"),
])
def test_validate_mm_version(mocker, file_content, version):
    with patch('mindmeld._version.open', mock_open(read_data=file_content)):
        pkg_resources_mock = mocker.patch.object(pkg_resources, 'require', return_value=None)
        validate_mindmeld_version("some_path")
        assert pkg_resources_mock.call_count == 1
        assert pkg_resources_mock.call_args[0][0] == [version]


@pytest.mark.parametrize("file_content", [
    "mindmeld",
    "mo1==1.1.0\nmindmeld",
    "mo1==1.1.0\nmo2==2.2.0",
    "",
])
def test_validate_mm_version_no_requirement(mocker, file_content):
    with patch('mindmeld._version.open', mock_open(read_data=file_content)):
        pkg_resources_mock = mocker.patch.object(pkg_resources, 'require', return_value=None)
        validate_mindmeld_version("some_path")
        assert pkg_resources_mock.call_count == 0
