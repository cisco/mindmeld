import pkg_resources

from mock import mock_open, patch
import pytest

from mmworkbench._version import validate_workbench_version


@pytest.mark.parametrize("file_content, version", [
    ("mmworkbench==3.3.0", "mmworkbench==3.3.0"),
    ("mo1==1.1.0\nmmworkbench==3.3.0", "mmworkbench==3.3.0"),
    ("mo1==1.1.0\nmmworkbench==3.3.0\nmo2", "mmworkbench==3.3.0"),
    ("mo1==1.1.0\nmmworkbench>=3.3.0\nmo2", "mmworkbench>=3.3.0"),
    ("mo1==1.1.0\nmmworkbench<=3.3.0\nmo2", "mmworkbench<=3.3.0"),
    ("mo1==1.1.0\nmmworkbench<=3.3.0,>=1.0.5\nmo2", "mmworkbench<=3.3.0,>=1.0.5"),
    ("mo1==1.1.0\nmmworkbench~=2.0\nmo2", "mmworkbench~=2.0"),
    ("mo1==1.1.0\nmmworkbench==rc1.0\nmo2", "mmworkbench==rc1.0"),
])
def test_validate_wb_version(mocker, file_content, version):
    with patch('mmworkbench._version.open', mock_open(read_data=file_content)):
        pkg_resources_mock = mocker.patch.object(pkg_resources, 'require', return_value=None)
        validate_workbench_version("some_path")
        assert pkg_resources_mock.call_count == 1
        assert pkg_resources_mock.call_args[0][0] == [version]


@pytest.mark.parametrize("file_content", [
    "mmworkbench",
    "mo1==1.1.0\nmmworkbench",
    "mo1==1.1.0\nmo2==2.2.0",
    "",
])
def test_validate_wb_version_no_requirement(mocker, file_content):
    with patch('mmworkbench._version.open', mock_open(read_data=file_content)):
        pkg_resources_mock = mocker.patch.object(pkg_resources, 'require', return_value=None)
        validate_workbench_version("some_path")
        assert pkg_resources_mock.call_count == 0
