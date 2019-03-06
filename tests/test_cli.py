from click.testing import CliRunner
from unittest.mock import patch
from mmworkbench import cli
from mmworkbench.cli import num_parser


def test_num_parse_already_running(mocker):
    runner = CliRunner()
    with patch('logging.Logger.info') as mocking:
        mocker.patch.object(cli, '_get_duckling_pid', return_value=[123])
        runner.invoke(num_parser, ['--start'])
        mocking.assert_any_call('Numerical parser running, PID %s', 123)


def test_num_parse_not_running(mocker):
    runner = CliRunner()
    with patch('logging.Logger.error') as mocking:
        mocker.patch.object(cli, '_get_duckling_pid', return_value=None)
        mocker.patch.object(cli, 'find_duckling_os_executable', return_value=None)
        runner.invoke(num_parser, ['--start'])
        mocking.assert_any_call(
            'OS is incompatible with duckling executable. Use docker to install duckling.')
