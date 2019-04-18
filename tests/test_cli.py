import os

from click.testing import CliRunner
from unittest.mock import patch
from mindmeld import cli
from mindmeld.cli import num_parser, clean


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
        mocker.patch.object(cli, '_find_duckling_os_executable', return_value=None)
        runner.invoke(num_parser, ['--start'])
        mocking.assert_any_call(
            'OS is incompatible with duckling executable. Use docker to install duckling.')


def test_clean_query_cache(mocker, fake_app):
    with patch('logging.Logger.info') as mocking:
        runner = CliRunner()
        mocker.patch.object(os.path, 'exists', return_value=False)
        runner.invoke(clean, ['--query-cache'], obj={'app': fake_app})
        mocking.assert_any_call('Query cache deleted')


def test_clean_model_cache(mocker, fake_app):
    with patch('logging.Logger.warning') as mocking:
        runner = CliRunner()
        mocker.patch.object(os.path, 'exists', return_value=True)
        mocker.patch.object(os, 'listdir', return_value=['123'])
        runner.invoke(clean, ['--model-cache'], obj={'app': fake_app})
        mocking.assert_any_call('Expected timestamped folder. '
                                'Ignoring the file %s.', '123/.generated/cached_models/123')
