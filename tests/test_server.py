import pytest
import json

from mindmeld.server import MindMeldServer
from mindmeld.app_manager import ApplicationManager


@pytest.fixture
def app_manager(kwik_e_mart_app_path, kwik_e_mart_nlp):
    return ApplicationManager(kwik_e_mart_app_path, nlp=kwik_e_mart_nlp)


@pytest.fixture
def client(app_manager):
    server = MindMeldServer(app_manager)._server.test_client()
    yield server


def test_parse_endpoint(client):
    test_request = {
        'text': 'where is the restaurant on 12th ave'
    }
    response = client.post('/parse', data=json.dumps(test_request),
                           content_type='application/json',
                           follow_redirects=True)
    assert response.status == '200 OK'
    assert json.loads(response.data.decode(
        'utf8'))['request']['entities'][0]['value'][0]['cname'] == '12th Avenue'
    assert set(json.loads(response.data.decode('utf8')).keys()) == {
        'version', 'history', 'params', 'frame', 'dialogue_state',
        'request_id', 'response_time', 'request', 'directives', 'slots'}


def test_parse_endpoint_fail(client):
    response = client.post('/parse')
    assert response.status == '415 UNSUPPORTED MEDIA TYPE'


def test_status_endpoint(client):
    response = client.get('/_status')
    assert response.status == '200 OK'
    assert set(json.loads(response.data.decode('utf8')).keys()) == {
        'package_version', 'status', 'response_time', 'version'}
