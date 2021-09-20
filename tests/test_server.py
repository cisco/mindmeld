import json

import pytest

from mindmeld.app_manager import ApplicationManager
from mindmeld.server import MindMeldServer


@pytest.fixture
def app_manager(kwik_e_mart_app_path, kwik_e_mart_nlp):
    return ApplicationManager(kwik_e_mart_app_path, nlp=kwik_e_mart_nlp)


@pytest.fixture
def client(app_manager):
    server = MindMeldServer(app_manager)._server.test_client()
    yield server


def test_parse_endpoint(client):
    test_request = {"text": "where is the restaurant on 12th ave"}
    response = client.post(
        "/parse",
        data=json.dumps(test_request),
        content_type="application/json",
        follow_redirects=True,
    )
    assert response.status == "200 OK"
    assert (
        json.loads(response.data.decode("utf8"))["request"]["entities"][0]["value"][0][
            "cname"
        ]
        == "12th Avenue"
    )
    assert set(json.loads(response.data.decode("utf8")).keys()) == {
        "version",
        "history",
        "params",
        "form",
        "frame",
        "dialogue_state",
        "request_id",
        "response_time",
        "request",
        "directives",
        "slots",
    }


def test_parse_endpoint_fail(client):
    response = client.post("/parse")
    assert response.status == "415 UNSUPPORTED MEDIA TYPE"


def test_status_endpoint(client):
    response = client.get("/_status")
    assert response.status == "200 OK"
    assert set(json.loads(response.data.decode("utf8")).keys()) == {
        "package_version",
        "status",
        "response_time",
        "version",
    }


def test_parse_endpoint_multiple_requests(client):
    test_request = {"text": "where is the restaurant on 12th ave"}
    response = client.post(
        "/parse",
        data=json.dumps(test_request),
        content_type="application/json",
        follow_redirects=True,
    )
    assert response.status == "200 OK"
    first_response = json.loads(response.data.decode("utf8"))
    second_request = {"text": "ok thanks!", "context": {"device": "webex"}}
    for key in ["history", "params", "frame"]:
        second_request[key] = first_response[key]
    response = client.post(
        "/parse",
        data=json.dumps(second_request),
        content_type="application/json",
        follow_redirects=True,
    )
    assert response.status == "200 OK"


@pytest.mark.parametrize(
    "request_body,error_message",
    [
        (
            {
                "text": "hello",
                "random_key": {}
            },
            "Bad request {'text': 'hello', 'random_key': {}} "
            "caused error {'random_key': ['Unknown field.']}"
        ),
        (
            {
                "text": "hello",
                "params": {
                    "allowed_intents": [],
                    "dynamic_resource": {},
                    "language": "en",
                    "locale": "en_US",
                    "target_dialogue_state": "transfer_money_handler",
                    "time_zone": '',
                },
            },
            "Bad request {'text': 'hello', "
            "'params': {'allowed_intents': [], 'dynamic_resource': {}, "
            "'language': 'en', 'locale': 'en_US', 'target_dialogue_state': "
            "'transfer_money_handler', 'time_zone': ''}} caused "
            "error {'params': {'time_zone': ['Invalid time_zone param:  "
            "is not a valid time zone.']}}"
        ),
    ],
)
def test_invalid_requests(client, request_body, error_message):
    response = client.post(
        "/parse",
        data=json.dumps(request_body),
        content_type="application/json",
        follow_redirects=True,
    )
    assert response.status == "400 BAD REQUEST"
    assert response.status_code == 400
    assert json.loads(response.get_data(as_text=True))['error'] == error_message
