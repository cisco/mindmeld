#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_ser
----------------------------------

Tests for `system_entity_recognizer` module
"""
import pytest
import requests

from mindmeld.system_entity_recognizer import SystemEntityRecognizer, DucklingRecognizer


NOW_TIMESTAMP = 1544706000000
SECONDS_IN_HOUR = 3600
SECONDS_IN_MINUTE = 60
DUCKLING_URL = "http://localhost:7151/parse"


@pytest.mark.parametrize(
    "query, predicted_texts, predicted_values",
    [
        ("is this room open for an hour", ["an hour"], [SECONDS_IN_HOUR]),
        (
            "is the room available for the next 5 and a half hours",
            ["5 and a half hours"],
            [SECONDS_IN_HOUR * 5.5],
        ),
        (
            "does anyone have this room for 15 minutes",
            ["15 minutes"],
            [SECONDS_IN_MINUTE * 15],
        ),
        (
            "is anyone using this room for the next 6 hours",
            ["6 hours"],
            [SECONDS_IN_HOUR * 6],
        ),
    ],
)
def test_duration(query, predicted_texts, predicted_values):
    data = {"text": query, "reftime": NOW_TIMESTAMP, "latent": True}

    res = requests.post(DUCKLING_URL, data=data)
    responses = res.json()
    response_texts = [r["body"] for r in responses]
    response_values_normalized_seconds = [
        r["value"]["normalized"]["value"]
        for r in responses
        if r["value"].get("normalized")
    ]

    for p in predicted_texts:
        assert p in response_texts

    for p in predicted_values:
        assert p in response_values_normalized_seconds


@pytest.mark.parametrize(
    "query, predicted_texts, predicted_from, predicted_to",
    [
        (
            "what will the weather be like tonight",
            ["tonight"],
            ["2018-12-13T18:00:00.000-08:00"],
            ["2018-12-14T00:00:00.000-08:00"],
        ),
        (
            "is this room available tomorrow afternoon",
            ["tomorrow afternoon"],
            ["2018-12-14T12:00:00.000-08:00"],
            ["2018-12-14T19:00:00.000-08:00"],
        ),
        (
            "set alarm for this morning",
            ["this morning"],
            ["2018-12-13T04:00:00.000-08:00"],
            ["2018-12-13T12:00:00.000-08:00"],
        ),
        (
            "please turn on my thursday evening alarm",
            ["thursday evening"],
            ["2018-12-13T18:00:00.000-08:00"],
            ["2018-12-14T00:00:00.000-08:00"],
        ),
    ],
)
def test_interval(query, predicted_texts, predicted_from, predicted_to):
    data = {"text": query, "reftime": NOW_TIMESTAMP, "latent": True}

    res = requests.post(DUCKLING_URL, data=data)
    responses = res.json()
    response_texts = [r["body"] for r in responses]
    response_from_value = [
        r["value"]["from"]["value"] for r in responses if r["value"].get("from")
    ]
    response_to_value = [
        r["value"]["to"]["value"] for r in responses if r["value"].get("to")
    ]

    for p in predicted_texts:
        assert p in response_texts
    for p in predicted_from:
        assert p in response_from_value
    for p in predicted_to:
        assert p in response_to_value


@pytest.mark.parametrize(
    "query, predicted_texts, predicted_values",
    [
        ("option 1", ["1"], [1]),
        ("call 10", ["10"], [10]),
        ("go to page 3", ["3"], [3]),
        ("number four", ["four"], [4]),
        ("set volume to 80 percent", ["80"], [80]),
        ("turn down volume by 2", ["2"], [2]),
    ],
)
def test_number(query, predicted_texts, predicted_values):
    data = {"text": query, "reftime": NOW_TIMESTAMP, "latent": True}

    res = requests.post(DUCKLING_URL, data=data)
    responses = res.json()
    response_texts = [r["body"] for r in responses if r["dim"] == "number"]
    response_values = [
        r["value"]["value"]
        for r in responses
        if r["value"].get("value") and r["dim"] == "number"
    ]

    for p in predicted_texts:
        assert p in response_texts
    for p in predicted_values:
        assert p in response_values


@pytest.mark.parametrize(
    "query, predicted_texts, predicted_values",
    [
        ("fifth", ["fifth"], [5]),
        ("call the eighth contact", ["eighth"], [8]),
        ("second option", ["second"], [2]),
        ("call the 2nd contact", ["2nd"], [2]),
    ],
)
def test_ordinal(query, predicted_texts, predicted_values):
    data = {"text": query, "reftime": NOW_TIMESTAMP, "latent": True}

    res = requests.post(DUCKLING_URL, data=data)
    responses = res.json()
    response_texts = [r["body"] for r in responses]
    response_values = [
        r["value"]["value"] for r in responses if r["value"].get("value")
    ]

    for p in predicted_texts:
        assert p in response_texts
    for p in predicted_values:
        assert p in response_values


@pytest.mark.parametrize(
    "query, predicted_texts, predicted_values",
    [
        ("place a call to +1649198338", ["+1649198338"], ["(+16) 49198338"]),
        ("please call 4388995284", ["4388995284"], ["4388995284"]),
        ("dial 772 771 21 14", ["772 771 21 14"], ["7727712114"]),
        (
            "i'd like to call phone number 39-387-84-596",
            ["39-387-84-596"],
            ["3938784596"],
        ),
        ("place a call to 8619244983", ["8619244983"], ["8619244983"]),
    ],
)
def test_phone_number(query, predicted_texts, predicted_values):
    data = {"text": query, "reftime": NOW_TIMESTAMP, "latent": True}

    res = requests.post(DUCKLING_URL, data=data)
    responses = res.json()
    response_texts = [r["body"] for r in responses]
    response_values = [
        r["value"]["value"] for r in responses if r["value"].get("value")
    ]

    for p in predicted_texts:
        assert p in response_texts
    for p in predicted_values:
        assert p in response_values


@pytest.mark.parametrize(
    "query, predicted_texts, predicted_values",
    [
        ("make it 5 degrees cooler in the bedroom", ["5"], [5]),
        ("turn it down a few degrees", ["a few degrees"], [3]),
        ("please lower temperature by 10", ["10"], [10]),
        ("increase the temperature by 3 degrees", ["3 degrees"], [3]),
        ("set thermostat to 65", ["65"], [65]),
    ],
)
def test_temperature(query, predicted_texts, predicted_values):
    data = {"text": query, "reftime": NOW_TIMESTAMP, "latent": True}

    res = requests.post(DUCKLING_URL, data=data)
    responses = res.json()
    response_texts = [r["body"] for r in responses if r["dim"] == "temperature"]
    response_values = [
        r["value"]["value"]
        for r in responses
        if r["value"].get("value") and r["dim"] == "temperature"
    ]

    for p in predicted_texts:
        assert p in response_texts
    for p in predicted_values:
        assert p in response_values


@pytest.mark.parametrize(
    "query, predicted_texts, predicted_values",
    [
        (
            "does anyone have the room at 3 pm",
            ["3 pm"],
            ["2018-12-13T15:00:00.000-08:00"],
        ),
        (
            "is this conference room bookable from 10 to 11",
            ["10", "11"],
            ["2018-12-13T10:00:00.000-08:00", "2018-12-13T11:00:00.000-08:00"],
        ),
        ("is this room reserved at noon", ["noon"], ["2018-12-13T12:00:00.000-08:00"]),
        (
            "does anyone have this room booked today for 7:06 am",
            ["7:06 am"],
            ["2018-12-13T07:06:00.000-08:00"],
        ),
        (
            "Launch the online meeting at 5 p.m.",
            ["5 p.m."],
            ["2018-12-13T17:00:00.000-08:00"],
        ),
        ("start the 10:29 meeting", ["10:29"], ["2018-12-13T10:29:00.000-08:00"]),
        (
            "what is the forecast for right now",
            ["right now"],
            ["2018-12-13T05:00:00.000-08:00"],
        ),
        (
            "Start the video meeting at 10 o'clock",
            ["10 o'clock"],
            ["2018-12-13T10:00:00.000-08:00"],
        ),
        ("Set an alarm for 615", ["615"], ["2018-12-13T06:15:00.000-08:00"]),
    ],
)
def test_time(query, predicted_texts, predicted_values):
    data = {"text": query, "reftime": NOW_TIMESTAMP, "latent": True}

    res = requests.post(DUCKLING_URL, data=data)
    responses = res.json()
    response_texts = [r["body"] for r in responses if r["dim"] == "time"]
    response_values = [
        r["value"]["value"]
        for r in responses
        if r["value"].get("value") and r["dim"] == "time"
    ]

    for p in predicted_texts:
        assert p in response_texts
    for p in predicted_values:
        assert p in response_values


def test_system_entity_recognizer_component_no_config(kwik_e_mart_app_path):
    # If the app has no config, then we need to default to duckling
    recognizer = SystemEntityRecognizer.load_from_app_path(kwik_e_mart_app_path)
    result = recognizer.parse("today is sunday")
    assert len(result[0]) > 0
    assert result[1] == 200


def test_system_entity_recognizer_component_empty_config(
    food_ordering_app_path, kwik_e_mart_app_path
):
    # If the app has an empty config (ie. {}), then it should not run system entity
    recognizer = SystemEntityRecognizer.load_from_app_path(food_ordering_app_path)
    result = recognizer.parse("today is sunday")
    assert result[0] == []
    assert result[1] == 200


test_data = [
    ("Đặt vé ngày mai", "vi", "ngày mai"),
    ("Ticket morgen buchen", "de", "morgen"),
    ("book ticket tomorrow", "en", "tomorrow"),
    ("明天订票", "zh", "明天"),
]


@pytest.mark.parametrize("text, language, expected_entity", test_data)
def test_get_candidates_for_text_language(text, language, expected_entity):
    candidates = DucklingRecognizer.get_instance().get_candidates_for_text(
        text, language=language
    )
    assert candidates[0]["body"] == expected_entity


test_data = [
    ("Đặt vé ngày mai", "vi_vi", "ngày mai"),
    ("Ticket morgen buchen", "de_de", "morgen"),
    ("book ticket tomorrow", "en_us", "tomorrow"),
    ("明天订票", "zh_cn", "明天"),
]


@pytest.mark.parametrize("text, locale, expected_entity", test_data)
def test_get_candidates_for_text_locale(text, locale, expected_entity):
    candidates = DucklingRecognizer.get_instance().get_candidates_for_text(
        text, locale=locale
    )
    assert candidates[0]["body"] == expected_entity
