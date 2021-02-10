#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_auto_annotator
----------------------------------

Tests for `auto_annotator` module
"""
import pytest

from mindmeld.auto_annotator import MultiLingualAnnotator


@pytest.fixture(scope="module")
def en_mla_annotator(kwik_e_mart_app_path):
    return MultiLingualAnnotator(kwik_e_mart_app_path)


@pytest.fixture(scope="module")
def es_mla_annotator(kwik_e_mart_app_path):
    return MultiLingualAnnotator(kwik_e_mart_app_path, language="es")


@pytest.fixture(scope="module")
def fr_mla_annotator(kwik_e_mart_app_path):
    return MultiLingualAnnotator(kwik_e_mart_app_path, language="fr")


# English Tests
@pytest.mark.parametrize(
    "query, body, value",
    [
        ("3rd place", "3rd", 3),
        ("fourth street", "fourth", 4),
        ("This is his fifth excuse", "fifth", 5),
        ("Who is the 1st born?", "1st", 1),
        ("5th", "5th", 5),
    ],
)
def en_test_ordinal_parse(en_mla_annotator, query, body, value):
    spacy_response = en_mla_annotator.parse(query)[0]
    assert body == spacy_response["body"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_ordinal"


@pytest.mark.parametrize(
    "query, body, value",
    [
        ("ten stones", "ten", 10),
        ("two hundred sheep", "two hundred", 200),
        ("58.67", "58.67", 58.67),
        ("thirty eight birds", "thirty eight", 38),
        ("1/4 apple", "1/4", 0.25),
        ("1,394,345.45 bricks", "1,394,345.45", 1394345.45),
        ("nine thousand and eight stories", "nine thousand and eight", 9008),
    ],
)
def en_test_cardinal_parse(en_mla_annotator, query, body, value):
    spacy_response = en_mla_annotator.parse(query)[0]
    assert body == spacy_response["body"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_number"


@pytest.mark.parametrize(
    "query, value",
    [
        ("four percent", 0.04),
        ("4 percent", 0.04),
        ("eight percent", 0.08),
        ("thirty eight percent", 0.38),
        ("12%", 0.12),
        ("seventy two percent", 0.72),
        ("thirty percent", 0.3),
    ],
)
def en_test_percent_parse(en_mla_annotator, query, value):
    spacy_response = en_mla_annotator.parse(query)[0]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_percent"


@pytest.mark.parametrize(
    "query, unit, value",
    [
        ("500 meters", "metre", 500),
        ("3 yards", "yard", 3),
        ("498 miles", "mile", 498),
        ("10 feet", "foot", 10),
        ("3.7 kilometers", "kilometre", 3.7),
        ("19 centimeters", "centimetre", 19),
        ("47.5 inches", "inch", 47.5),
    ],
)
def en_test_distance_parse(en_mla_annotator, query, unit, value):
    spacy_response = en_mla_annotator.parse(query)[0]
    assert unit == spacy_response["value"]["unit"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_distance"


@pytest.mark.parametrize(
    "query, unit, value",
    [
        ("400 pound", "pound", 400),
        ("3 grams", "gram", 3),
        ("498 kilograms", "gram", 498000),
        ("10 lbs", "pound", 10),
        ("3.7 kg", "gram", 3700),
        ("19 grams", "gram", 19),
        ("47.5 mg", "gram", 0.0475),
    ],
)
def en_test_weight_parse(en_mla_annotator, query, unit, value):
    spacy_response = en_mla_annotator.parse(query)[0]
    assert unit == spacy_response["value"]["unit"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_weight"


@pytest.mark.parametrize(
    "query, value",
    [
        ("Ron Paul", "Ron Paul"),
        ("James Madison's", "James Madison"),
        ("Bob Frank's", "Bob Frank"),
        ("Vickie Parilla", "Vickie Parilla"),
    ],
)
def en_test_person_parse(en_mla_annotator, query, value):
    spacy_response = en_mla_annotator.parse(query)[0]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_person"


@pytest.mark.parametrize(
    "query, unit, value",
    [
        ("ten dollars", "$", 10),
        ("two dollars", "$", 2),
        ("two hundred dollars", "$", 200),
        ("$58.67", "$", 58.67),
        ("thirty-eight euros", "EUR", 38),
        ("1/4 dollar", "$", 0.25),
        ("seventy-eight euros", "EUR", 78),
        ("$70k", "$", 70000),
    ],
)
def en_test_money_parse(en_mla_annotator, query, unit, value):
    spacy_response = en_mla_annotator.parse(query)[0]
    assert unit == spacy_response["value"]["unit"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_amount-of-money"


@pytest.mark.parametrize(
    "query, unit, value",
    [
        ("about 2 hours", "hour", 2),
        ("2 hours", "hour", 2),
        ("half an hour", "minute", 30),
        ("15 minutes", "minute", 15),
        ("nearly 15 minutes", "minute", 15),
        ("2 days", "day", 2),
        ("4 years", "year", 4),
        ("3 weeks", "week", 3),
    ],
)
def en_test_duration_parse(en_mla_annotator, query, unit, value):
    spacy_response = en_mla_annotator.parse(query)[0]
    assert unit == spacy_response["value"]["unit"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_duration"


@pytest.mark.parametrize(
    "query, grain, value",
    [
        ("March 3rd 2012", "day", "2012-03-03T00:00:00.000-08:00"),
        ("Feb 18 2019", "day", "2019-02-18T00:00:00.000-08:00"),
        ("first thursday of 2015", "day", "2015-01-01T00:00:00.000-08:00"),
    ],
)
def en_test_time_parse(en_mla_annotator, query, grain, value):
    spacy_response = en_mla_annotator.parse(query)[0]
    assert grain == spacy_response["value"]["grain"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_time"


# Spanish Tests
@pytest.mark.parametrize(
    "query, unit, value",
    [
        ("diez dólares", "$", 10),
        ("dos dólares", "$", 2),
        ("setenta y ocho euros", "EUR", 78),
        ("$ 58.67", "$", 58.67),
        ("cuatrocientos DOLARES", "$", 400),
        ("$ 100000", "$", 100000),
    ],
)
def es_test_money_parse(es_mla_annotator, query, unit, value):
    spacy_response = es_mla_annotator.parse(query)[0]
    assert unit == spacy_response["value"]["unit"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_amount-of-money"


@pytest.mark.parametrize(
    "query, unit, value",
    [("2 horas", "hour", 2), ("15 minutos", "minute", 15), ("3 días", "day", 3)],
)
def es_test_duration_parse(es_mla_annotator, query, unit, value):
    spacy_response = es_mla_annotator.parse(query)[0]
    assert unit == spacy_response["value"]["unit"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_duration"


@pytest.mark.parametrize(
    "query, body, value",
    [
        ("Tengo 32 reuniones", "32", 32),
        ("diez piedras", "diez", 10),
        ("James Bond tiene 58,67 frutos", "58,67", 58.67),
        ("treinta y ocho pájaros", "treinta y ocho", 38),
        ("1/4 manzana", "1/4", 0.25),
        ("1.394.345,45 ladrillos", "1.394.345,45", 1394345.45),
        ("nueve mil ocho mujeres", "nueve", 9),
    ],
)
def es_test_cardinal_parse(es_mla_annotator, query, body, value):
    spacy_response = es_mla_annotator.parse(query)[0]
    assert body == spacy_response["body"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_number"


@pytest.mark.parametrize(
    "query, body, value",
    [
        ("¿Cuándo es tu segunda reunión?", "segunda", 2),
        ("caminando por la cuarta calle", "cuarta", 4),
        ("Esta es su quinta excusa", "quinta", 5),
        ("¿Quién es el primero en nacer?", "primero", 1),
    ],
)
def es_test_ordinal_parse(es_mla_annotator, query, body, value):
    spacy_response = es_mla_annotator.parse(query)[0]
    assert body == spacy_response["body"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_ordinal"


@pytest.mark.parametrize(
    "query, value",
    [
        ("Hagamos que Jason se una a la reunión", "Jason"),
        ("Llama a Sarah ahora", "Sarah"),
        ("¿Dónde está Robert?", "Robert"),
        ("¿Samantha está enferma hoy?", "Samantha"),
        ("Richard, necesito tu ayuda", "Richard"),
    ],
)
def es_test_person_parse(es_mla_annotator, query, value):
    spacy_response = es_mla_annotator.parse(query)[0]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_person"


@pytest.mark.parametrize(
    "query, grain, value",
    [
        ("3 de marzo de 2012", "day", "2012-03-03T00:00:00.000-08:00"),
        ("18 de febrero de 2019", "day", "2019-02-18T00:00:00.000-08:00"),
        ("primer jueves de 2015", "day", "2015-01-01T00:00:00.000-08:00"),
    ],
)
def es_test_time_parse(es_mla_annotator, query, grain, value):
    spacy_response = es_mla_annotator.parse(query)[0]
    assert grain == spacy_response["value"]["grain"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_time"


# French Tests
@pytest.mark.parametrize(
    "query, unit, value",
    [
        ("dix dollars", "$", 10),
        ("deux dollars", "$", 2),
        ("soixante euros", "EUR", 60),
        ("58,67 $", "$", 58.67),
        ("quatre cents", "cent", 4),
        ("100 000 $", "$", 100000),
    ],
)
def fr_test_money_parse(fr_mla_annotator, query, unit, value):
    spacy_response = fr_mla_annotator.parse(query)[0]
    assert unit == spacy_response["value"]["unit"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_amount-of-money"


@pytest.mark.parametrize(
    "query, unit, value",
    [("2 heures", "hour", 2), ("15 minutes", "minute", 15), ("3 jours", "day", 3)],
)
def fr_test_duration_parse(fr_mla_annotator, query, unit, value):
    spacy_response = fr_mla_annotator.parse(query)[0]
    assert unit == spacy_response["value"]["unit"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_duration"


@pytest.mark.parametrize(
    "query, body, value",
    [
        ("32 réunions", "32", 32),
        ("dix pierres", "dix", 10),
        ("deux moutons", "deux", 2),
        ("J'ai acheté 58,67 fruits", "58,67", 58.67),
        ("trente huit oiseaux", "trente huit", 38),
        ("1/4 pomme", "1/4", 0.25),
        ("neuf mille huit femmes", "neuf mille huit", 9008),
    ],
)
def fr_test_cardinal_parse(fr_mla_annotator, query, body, value):
    spacy_response = fr_mla_annotator.parse(query)[0]
    assert body == spacy_response["body"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_number"


@pytest.mark.parametrize(
    "query, body, value",
    [
        ("Quand a lieu votre deuxième rencontre?", "deuxième", 2),
        ("3ème place", "3ème", 3),
        ("marchant dans la quatrième rue", "quatrième", 4),
        ("cinquième excuse", "cinquième", 5),
        ("Qui est le premier né?", "premier", 1),
    ],
)
def fr_test_ordinal_parse(fr_mla_annotator, query, body, value):
    spacy_response = fr_mla_annotator.parse(query, entity_types=["sys_ordinal"])[0]
    assert body == spacy_response["body"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_ordinal"


@pytest.mark.parametrize(
    "query, value",
    [
        ("Laissons Jason rejoindre la réunion", "Laissons Jason"),
        ("Où est Robert?", "Robert"),
        ("Samantha est-elle malade aujourd'hui?", "Samantha"),
        ("Richard, j'ai besoin de ton aide", "Richard"),
    ],
)
def fr_test_person_parse(fr_mla_annotator, query, value):
    spacy_response = fr_mla_annotator.parse(query)[0]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_person"


@pytest.mark.parametrize(
    "query, grain, value",
    [
        ("3 mars 2012", "day", "2012-03-03T00:00:00.000-08:00"),
        ("18 février 2019", "day", "2019-02-18T00:00:00.000-08:00"),
        ("premier jeudi de 2015", "day", "2015-01-01T00:00:00.000-08:00"),
    ],
)
def fr_test_time_parse(fr_mla_annotator, query, grain, value):
    spacy_response = fr_mla_annotator.parse(query)[0]
    assert grain == spacy_response["value"]["grain"]
    assert value == spacy_response["value"]["value"]
    assert spacy_response["dim"] == "sys_time"


@pytest.mark.parametrize(
    "rule, pattern",
    [
        (
            {
                "domains": "(faq|salary)",
                "intents": ".*",
                "files": "(train.txt|test.txt)",
                "entities": "(sys_amt-of-money|sys_time)",
            },
            ".*/(faq|salary)/.*/(train.txt|test.txt)",
        ),
        (
            {
                "domains": "salary",
                "intents": "(get_salary_aggregate|get_salary)",
                "files": "(train.txt|test.txt)",
                "entities": "(sys_amt-of-money)",
            },
            ".*/salary/(get_salary_aggregate|get_salary)/(train.txt|test.txt)",
        ),
        (
            {
                "domains": "date",
                "intents": "get_date",
                "files": "(train.txt|test.txt)",
                "entities": "sys_duration|sys_interval|sys_time",
            },
            ".*/date/get_date/(train.txt|test.txt)",
        ),
        (
            {"domains": "general", "intents": ".+", "files": ".+", "entities": "*"},
            ".*/general/.+/.+",
        ),
        (
            {"domains": ".+", "intents": ".+", "files": ".+", "entities": "*"},
            ".*/.+/.+/.+",
        ),
        (
            {
                "domains": "[A-Z]*",
                "intents": "[a-z]*",
                "files": "[a-z]*",
                "entities": "*",
            },
            ".*/[A-Z]*/[a-z]*/[a-z]*",
        ),
    ],
)
def test_rule_to_regex_pattern_parser(en_mla_annotator, rule, pattern):
    assert pattern == en_mla_annotator._get_pattern(rule)
