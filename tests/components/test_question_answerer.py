#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_question_answerer
----------------------------------

Tests for `question_answerer` module.
"""
import os

# pylint: disable=locally-disabled,redefined-outer-name
import pytest

from mindmeld.components._elasticsearch_helpers import create_es_client
from mindmeld.components.question_answerer import QuestionAnswerer

ENTITY_TYPE = "store_name"
STORE_DATA_FILE_PATH = os.path.dirname(__file__) + "/../kwik_e_mart/data/stores.json"
DISH_DATA_FILE_PATH = (
    os.path.dirname(__file__) + "/../food_ordering/data/menu_items.json"
)


@pytest.fixture
def es_client():
    """An Elasticsearch client"""
    return create_es_client()


@pytest.fixture
def answerer(kwik_e_mart_app_path, es_client):
    QuestionAnswerer.load_kb(
        app_namespace="kwik_e_mart",
        index_name="store_name",
        data_file=STORE_DATA_FILE_PATH,
    )
    es_client.indices.flush(index="_all", force=True)

    qa = QuestionAnswerer(kwik_e_mart_app_path)
    return qa


@pytest.fixture
def relative_answerer(kwik_e_mart_app_path, es_client):
    QuestionAnswerer.load_kb(
        app_namespace="kwik_e_mart",
        index_name="store_name",
        data_file=STORE_DATA_FILE_PATH,
    )
    es_client.indices.flush(index="_all", force=True)
    old_cwd = os.getcwd()
    os.chdir(kwik_e_mart_app_path)
    qa = QuestionAnswerer(".")
    os.chdir(old_cwd)
    return qa


@pytest.fixture
def food_ordering_answerer(food_ordering_app_path, es_client):
    QuestionAnswerer.load_kb(
        app_namespace="food_ordering",
        index_name="menu_items",
        data_file=DISH_DATA_FILE_PATH,
    )
    es_client.indices.flush(index="_all", force=True)

    qa = QuestionAnswerer(food_ordering_app_path)
    return qa


@pytest.mark.extras
@pytest.fixture
def food_ordering_with_bert(food_ordering_app_path, es_client):
    bert_qa_config = {
        "model_type": "embedder",
        "model_settings": {
            "embedder_type": "bert",
            "embedding_fields": {"menu_items_bert": ["name"]},
        },
    }
    QuestionAnswerer.load_kb(
        app_namespace="food_ordering",
        index_name="menu_items_bert",
        data_file=DISH_DATA_FILE_PATH,
        config=bert_qa_config,
    )
    es_client.indices.flush(index="_all", force=True)

    qa = QuestionAnswerer(food_ordering_app_path, config=bert_qa_config)
    return qa


@pytest.mark.extras
@pytest.fixture
def food_ordering_with_glove(food_ordering_app_path, es_client):
    glove_qa_config = {
        "model_type": "embedder",
        "model_settings": {
            "embedder_type": "glove",
            "embedding_fields": {"menu_items_glove": ["name"]},
        },
    }
    QuestionAnswerer.load_kb(
        app_namespace="food_ordering",
        index_name="menu_items_glove",
        data_file=DISH_DATA_FILE_PATH,
        config=glove_qa_config,
    )
    es_client.indices.flush(index="_all", force=True)

    qa = QuestionAnswerer(food_ordering_app_path, config=glove_qa_config)
    return qa


def test_basic_search(answerer):
    """Test basic search."""

    # retrieve object using ID
    res = answerer.get(index="store_name", id="20")
    assert len(res) > 0

    # simple text query
    res = answerer.get(index="store_name", store_name="peanut")
    assert len(res) > 0

    # simple text query
    res = answerer.get(index="store_name", store_name="Springfield Heights")
    assert len(res) > 0

    # multiple text queries
    res = answerer.get(index="store_name", store_name="peanut", address="peanut st")
    assert len(res) > 0


def test_basic_relative_search(relative_answerer):
    """Test basic search."""

    # retrieve object using ID
    res = relative_answerer.get(index="store_name", id="20")
    assert len(res) > 0

    # simple text query
    res = relative_answerer.get(index="store_name", store_name="peanut")
    assert len(res) > 0

    # simple text query
    res = relative_answerer.get(index="store_name", store_name="Springfield Heights")
    assert len(res) > 0

    # multiple text queries
    res = relative_answerer.get(
        index="store_name", store_name="peanut", address="peanut st"
    )
    assert len(res) > 0


def test_advanced_search(answerer):
    """Test advanced search."""

    s = answerer.build_search(index="store_name")
    res = s.query(store_name="peanut").execute()
    assert len(res) > 0


def test_partial_match(answerer):
    """Test partial match."""

    # test partial match
    res = answerer.get(index="store_name", store_name="Garden")
    assert len(res) > 0


def test_sort_by_distance(answerer):
    """Test sort by distance."""

    # retrieve object using ID
    res = answerer.get(
        index="store_name",
        _sort="location",
        _sort_type="distance",
        _sort_location="44.24,-123.12",
    )
    assert len(res) > 0
    assert res[0].get("id") == "19"


def test_basic_search_validation(food_ordering_answerer):
    """Test validation."""

    # index not exist
    with pytest.raises(ValueError):
        food_ordering_answerer.get(index="nosuchindex", nosuchfield="novalue")

    # field not exist
    with pytest.raises(ValueError):
        food_ordering_answerer.get(index="menu_items", nosuchfield="novalue")

    # invalid field type
    with pytest.raises(ValueError):
        food_ordering_answerer.get(index="menu_items", price="novalue")

    # invalid sort type
    with pytest.raises(ValueError):
        food_ordering_answerer.get(
            index="menu_items", _sort="price", _sort_type="distance"
        )

    # invalid sort type
    with pytest.raises(ValueError):
        food_ordering_answerer.get(
            index="menu_items", _sort="location", _sort_type="asc"
        )

    # missing origin
    with pytest.raises(ValueError):
        food_ordering_answerer.get(
            index="menu_items", _sort="location", _sort_type="distance"
        )


def test_unstructured_search(food_ordering_answerer):
    res = food_ordering_answerer.get(
        index="menu_items",
        query_type="text",
        description="something with crab meat and scallops",
    )
    assert len(res) > 0

    s = food_ordering_answerer.build_search(index="menu_items")
    res = s.query(
        query_type="text", description="maybe a roll with some salmon"
    ).execute()
    assert len(res) > 0

    res = (
        s.query(query_type="text", description="maybe a spicy roll with some salmon")
        .filter(query_type="text", name="spicy roll")
        .execute()
    )
    assert len(res) > 0


def test_advanced_search_validation(answerer):
    """Tests validation in advanced search."""

    # index not exist
    with pytest.raises(ValueError):
        s = answerer.build_search(index="nosuchindex")
        s.query(fieldnotexist="test")

    # field not exist
    with pytest.raises(ValueError):
        s = answerer.build_search(index="store_name")
        s.query(fieldnotexist="test")

    # invalid field type
    with pytest.raises(ValueError):
        s = answerer.build_search(index="store_name")
        s.query(location="testlocation")

    # range filter can only be specified with number or date fields.
    with pytest.raises(ValueError):
        s = answerer.build_search(index="store_name")
        s.filter(field="phone_number", gt=10)

    # sort field to be number or date type.
    with pytest.raises(ValueError):
        s = answerer.build_search(index="store_name")
        s.sort(field="store_name", sort_type="asc")

    # missing origin
    with pytest.raises(ValueError):
        s = answerer.build_search(index="store_name")
        s.sort(field="location", sort_type="distance")


@pytest.mark.extras
@pytest.mark.bert
def test_embedder_search_bert(food_ordering_with_bert):
    BERT_EMBEDDING_LEN = 768
    res = food_ordering_with_bert.get(
        index="menu_items_bert", query_type="embedder", name="pasta with tomato sauce"
    )
    assert len(res) > 0
    assert len(res[0].get("name_embedding")) == BERT_EMBEDDING_LEN

    res = food_ordering_with_bert.get(
        index="menu_items_bert",
        query_type="embedder_keyword",
        name="pasta with tomato sauce",
    )
    assert len(res) > 0
    assert len(res[0].get("name_embedding")) == BERT_EMBEDDING_LEN

    res = food_ordering_with_bert.get(
        index="menu_items_bert",
        query_type="embedder_text",
        name="pasta with tomato sauce",
    )
    assert len(res) > 0
    assert len(res[0].get("name_embedding")) == BERT_EMBEDDING_LEN


@pytest.mark.extras
def test_embedder_search_glove(food_ordering_with_glove):
    GLOVE_EMBEDDING_LEN = 300
    res = food_ordering_with_glove.get(
        index="menu_items_glove", query_type="embedder", name="pasta with tomato sauce"
    )
    assert len(res) > 0
    assert len(res[0].get("name_embedding")) == GLOVE_EMBEDDING_LEN

    res = food_ordering_with_glove.get(
        index="menu_items_glove",
        query_type="embedder_keyword",
        name="pasta with tomato sauce",
    )
    assert len(res) > 0
    assert len(res[0].get("name_embedding")) == GLOVE_EMBEDDING_LEN

    res = food_ordering_with_glove.get(
        index="menu_items_glove",
        query_type="embedder_text",
        name="pasta with tomato sauce",
    )
    assert len(res) > 0
    assert len(res[0].get("name_embedding")) == GLOVE_EMBEDDING_LEN
