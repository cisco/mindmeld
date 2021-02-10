import pytest
from attr.exceptions import FrozenInstanceError

from mindmeld.components.request import FrozenParams, Params, Request
from mindmeld.app_manager import freeze_params
from immutables import Map


@pytest.fixture
def sample_request():
    dict_list = [{"key": "value"}, {"key": "value"}, {"key": "value"}]
    dict_list_of_lists = [[{"key": "value"}, {"key": "value"}, {"key": "value"}],
                          [{"key": "value"}, {"key": "value"}, {"key": "value"}]]
    lists = ['key_1', 'key_2', 'key_3']
    return Request(
        domain="some_domain",
        intent="some_intent",
        entities=dict_list,
        text="some_text",
        history=dict_list,
        nbest_transcripts_text=lists,
        nbest_transcripts_entities=dict_list_of_lists,
        nbest_aligned_entities=dict_list_of_lists,
    )


def test_domain(sample_request):
    with pytest.raises(FrozenInstanceError):
        sample_request.domain = "new_domain"


def test_intent(sample_request):
    with pytest.raises(FrozenInstanceError):
        sample_request.intent = "new_intent"


def test_entities(sample_request):
    with pytest.raises(FrozenInstanceError):
        sample_request.entities = ("some_entity",)
    assert_tuple_of_immutable_maps(sample_request.entities)


def test_history(sample_request):
    for x in sample_request.history:
        assert isinstance(x, Map)
    assert_tuple_of_immutable_maps(sample_request.history)


def test_text(sample_request):
    with pytest.raises(FrozenInstanceError):
        sample_request.text = "some_text"


def test_frame(sample_request):
    with pytest.raises(FrozenInstanceError):
        sample_request.frame = {"key": "value"}


def test_params(sample_request):
    with pytest.raises(FrozenInstanceError):
        sample_request.params = {"key": "value"}


def test_context(sample_request):
    with pytest.raises(FrozenInstanceError):
        sample_request.context = {"key": "value"}


def test_nbest(sample_request):
    with pytest.raises(FrozenInstanceError):
        sample_request.confidences = {"key": "value"}

    with pytest.raises(FrozenInstanceError):
        sample_request.nbest_transcripts_text = ["some_text", "some_text_2"]
    assert isinstance(sample_request.nbest_transcripts_text, tuple)

    with pytest.raises(FrozenInstanceError):
        sample_request.nbest_transcripts_entities = [[{"key": "value"}], [{"key": "value"}]]
    assert_tuple_of_tuple_of_immutable_maps(sample_request.nbest_transcripts_entities)

    with pytest.raises(FrozenInstanceError):
        sample_request.nbest_aligned_entities = [[{"key": "value"}], [{"key": "value"}]]
    assert_tuple_of_tuple_of_immutable_maps(sample_request.nbest_aligned_entities)


def assert_tuple_of_immutable_maps(tuple_of_immutable_maps):
    for idx, immutable_map in enumerate(tuple_of_immutable_maps):
        with pytest.raises(TypeError):
            tuple_of_immutable_maps[idx] = {"key": "value"}
        assert isinstance(immutable_map, Map)
        with pytest.raises(TypeError):
            immutable_map["key"] = "value"


def assert_tuple_of_tuple_of_immutable_maps(tuple_of_tuple_of_immutable_maps):
    assert isinstance(tuple_of_tuple_of_immutable_maps, tuple)
    for idx_1, tuple_of_immutable_maps in enumerate(tuple_of_tuple_of_immutable_maps):
        assert_tuple_of_immutable_maps(tuple_of_immutable_maps)


def test_immutability_of_sample_request_and_params():
    """Test the immutability of the sample_request and params objects"""
    with pytest.raises(FrozenInstanceError):
        params = FrozenParams()
        params.allowed_intents = []

    with pytest.raises(TypeError):
        params = FrozenParams()
        params.dynamic_resource["a"] = "b"

    with pytest.raises(FrozenInstanceError):
        request = Request()
        request.params = Params()

    with pytest.raises(TypeError):
        request = Request()
        request.frame["a"] = "b"


@pytest.mark.parametrize(
    "allowed_intents, target_dialogue_state, time_zone, timestamp, language, locale, "
    "dynamic_resource",
    [
        (
            ["some-intents", "some-intents-2"],
            "some-state",
            "America/Los_Angeles",
            "1234",
            "en",
            "en_US",
            {"resource": "dynamic"},
        )
    ],
)
def test_serialization_params(
    allowed_intents,
    target_dialogue_state,
    time_zone,
    timestamp,
    language,
    locale,
    dynamic_resource,
):
    params = Params()
    params.allowed_intents = allowed_intents
    params.target_dialogue_state = target_dialogue_state
    params.time_zone = time_zone
    params.timestamp = timestamp
    params.language = language
    params.locale = locale
    params.dynamic_resource = Map(dynamic_resource)
    dict_result = dict(params)
    assert allowed_intents == dict_result["allowed_intents"]
    assert target_dialogue_state == dict_result["target_dialogue_state"]
    assert time_zone == dict_result["time_zone"]
    assert timestamp == dict_result["timestamp"]
    assert language == dict_result["language"]
    assert locale == dict_result["locale"]
    assert dynamic_resource == dict_result["dynamic_resource"]


@pytest.mark.parametrize(
    "allowed_intents, target_dialogue_state, time_zone, timestamp, language, locale, "
    "dynamic_resource",
    [
        (
            ["some-intents", "some-intents-2"],
            "some-state",
            "America/Los_Angeles",
            "1234",
            "en",
            "en_US",
            {"resource": "dynamic"},
        )
    ],
)
def test_serialization_frozen_params(
    allowed_intents,
    target_dialogue_state,
    time_zone,
    timestamp,
    language,
    locale,
    dynamic_resource,
):
    params = Params()
    params.allowed_intents = allowed_intents
    params.target_dialogue_state = target_dialogue_state
    params.time_zone = time_zone
    params.timestamp = timestamp
    params.language = language
    params.locale = locale
    params.dynamic_resource = Map(dynamic_resource)
    frozen_params = freeze_params(params)
    dict_result = dict(frozen_params)
    assert allowed_intents == dict_result["allowed_intents"]
    assert target_dialogue_state == dict_result["target_dialogue_state"]
    assert time_zone == dict_result["time_zone"]
    assert timestamp == dict_result["timestamp"]
    assert language == dict_result["language"]
    assert locale == dict_result["locale"]
    assert dynamic_resource == dict_result["dynamic_resource"]
