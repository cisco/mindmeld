import pytest
from attr.exceptions import FrozenInstanceError

from mindmeld.components.request import FrozenParams, Params, Request


@pytest.fixture
def sample_request():
    return Request(
        domain="some_domain", intent="some_intent", entities=(), text="some_text"
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
        sample_request.nbest_transcripts_text = ["some_text"]

    with pytest.raises(FrozenInstanceError):
        sample_request.nbest_transcripts_entities = [{"key": "value"}]

    with pytest.raises(FrozenInstanceError):
        sample_request.nbest_aligned_entities = [{"key": "value"}]


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
