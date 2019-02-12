import pytest
from attr.exceptions import FrozenInstanceError
from mmworkbench.components.request import Request, Params, FrozenParams


@pytest.fixture
def request():
    return Request(domain='some_domain', intent='some_intent', entities=(), text='some_text')


def test_domain(request):
    with pytest.raises(FrozenInstanceError):
        request.domain = 'new_domain'


def test_intent(request):
    with pytest.raises(FrozenInstanceError):
        request.intent = 'new_intent'


def test_entities(request):
    with pytest.raises(FrozenInstanceError):
        request.entities = ('some_entity',)


def test_text(request):
    with pytest.raises(FrozenInstanceError):
        request.text = 'some_text'


def test_frame(request):
    with pytest.raises(FrozenInstanceError):
        request.frame = {'key': 'value'}


def test_params(request):
    with pytest.raises(FrozenInstanceError):
        request.params = {'key': 'value'}


def test_context(request):
    with pytest.raises(FrozenInstanceError):
        request.context = {'key': 'value'}


def test_nbest(request):
    with pytest.raises(FrozenInstanceError):
        request.confidence = {'key': 'value'}

    with pytest.raises(FrozenInstanceError):
        request.nbest_transcripts_text = ['some_text']

    with pytest.raises(FrozenInstanceError):
        request.nbest_transcripts_entities = [{'key': 'value'}]

    with pytest.raises(FrozenInstanceError):
        request.nbest_aligned_entities = [{'key': 'value'}]


def test_immutability_of_request_and_params():
    """Test the immutability of the request and params objects"""
    with pytest.raises(FrozenInstanceError):
        params = FrozenParams()
        params.allowed_intents = []

    with pytest.raises(TypeError):
        params = FrozenParams()
        params.dynamic_resource['a'] = 'b'

    with pytest.raises(FrozenInstanceError):
        request = Request()
        request.params = Params()

    with pytest.raises(TypeError):
        request = Request()
        request.frame['a'] = 'b'
