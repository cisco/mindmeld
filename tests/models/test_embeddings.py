import os
import pytest
from numpy import ndarray

from mindmeld.models.taggers.embeddings import GloVeEmbeddingsContainer
from mindmeld.models.embedder_models import BertEmbedder, GloveEmbedder

APP_NAME = "kwik_e_mart"
APP_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), APP_NAME
)


@pytest.mark.xfail(strict=False)
def test_embedding_size_is_correct():
    """Tests the size and type of the embedding"""
    token_to_embedding_mapping = GloVeEmbeddingsContainer(
        50, None
    ).get_pretrained_word_to_embeddings_dict()
    assert len(token_to_embedding_mapping[b"sandberger"]) == 50
    assert type(token_to_embedding_mapping[b"sandberger"]) == ndarray


@pytest.mark.skip(reason="sentence bert embedding URL is failing to download")
@pytest.mark.xfail(strict=False)
@pytest.mark.extras
@pytest.mark.bert
def test_bert_embedder():
    embedder = BertEmbedder(
        APP_PATH, **{"model_name": "bert-base-nli-mean-tokens", "embedder_type": "bert"}
    )
    encoded_vec = embedder.encode(["test string"])[0]
    assert len(encoded_vec) == 768
    assert type(encoded_vec) == ndarray


@pytest.mark.skip(reason="sentence bert embedding URL is failing to download")
@pytest.mark.xfail(strict=False)
@pytest.mark.extras
@pytest.mark.bert
def test_bert_embedder_without_model_name():
    with pytest.raises(ValueError):
        BertEmbedder(APP_PATH)


@pytest.mark.xfail(strict=False)
def test_glove_embedder():
    embedder = GloveEmbedder(
        APP_PATH,
        **{
            "token_embedding_dimension": 300,
            "model_name": "glove300",
            "embedder_type": "glove",
        }
    )
    encoded_vec = embedder.encode(["test string"])[0]
    assert len(encoded_vec) == 300
    assert type(encoded_vec) == ndarray
