import os

import pytest
from numpy import ndarray

from mindmeld.models import Embedder
from mindmeld.models.embedder_models import BertEmbedder, GloveEmbedder
from mindmeld.models.taggers.embeddings import GloVeEmbeddingsContainer

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


@pytest.mark.extras
@pytest.mark.torch
def test_custom_embedder():
    class MyCustomEmbedder(Embedder):
        def load(self, custom_dim=10):
            self.custom_dim = 10
            self.offset = 0.0

        # return the loaded model
        def encode(self, text_list):
            encoded = [[float(self.offset + i)] * self.custom_dim for i, text in
                       enumerate(text_list)]
            self.offset += len(set(text_list))
            return encoded

    text_list = ["hello", "world", "hello", "monday"]

    my_custom_embedder = MyCustomEmbedder()
    my_custom_embedder.get_encodings(text_list)
    assert len(my_custom_embedder.cache) == 3
    assert my_custom_embedder.get_encodings(["hello"]) == [[0.0] * my_custom_embedder.custom_dim]
    assert my_custom_embedder.find_similarity(["new_word"])[0] == [
        ('world', 1.0), ('monday', 1.0), ('hello', 0.0)]
    with pytest.raises(ValueError):
        # ValueError: Invalid cache path '(None)' provided for EmbeddingsCache.
        my_custom_embedder.dump()

    my_custom_embedder = MyCustomEmbedder(app_path=APP_PATH)
    my_custom_embedder.get_encodings(text_list)
    my_custom_embedder.dump()
    assert os.path.exists(os.path.join(APP_PATH, ".generated/indexes", "default_cache.pkl"))

    class MyAnotherCustomEmbedder(MyCustomEmbedder):
        @property
        def model_id(self):
            # identify indices with a unique name
            return "my_another_custom_embedder_0"

    my_custom_another_embedder = MyAnotherCustomEmbedder(app_path=APP_PATH)
    my_custom_another_embedder.get_encodings(text_list)
    my_custom_another_embedder.dump()
    assert os.path.exists(
        os.path.join(APP_PATH, ".generated/indexes", "my_another_custom_embedder_0_cache.pkl")
    )
