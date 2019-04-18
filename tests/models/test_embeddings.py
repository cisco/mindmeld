from mindmeld.models.taggers.embeddings import GloVeEmbeddingsContainer
from numpy import ndarray


def test_embedding_size_is_correct():
    """Tests the size and type of the embedding"""
    token_to_embedding_mapping = \
        GloVeEmbeddingsContainer(50, None).get_pretrained_word_to_embeddings_dict()
    assert len(token_to_embedding_mapping[b'sandberger']) == 50
    assert type(token_to_embedding_mapping[b'sandberger']) == ndarray
