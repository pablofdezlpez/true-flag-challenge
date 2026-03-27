import pytest
import numpy as np

from src.Database.retriever import Retriever


@pytest.fixture
def retriever():
    return Retriever("chroma_db")


def test_retriever_query_text(retriever, mocker):
    fake_embedding = [np.zeros(3072)]
    mock_embed = mocker.patch(
        "src.Database.retriever.embed_texts",
        return_value=fake_embedding,
    )
    query_text = "What is the capital of France?"
    evidence = retriever.query_by_text(query_text)
    mock_embed.assert_called_once_with([query_text])
    assert isinstance(evidence, list)
    assert len(evidence) > 0


def test_retriever_query_image(retriever, mocker):
    fake_embedding = np.zeros(3072)
    mock_embed = mocker.patch(
        "src.Database.retriever.embed_image_bytes",
        return_value=fake_embedding,
    )
    image_path = "test/test.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    evidence = retriever.query_by_image(image_bytes)
    mock_embed.assert_called_once_with(image_bytes)
    assert isinstance(evidence, list)
    assert len(evidence) > 0
