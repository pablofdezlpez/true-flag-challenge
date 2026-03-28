import pytest
import numpy as np

from src.Database.retriever import Retriever


@pytest.fixture
def retriever():
    return Retriever("chroma_db")


def test_retriever_query_text(retriever):
    fake_embedding = [np.zeros(3072)]

    query_text = "What is the capital of France?"
    evidence = retriever.query_by_text(query_text)
    assert isinstance(evidence, list)
    assert len(evidence) > 0


def test_retriever_query_image(retriever):
    fake_embedding = np.zeros(3072)

    image_path = "test/test.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    evidence = retriever.query_by_image(image_bytes)
    assert isinstance(evidence, list)
    assert len(evidence) > 0
