import pytest
from src.database import Database as Indexer


@pytest.fixture
def retriever(tmp_path_factory):
    fn = tmp_path_factory.mktemp("test_chroma_db")
    return Indexer(str(fn))


def test_indexer_index_once_per_summary(retriever):
    retriever.index_csv("test/test_data.csv")

    collection = retriever._client.get_collection("summaries")

    assert len(retriever._client.list_collections()) == 1
    assert collection.count() == 5


def test_retriever_query_text(retriever):

    query_text = "What is the capital of France?"
    evidence = retriever.query_by_text(query_text)
    assert isinstance(evidence, list)
    assert len(evidence) > 0


def test_retriever_query_image(retriever):

    image_path = "test/test.jpg"
    with open(image_path, "rb") as f:
        image_bytes = f.read()
    evidence = retriever.query_by_image(image_bytes)
    assert isinstance(evidence, list)
    assert len(evidence) > 0
