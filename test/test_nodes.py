from src.nodes import (
    State,
    generate_answer,
    rag_search,
    judge_answer,
    have_judge_approved,
    where_documents_found_retrieved,
)
from src.database import Database as Retriever
import pytest
from unittest.mock import Mock
from langgraph.graph import END


@pytest.fixture
def agentic_model():
    mock_agent = Mock()
    mock_agent.call.return_value = Mock(
        candidates=[Mock(content=Mock(parts=[Mock(text="An answer")]))]
    )
    return mock_agent


@pytest.fixture
def judge():
    mock_judge = Mock()
    mock_judge.call.return_value = (True, "Reasoning for approval")
    return mock_judge


@pytest.fixture
def retriever():
    return Retriever("chroma_db")


@pytest.fixture
def state(retriever, judge, agentic_model):
    return State(
        retriever=retriever,
        query_text="What is the capital of France?",
        query_image_bytes=None,
        retrieved_docs=None,
        current_doc=None,
        judge=judge,
        judge_approve=False,
        judge_reasoning="",
        agent=agentic_model,
        answer="",
    )


def test_rag_search_with_text_query(state):
    updated_state = rag_search(state)
    assert updated_state.retriever == state.retriever
    assert updated_state.query_text == state.query_text
    assert updated_state.query_image_bytes == state.query_image_bytes
    assert updated_state.retrieved_docs is not None
    assert updated_state.current_doc is not None
    assert updated_state.answer == ""


def test_rag_search_with_image_query(state):
    # Load test image bytes
    with open("test/test.jpg", "rb") as f:
        image_bytes = f.read()

    state.query_image_bytes = image_bytes
    updated_state = rag_search(state)
    assert updated_state.retriever == state.retriever
    assert updated_state.query_text == state.query_text
    assert updated_state.query_image_bytes == state.query_image_bytes
    assert updated_state.retrieved_docs is not None
    assert updated_state.current_doc is not None
    assert updated_state.answer == ""


def test_generate_answer(state):
    # First run retrieval to populate current_doc
    state = rag_search(state)
    updated_state = generate_answer(state)
    assert updated_state.answer == "An answer"


def test_route_judge_saves_response_and_reasoning(state):
    mocked_document = Mock()
    mocked_document.url = "url1"
    mocked_document.title = "title1"
    mocked_document.text = "text1"
    state.current_doc = mocked_document
    state = judge_answer(state)
    assert state.judge_approve is True
    assert state.judge_reasoning == "Reasoning for approval"


def test_have_judge_approved_not_approved_goes_to_next_doc(state):
    state.judge_approve = False
    state.judge_reasoning = "Reasoning for approval"
    state.current_doc = Mock()

    mocked_document2 = Mock()
    mocked_document2.url = "url2"
    state.retrieved_docs = [mocked_document2]

    next_node = have_judge_approved(state)
    assert next_node is False
    assert state.current_doc.url == mocked_document2.url


def test_have_judge_approved_not_approved_no_more_docs(state):
    state.judge_approve = False
    state.judge_reasoning = "Reasoning for approval"
    state.current_doc = Mock()

    state.retrieved_docs = []
    next_node = have_judge_approved(state)
    assert next_node is True


def test_have_judge_approved_approved_goes_to_end(state):
    state.judge_approve = True
    state.judge_reasoning = "Reasoning for approval"
    state.current_doc = Mock()
    next_node = have_judge_approved(state)
    assert next_node is True


def test_where_documents_found_retrieved(state):
    mocked_document = Mock()
    mocked_document.url = "url1"
    mocked_document.title = "title1"
    mocked_document.text = "text1"
    state.current_doc = mocked_document
    assert where_documents_found_retrieved(state) is True


def test_where_documents_found_retrieved_no_docs(state):
    state.current_doc = None
    assert where_documents_found_retrieved(state) is False
