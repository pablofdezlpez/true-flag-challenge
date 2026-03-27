from src.Chatbot.nodes import State, generate_answer, rag_search, route_judge_answer
from src.Database.retriever import Retriever
import pytest
from unittest.mock import Mock
from langgraph.graph import END


@pytest.fixture
def image_crescription_model():
    mock_model = Mock()
    mock_response = Mock()
    mock_response.content = "An image"
    mock_model.invoke.return_value = mock_response
    return mock_model


@pytest.fixture
def agentic_model():
    mock_agent = Mock()
    mock_agent.call.return_value = "An answer"
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
        judge_approve="",
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


def test_route_judge_answer_approved(state):
    mocked_document = Mock()
    mocked_document.url = "url1"
    mocked_document.title = "title1"
    mocked_document.text = "text1"
    state.current_doc = mocked_document
    next_node = route_judge_answer(state)
    assert next_node == END


def test_route_judge_answer_not_approved_with_more_docs(state):
    mock_judge = Mock()
    mock_judge.call.return_value = (False, "Reasoning for approval")
    state.judge = mock_judge

    mocked_document = Mock()
    mocked_document.url = "url1"
    mocked_document.title = "title1"
    mocked_document.text = "text1"

    mocked_document2 = Mock()
    mocked_document2.url = "url2"
    state.retrieved_docs = [mocked_document2]
    state.current_doc = mocked_document

    next_node = route_judge_answer(state)
    assert next_node == "generate_answer"
    assert state.current_doc.url == mocked_document2.url


def test_route_judge_answer_not_approved_no_more_docs(state):
    mock_judge = Mock()
    mock_judge.call.return_value = (False, "Reasoning for disapproval")
    state.judge = mock_judge

    mocked_document = Mock()
    mocked_document.url = "url1"
    mocked_document.title = "title1"
    mocked_document.text = "text1"

    state.retrieved_docs = []
    state.current_doc = mocked_document
    next_node = route_judge_answer(state)
    assert next_node == END
