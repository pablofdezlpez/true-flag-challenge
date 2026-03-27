from dataclasses import dataclass
from typing import Optional

from src.Chatbot.agents import AnswerAgent, Judge
from src.Database.retriever import Retriever


@dataclass
class State:
    # Init
    query_text: str
    query_image_bytes: Optional[bytes]

    retriever: Retriever
    judge: Judge
    agent: AnswerAgent

    # RAG
    retrieved_docs: Optional[list[str]] = None
    current_doc: Optional[str] = None

    # Judge results
    judge_approve: bool = False
    judge_reasoning: str = "Did not run the judge yet."

    # Final answer
    answer: str = "After searching for relevant information, I was not able to find any credible sources that address the claim. Therefore, I cannot provide a definitive answer at this moment."


def rag_search(state: State) -> dict:
    """
    Retrieve RETRIEVAL_TOP_K candidate documents.

    If the input contains an image, the image embedding is used for querying.
    If no image results are found, the hierarchical
    text search (summary then chunks) is used.
    """
    print("Running RAG search")
    retriever = state.retriever
    image_query = state.query_image_bytes
    text_query = state.query_text

    if image_query:
        retrieved_docs = retriever.query_by_image(image_query)
    else:
        retrieved_docs = retriever.query_by_text(text_query)

    state.retrieved_docs = retrieved_docs
    state.current_doc = retrieved_docs.pop() if retrieved_docs else None

    return state


def generate_answer(state: State) -> State:
    """
    Run the LLM agent to generate an answer based on the approved document and web search results.
    """
    print("generating answer")

    doc = state.current_doc
    query_text = state.query_text
    query_image = state.query_image_bytes
    answer = state.agent.call(query_text, doc, query_image)
    state.answer = answer
    print(f"Intermidiate answer generated: {answer}")
    return state


def judge_answer(state: State) -> State:
    print("judging answer")

    answer = state.answer
    document = state.current_doc
    user_query = state.query_text
    user_query_image = state.query_image_bytes

    approve, reasoning = state.judge.call(
        user_query, evidence=document, answer=answer, query_image=user_query_image
    )

    state.judge_approve = approve
    state.judge_reasoning = reasoning
    return state


def where_documents_found_retrieved(state: State) -> str:
    """If we have candidate docs, start evaluating; otherwise go to no_answer."""
    if state.current_doc:
        return True
    return False


def have_judge_approved(state: State) -> dict:
    """
    Ask the judge whether the *current_doc* answers the user's query.

    The judge evaluates a single document at a time.
    """
    approve = state.judge_approve
    reasoning = state.judge_reasoning

    print(f"Judge decision: {'APPROVE' if approve else 'REJECT'} - {reasoning}")
    if approve:
        return True
    try:
        next_doc = state.retrieved_docs.pop()
        state.current_doc = next_doc
        return False
    except IndexError:
        state.answer = "After searching for relevant information, I was not able to find any credible sources that address the claim. Therefore, I cannot provide a definitive answer at this moment."
        return True
