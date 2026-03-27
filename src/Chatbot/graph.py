import asyncio
import logging
import sys
import warnings

from langgraph.graph import START, StateGraph

from src.Chatbot.agents import AnswerAgent, Judge
from src.Chatbot.nodes import (
    State,
    generate_answer,
    rag_search,
    route_judge_answer,
    route_after_search,
)
from src.Database.retriever import Retriever
from src.config import ANSWER_AGENT_MODEL, JUDGE_MODEL

judge = Judge(model=JUDGE_MODEL)
answer_agent = AnswerAgent(model=ANSWER_AGENT_MODEL)


def build_graph():
    """Construct and compile the LangGraph pipeline."""
    graph = StateGraph(State)

    # Register nodes
    graph.add_node("rag_search", rag_search)
    graph.add_node("generate_answer", generate_answer)

    # Linear edges
    graph.add_edge(START, "rag_search")

    # Conditional: after search, either start judging or go to no_answer
    graph.add_conditional_edges("rag_search", route_after_search)

    # Conditional: after each document evaluation
    graph.add_conditional_edges("generate_answer", route_judge_answer)

    return graph.compile()


# Module-level singleton – compiled once and reused across Streamlit reruns
pipeline = build_graph()


def run_pipeline(
    query_text: str, query_image_bytes: bytes | None, dataset: str
) -> dict:
    """
    Execute the H-RAG pipeline for a user query.

    At least one of query_text or query_image_bytes must be provided.
    If both are given, the image embedding is used for retrieval first and
    the text is used for the judge and answer generation.

    Returns the `final_answer` dict populated by return_final_answer.
    """
    initial_state: State = {
        "retriever": Retriever(dataset),
        "query_text": query_text,
        "query_image_bytes": query_image_bytes,
        "judge": judge,
        "agent": answer_agent,
    }
    return pipeline.invoke(initial_state)


if __name__ == "__main__":
    # Example usage
    query_text = "What is the capital of France?"
    query_image_bytes = None  # Or load image bytes if testing image queries
    dataset = "chroma_db"
    answer = run_pipeline(query_text, query_image_bytes, dataset)
    print("Final Answer:", answer)
