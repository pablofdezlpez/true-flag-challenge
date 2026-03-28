from src.Chatbot.agents import AnswerAgent, Judge


def test_judge_response_correct_types():
    judge = Judge(model="gemini-3.1-flash-lite-preview")
    approve, reasoning = judge.call(
        query_text="What is the capital of France?",
        doc={"text": "The capital of France is Paris."},
        answer="The capital of France is Paris.",
    )
    assert isinstance(approve, bool), "Judge approve should be a boolean"
    assert isinstance(reasoning, str), "Judge reasoning should be a string"


def test_answer_agent_calls_tool():
    agent = AnswerAgent(model="gemini-3.1-pro-preview")
    answer = agent.call(
        query_text="What is this library used for?",
        doc={"text": "https://docs.python.org/3/library/unittest.mock.html"},
    )
    assert (
        answer.candidates[0].url_context_metadata.dict()["url_metadata"][0][
            "retrieved_url"
        ]
        == "https://docs.python.org/3/library/unittest.mock.html"
    ), "AnswerAgent should call the url_context tool with the correct URL"
