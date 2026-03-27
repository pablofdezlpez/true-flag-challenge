from pydantic import BaseModel

from google import genai
from google.genai import types
from google.genai.types import GenerateContentConfig
from src.Chatbot.prompts import _ANSWER_JUDGE_SYSTEM, AGENT_SYSTEM_PROMPT


class JudgeAnswer(BaseModel):
    approve: bool
    reasoning: str


class Judge:
    """LLM-based judge that evaluates the quality of retrieved evidence and generated answers."""

    def __init__(self, model: str):
        self.model = model
        self.client = genai.Client()

    def call(
        self,
        query_text: str,
        evidence: str,
        answer: str,
        query_image: bytes | None = None,
    ) -> tuple[bool, str]:
        message = [
            f"QUERY:\n{query_text}\n\nEVIDENCE:\n{evidence}\n\nGENERATED ANSWER:\n{answer}"
        ]

        if query_image:
            message.append(
                types.Part.from_bytes(data=query_image, mime_type="image/jpeg")
            )
        response = self.client.models.generate_content(
            model=self.model,
            contents=message,
            config=GenerateContentConfig(
                system_instruction=_ANSWER_JUDGE_SYSTEM,
                temperature=0,
                response_schema=JudgeAnswer,
            ),
        )
        response = JudgeAnswer.model_validate_json(
            response.candidates[0].content.parts[0].text
        )
        return response.approve, response.reasoning


class AnswerAgent:
    """LLM-based agent that generates answers based on retrieved evidence and user queries."""

    def __init__(self, model: str):
        self.async_browser = None
        self.agent = None
        self.client = genai.Client()
        self.tools = [{"url_context": {}}]
        self.model = model

    def call(
        self, query_text: str, evidence: str, query_image: bytes | None = None
    ) -> str:
        message = [f"QUERY:\n{query_text}\n\nEVIDENCE:\n{evidence}"]
        if query_image:
            message.append(
                types.Part.from_bytes(data=query_image, mime_type="image/jpeg")
            )

        response = self.client.models.generate_content(
            model=self.model,
            contents=message,
            config=GenerateContentConfig(
                tools=self.tools, system_instruction=AGENT_SYSTEM_PROMPT
            ),
        )

        return response.candidates[0].content.parts[0].text
