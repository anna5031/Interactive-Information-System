from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List

from langchain_groq import ChatGroq
from pydantic import BaseModel, Field, ValidationError


class AnswerOutput(BaseModel):
    final_answer: str = Field(..., description="사용자에게 전달할 답변")
    reasoning_steps: List[str] = Field(default_factory=list)
    cites: List[str] = Field(default_factory=list)
    needs_navigation: bool = Field(False, description="경로 안내 필요 여부")
    navigation_trigger: str | None = Field(default=None, description="경로 안내를 요구한 이유")


@dataclass(slots=True)
class AnswerLLMConfig:
    model_name: str = "openai/gpt-oss-120b"


class AnswerLLM:
    def __init__(self, config: AnswerLLMConfig | None = None) -> None:
        self.config = config or AnswerLLMConfig()
        self.client = ChatGroq(model_name=self.config.model_name, temperature=0.2)

    def generate(self, question: str, documents: List[str], indoor_map: dict, conversation_digest: str) -> AnswerOutput:
        prompt = f"""
당신은 기계공학과 안내 도우미입니다. 아래 규칙을 따르세요.
1. 검색 문서와 실내 지도 정보를 참고하여 질문에 답합니다.
2. 사실 기반으로만 답하고, 부족하면 솔직히 모른다고 답합니다.
3. 답변 마지막에는 반드시 "추가 질문이 있다면 물어봐 주세요." 문장을 덧붙입니다.
4. 경로 안내가 필요하거나 사용자가 위치를 물었다면 needs_navigation=true, navigation_trigger에 이유를 적습니다.
5. 아래 JSON 형식으로만 응답하세요:
{{
  "final_answer": string,
  "reasoning_steps": string[],
  "cites": string[],
  "needs_navigation": true/false,
  "navigation_trigger": string|null
}}

[Conversation]
{conversation_digest}

[Question]
{question}

[Retrieved Documents]
{chr(10).join(documents) if documents else 'None'}

[Indoor Map]
{indoor_map}
"""
        response = self.client.invoke(prompt)
        text = self._extract_content(response)
        payload = self._parse_json(text)
        try:
            return AnswerOutput(**payload)
        except ValidationError:
            return AnswerOutput(
                final_answer="죄송합니다. 답변을 생성하는 중 오류가 발생했습니다. 추가 질문이 있다면 물어봐 주세요.",
                reasoning_steps=[],
                cites=[],
                needs_navigation=False,
                navigation_trigger=None,
            )

    @staticmethod
    def _extract_content(message) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, list):
            combined = []
            for part in content:
                if isinstance(part, dict) and "text" in part:
                    combined.append(part["text"])
                elif isinstance(part, str):
                    combined.append(part)
            return "".join(combined)
        if isinstance(content, str):
            return content
        return str(content)

    @staticmethod
    def _parse_json(text: str) -> dict:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = stripped.strip("`")
            if stripped.startswith("json"):
                stripped = stripped[4:]
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            return {}
