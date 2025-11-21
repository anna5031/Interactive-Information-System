from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from groq import Groq
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field


class AnswerOutput(BaseModel):
    final_answer: str = Field(..., description="사용자에게 전달할 답변")
    reasoning_steps: List[str] = Field(default_factory=list)
    cites: List[str] = Field(default_factory=list)
    needs_navigation: bool = Field(False, description="경로 안내 필요 여부")
    navigation_trigger: Optional[str] = Field(default=None, description="경로 안내를 요구한 이유")
    destination_room: Optional[str] = Field(default=None, description="경로 안내 목적지 방/구역")


@dataclass(slots=True)
class AnswerLLMConfig:
    model_name: str = "openai/gpt-oss-120b"


ANSWER_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "final_answer": {"type": "string", "description": "사용자에게 전달할 답변"},
        "reasoning_steps": {
            "type": "array",
            "items": {"type": "string"},
            "description": "추론 단계",
        },
        "cites": {"type": "array", "items": {"type": "string"}, "description": "참조"},
        "needs_navigation": {"type": "boolean", "description": "경로 안내 필요 여부"},
        "navigation_trigger": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "경로 안내 사유",
        },
        "destination_room": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "description": "목적지",
        },
    },
    "required": [
        "final_answer",
        "reasoning_steps",
        "cites",
        "needs_navigation",
        "navigation_trigger",
        "destination_room",
    ],
    "additionalProperties": False,
}


class AnswerLLM:
    def __init__(self, config: AnswerLLMConfig | None = None) -> None:
        self.config = config or AnswerLLMConfig()
        self._groq_client = Groq()
        self._chat_client = ChatGroq(model_name=self.config.model_name, temperature=0.2)
        self._response_format = {
            "type": "json_schema",
            "json_schema": {
                "name": "answer_output",
                "strict": True,
                "schema": ANSWER_JSON_SCHEMA,
            },
        }

    def generate(
        self,
        question: str,
        documents: List[str],
        indoor_map: dict,
        conversation_digest: str,
        question_type: str = "INFORMATION",
    ) -> AnswerOutput:
        mode = (question_type or "INFORMATION").upper()
        if mode == "RECOMMENDATION":
            prompt = self._build_recommendation_prompt(question, documents, indoor_map, conversation_digest)
        else:
            prompt = self._build_information_prompt(question, documents, indoor_map, conversation_digest)
        try:
            structured_payload = self._invoke_structured(prompt)
            return AnswerOutput(**structured_payload)
        except Exception:
            try:
                response = self._chat_client.invoke(prompt)
                text = self._extract_content(response)
                payload = self._parse_json(text)
                return AnswerOutput(**payload)
            except Exception:
                return self._build_fallback_answer(documents)

    def _invoke_structured(self, prompt: str) -> dict:
        response = self._groq_client.chat.completions.create(
            model=self.config.model_name,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
            response_format=self._response_format,
        )
        message = response.choices[0].message
        content = getattr(message, "content", "") or ""
        return json.loads(content)

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

    @staticmethod
    def _build_fallback_answer(documents: List[str]) -> AnswerOutput:
        if documents:
            snippet = documents[0][:400]
            final_answer = f"다음 문서를 참고해 주세요:\n{snippet}\n\n추가 질문이 있다면 물어봐 주세요."
        else:
            final_answer = "죄송합니다. 현재 정보를 찾을 수 없습니다. 추가 질문이 있다면 물어봐 주세요."
        return AnswerOutput(
            final_answer=final_answer,
            reasoning_steps=[],
            cites=[],
            needs_navigation=False,
            navigation_trigger=None,
            destination_room=None,
        )

    @staticmethod
    def _format_documents(documents: List[str]) -> str:
        return "\n".join(documents) if documents else "None"

    def _build_information_prompt(
        self,
        question: str,
        documents: List[str],
        indoor_map: dict,
        conversation_digest: str,
    ) -> str:
        docs_block = self._format_documents(documents)
        return f"""
당신은 기계공학과 학생을 도와주는 지식 어시스턴트입니다.
- 검색 문서와 실내 지도 정보를 기반으로 정확하게 답하세요.
- 정보가 부족하면 추측하지 말고 모른다고 답하세요.
- 답변 마지막에 "추가 질문이 있다면 물어봐 주세요."를 덧붙이세요.
- 필요 시 needs_navigation, navigation_trigger, destination_room을 채운 JSON만 응답하세요.

[Conversation]
{conversation_digest or '없음'}

[Question]
{question}

[Retrieved Documents]
{docs_block}

[Indoor Map]
{indoor_map}
"""

    def _build_recommendation_prompt(
        self,
        question: str,
        documents: List[str],
        indoor_map: dict,
        conversation_digest: str,
    ) -> str:
        docs_block = self._format_documents(documents)
        return f"""
당신은 기계공학과 학생의 세미나실·강의 추천 전문가입니다.
- 주어진 후보 목록을 활용해 최고의 선택을 제안하고 각 추천 이유를 설명하세요.
- 가능하면 점수를 언급하거나 근거를 명시하고, 정보가 없으면 솔직히 모른다고 답하세요.
- 답변 형식은 다음을 따릅니다.
  ### 최고 추천
  **이름** (이유)

  ### 대안
  **이름** (이유)

  ### 추가 팁
  ...
- 답변 마지막에 "추가 질문이 있다면 물어봐 주세요."를 덧붙이고 JSON 스키마를 지키세요.

[Conversation]
{conversation_digest or '없음'}

[Question]
{question}

[Recommendation Candidates]
{docs_block}

[Indoor Map]
{indoor_map}
"""
