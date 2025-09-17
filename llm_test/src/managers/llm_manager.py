"""구성 기반으로 Groq 채팅을 제어하는 관리자."""

import copy
import logging
import os
import sys
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

logger = logging.getLogger(__name__)


CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config")
if CONFIG_PATH not in sys.path:
    sys.path.insert(0, CONFIG_PATH)
try:
    from llm_config import LLM_CONFIG as USER_LLM_CONFIG
except ImportError as exc:
    raise RuntimeError("config/llm_config.py 를 불러올 수 없습니다.") from exc


def _load_llm_config() -> dict:
    """LLM 설정을 안전하게 복사해 반환."""
    if not isinstance(USER_LLM_CONFIG, dict):
        raise RuntimeError("llm_config.LLM_CONFIG 형식이 올바르지 않습니다.")

    return copy.deepcopy(USER_LLM_CONFIG)


class LLMManager:
    def __init__(self):
        """LLM 관리자 초기화"""
        load_dotenv()

        self.llm_config = _load_llm_config()

        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.llm_model = self._resolve_model_name("llm_model")
        self.chat_parameters = self._resolve_chat_parameters()

        self.system_prompt = self._load_system_prompt()

        logger.info("LLM 초기화 완료 (model=%s)", self.llm_model)

    def _resolve_model_name(self, config_key: str) -> str:
        """설정된 LLM 모델 이름을 확인해 반환."""
        if config_key not in self.llm_config:
            raise RuntimeError(f"llm_config 설정에 '{config_key}' 키가 없습니다.")

        config_value = self.llm_config[config_key]
        if not isinstance(config_value, str) or not config_value.strip():
            raise RuntimeError(
                f"llm_config 설정 '{config_key}' 값이 올바르지 않습니다."
            )

        resolved = config_value.strip()
        logger.info("LLM 모델 설정: %s", resolved)
        return resolved

    def _resolve_chat_parameters(self) -> dict:
        if "chat_parameters" not in self.llm_config:
            raise RuntimeError("llm_config 설정에 'chat_parameters' 키가 없습니다.")

        params = self.llm_config["chat_parameters"]
        if not isinstance(params, dict):
            raise RuntimeError("llm_config.chat_parameters 설정이 올바르지 않습니다.")

        required = {
            "temperature",
            "max_tokens",
            "top_p",
            "presence_penalty",
            "frequency_penalty",
            "stream",
        }
        missing = required - params.keys()
        if missing:
            raise RuntimeError(
                f"llm_config.chat_parameters 설정에 누락된 키가 있습니다: {', '.join(sorted(missing))}"
            )

        temperature = params["temperature"]
        max_tokens = params["max_tokens"]
        top_p = params["top_p"]
        presence_penalty = params["presence_penalty"]
        frequency_penalty = params["frequency_penalty"]
        stream = params["stream"]

        if not isinstance(temperature, (int, float)):
            raise RuntimeError(
                "llm_config.chat_parameters.temperature 값이 숫자가 아닙니다."
            )
        if not isinstance(max_tokens, int) or max_tokens <= 0:
            raise RuntimeError(
                "llm_config.chat_parameters.max_tokens 값이 양의 정수가 아닙니다."
            )
        if not isinstance(top_p, (int, float)):
            raise RuntimeError("llm_config.chat_parameters.top_p 값이 숫자가 아닙니다.")
        if not isinstance(presence_penalty, (int, float)):
            raise RuntimeError(
                "llm_config.chat_parameters.presence_penalty 값이 숫자가 아닙니다."
            )
        if not isinstance(frequency_penalty, (int, float)):
            raise RuntimeError(
                "llm_config.chat_parameters.frequency_penalty 값이 숫자가 아닙니다."
            )
        if not isinstance(stream, bool):
            raise RuntimeError(
                "llm_config.chat_parameters.stream 값이 boolean이 아닙니다."
            )

        resolved = {
            "temperature": float(temperature),
            "max_tokens": int(max_tokens),
            "top_p": float(top_p),
            "presence_penalty": float(presence_penalty),
            "frequency_penalty": float(frequency_penalty),
            "stream": stream,
        }
        logger.info(
            "LLM 채팅 파라미터 로드 완료 (temperature=%s, max_tokens=%s, top_p=%s)",
            resolved["temperature"],
            resolved["max_tokens"],
            resolved["top_p"],
        )
        return resolved

    def generate_response(
        self,
        user_text: str,
        conversation_history: Optional[list] = None,
    ) -> Optional[str]:
        """채팅 완성 요청을 보내고 모델 응답을 반환."""
        try:
            # 메시지 구성
            messages = [{"role": "system", "content": self.system_prompt}]

            # 대화 히스토리 추가 (선택적)
            if conversation_history:
                messages.extend(conversation_history)

            # 현재 사용자 메시지 추가
            messages.append({"role": "user", "content": user_text})

            # Groq API 호출
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=self.chat_parameters["temperature"],
                max_tokens=self.chat_parameters["max_tokens"],
                top_p=self.chat_parameters["top_p"],
                presence_penalty=self.chat_parameters["presence_penalty"],
                frequency_penalty=self.chat_parameters["frequency_penalty"],
                stream=self.chat_parameters["stream"],
            )

            response_text = completion.choices[0].message.content
            logger.info(f"AI 응답 생성 완료: {response_text[:100]}...")
            return response_text

        except Exception as exc:
            logger.error("응답 생성 실패: %s", exc)
            raise

    def check_exit_command(self, text: str) -> bool:
        """종료 키워드가 포함되면 True를 반환."""
        exit_keywords = {
            "quit",
            "exit",
            "종료",
            "끝",
            "그만",
            "나가기",
            "시스템 종료",
            "종료해",
            "끝내",
            "그만해",
        }

        normalized = text.lower().strip()
        return any(keyword in normalized for keyword in exit_keywords)

    def set_system_prompt(self, new_prompt: str):
        """시스템 프롬프트 변경"""
        self.system_prompt = new_prompt
        logger.info("시스템 프롬프트가 변경되었습니다.")

    def set_llm_model(self, llm_model: str):
        """사용할 LLM 모델 변경"""
        if llm_model:
            self.llm_model = llm_model
            logger.info("LLM 모델 변경: %s", llm_model)

    def test_connection(self) -> bool:
        """Groq API 연결 테스트"""
        try:
            # 간단한 API 호출로 연결 테스트
            test_completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )

            logger.info("✅ Groq API 연결 테스트 성공")
            return True

        except Exception as e:
            logger.error(f"❌ Groq API 연결 테스트 실패: {e}")
            return False

    def get_current_config(self) -> dict:
        """현재 설정 정보 반환"""
        return {
            "llm_model": self.llm_model,
            "chat_parameters": self.chat_parameters.copy(),
            "system_prompt": self.system_prompt,
        }

    def _load_system_prompt(self) -> str:
        """시스템 프롬프트 파일 로드"""
        current_dir = os.path.dirname(os.path.abspath(__file__))
        prompt_path = os.path.join(current_dir, "..", "..", "config", "base_prompt.md")

        try:
            with open(prompt_path, "r", encoding="utf-8") as file:
                content = file.read()
        except Exception as exc:
            raise RuntimeError(
                f"시스템 프롬프트 파일을 읽을 수 없습니다: {prompt_path}"
            ) from exc

        lines = [line.strip() for line in content.splitlines()]
        prompt_parts = [
            line
            for line in lines
            if line and not line.startswith("#") and not line.startswith("-")
        ]

        if not prompt_parts:
            raise RuntimeError("시스템 프롬프트 파일에 사용할 수 있는 내용이 없습니다.")

        resolved_prompt = " ".join(prompt_parts)
        logger.info("시스템 프롬프트 로드 완료 (길이 %d)", len(resolved_prompt))
        return resolved_prompt
