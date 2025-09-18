"""
LLM 설정 파일
"""

LLM_CONFIG = {
    # 기본 LLM 모델
    # "llm_model": "llama-3.1-8b-instant",
    "llm_model": "openai/gpt-oss-20b",
    # 채팅 생성 파라미터
    "chat_parameters": {
        "temperature": 0.7,
        "max_tokens": 1024,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stream": False,
    },
    "structured_output": {
        "name": "voice_ai_structured_response",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "text": {
                    "type": "string",
                    "description": "자연스러운 한국어 음성 안내 문장",
                },
                "type": {
                    "type": "string",
                    "enum": ["info", "map", "clarify"],
                    "description": "info=일반 정보, map=경로 안내, clarify=추가 질문 필요",
                },
            },
            "required": ["text", "type"],
            "additionalProperties": False,
        },
    },
}
