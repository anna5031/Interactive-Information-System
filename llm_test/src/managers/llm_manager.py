#!/usr/bin/env python3
"""
LLM 관리 모듈 - Groq API를 통한 음성 인식 및 응답 생성
"""

import os
import tempfile
import wave
import logging
from groq import Groq
from typing import Optional

logger = logging.getLogger(__name__)


class LLMManager:
    def __init__(self, api_key: str):
        """LLM 관리자 초기화"""
        self.client = Groq(api_key=api_key)
        self.api_key = api_key

        # 모델 설정
        self.whisper_model = "whisper-large-v3-turbo"
        self.llm_model = "llama-3.1-8b-instant"

        # 시스템 프롬프트 로드
        self.system_prompt = self._load_system_prompt()

        # 오디오 설정
        self.audio_format = {
            'channels': 1,
            'sample_width': 2,  # 16-bit
            'frame_rate': 16000
        }

        logger.info("LLM 관리자 초기화 완료")

    def transcribe_audio(self, audio_data: bytes, audio_format: dict = None) -> Optional[str]:
        """Groq Whisper를 사용한 음성 인식"""
        try:
            # 오디오 포맷 설정
            if audio_format is None:
                audio_format = self.audio_format

            # 임시 WAV 파일 생성
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
                with wave.open(temp_file.name, 'wb') as wav_file:
                    wav_file.setnchannels(audio_format['channels'])
                    wav_file.setsampwidth(audio_format['sample_width'])
                    wav_file.setframerate(audio_format['frame_rate'])
                    wav_file.writeframes(audio_data)

                # Groq Whisper API 호출
                with open(temp_file.name, "rb") as file:
                    transcription = self.client.audio.transcriptions.create(
                        file=file,
                        model=self.whisper_model,
                        language="ko",  # 한국어 설정
                        temperature=0.0  # 일관된 결과를 위해 낮은 temperature
                    )

                # 임시 파일 삭제
                os.unlink(temp_file.name)

                transcribed_text = transcription.text.strip()
                logger.info(f"음성 인식 결과: {transcribed_text}")
                return transcribed_text

        except Exception as e:
            logger.error(f"음성 인식 오류: {e}")
            return None

    def generate_response(self, user_text: str, conversation_history: list = None) -> Optional[str]:
        """Groq Llama를 사용한 응답 생성"""
        try:
            # 메시지 구성
            messages = [
                {
                    "role": "system",
                    "content": self.system_prompt
                }
            ]

            # 대화 히스토리 추가 (선택적)
            if conversation_history:
                messages.extend(conversation_history)

            # 현재 사용자 메시지 추가
            messages.append({
                "role": "user",
                "content": user_text
            })

            # Groq API 호출
            completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=messages,
                temperature=0.7,
                max_tokens=1024,
                top_p=1,
                stream=False
            )

            response_text = completion.choices[0].message.content
            logger.info(f"AI 응답 생성 완료: {response_text[:100]}...")
            return response_text

        except Exception as e:
            logger.error(f"응답 생성 오류: {e}")
            return "죄송합니다. 응답을 생성하는 중 오류가 발생했습니다."

    def check_exit_command(self, text: str) -> bool:
        """종료 명령어 확인"""
        exit_keywords = [
            'quit', 'exit', '종료', '끝', '그만', '나가기',
            '시스템 종료', '종료해', '끝내', '그만해'
        ]

        text_lower = text.lower().strip()
        return any(keyword in text_lower for keyword in exit_keywords)

    def process_voice_input(self, audio_data: bytes, audio_format: dict = None,
                          conversation_history: list = None) -> tuple[Optional[str], Optional[str], bool]:
        """음성 입력을 처리하여 텍스트와 응답 반환

        Returns:
            tuple: (transcribed_text, ai_response, should_exit)
        """
        # 1. 음성 인식
        transcribed_text = self.transcribe_audio(audio_data, audio_format)

        if not transcribed_text:
            return None, None, False

        # 2. 종료 명령어 확인
        if self.check_exit_command(transcribed_text):
            return transcribed_text, "안녕히 가세요!", True

        # 3. AI 응답 생성
        ai_response = self.generate_response(transcribed_text, conversation_history)

        return transcribed_text, ai_response, False

    def set_system_prompt(self, new_prompt: str):
        """시스템 프롬프트 변경"""
        self.system_prompt = new_prompt
        logger.info("시스템 프롬프트가 변경되었습니다.")

    def set_models(self, whisper_model: str = None, llm_model: str = None):
        """사용할 모델 변경"""
        if whisper_model:
            self.whisper_model = whisper_model
            logger.info(f"Whisper 모델 변경: {whisper_model}")

        if llm_model:
            self.llm_model = llm_model
            logger.info(f"LLM 모델 변경: {llm_model}")

    def test_connection(self) -> bool:
        """Groq API 연결 테스트"""
        try:
            # 간단한 API 호출로 연결 테스트
            test_completion = self.client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {
                        "role": "user",
                        "content": "Hello"
                    }
                ],
                max_tokens=10
            )

            logger.info("✅ Groq API 연결 테스트 성공")
            return True

        except Exception as e:
            logger.error(f"❌ Groq API 연결 테스트 실패: {e}")
            return False

    def get_available_models(self) -> dict:
        """사용 가능한 모델 목록 반환"""
        return {
            "whisper_models": [
                "whisper-large-v3-turbo",
                "whisper-large-v3"
            ],
            "llm_models": [
                "llama-3.1-8b-instant",
                "llama-3.1-70b-versatile",
                "mixtral-8x7b-32768",
                "gemma-7b-it"
            ]
        }

    def get_current_config(self) -> dict:
        """현재 설정 정보 반환"""
        return {
            "whisper_model": self.whisper_model,
            "llm_model": self.llm_model,
            "system_prompt": self.system_prompt,
            "audio_format": self.audio_format
        }

    def _load_system_prompt(self) -> str:
        """시스템 프롬프트 파일 로드"""
        try:
            # 현재 파일 위치에서 상대 경로로 base_prompt.md 찾기
            current_dir = os.path.dirname(os.path.abspath(__file__))
            prompt_path = os.path.join(current_dir, '..', 'config', 'base_prompt.md')

            with open(prompt_path, 'r', encoding='utf-8') as file:
                content = file.read()

                # 마크다운에서 실제 텍스트 추출 (간단한 방식)
                lines = content.split('\n')
                prompt_parts = []

                for line in lines:
                    line = line.strip()
                    # 헤더나 빈 줄 제외하고 실제 내용만 추출
                    if line and not line.startswith('#') and not line.startswith('-'):
                        prompt_parts.append(line)

                if prompt_parts:
                    return ' '.join(prompt_parts)

        except Exception as e:
            logger.error(f"시스템 프롬프트 파일 로드 실패: {e}")

        # 파일 로드 실패시 간단한 기본 프롬프트
        return "당신은 친근하고 도움이 되는 한국어 AI 어시스턴트입니다."
