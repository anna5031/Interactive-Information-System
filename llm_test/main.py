#!/usr/bin/env python3
"""
젯슨 오린 나노 음성 AI 시스템 (모듈화 버전)
Groq API (Whisper + Llama 3 8B) + gTTS 사용
"""

import os
import logging
from dotenv import load_dotenv

# 커스텀 모듈 import
from src.managers.audio_manager import AudioManager
from src.managers.microphone_manager import MicrophoneManager
from src.managers.llm_manager import LLMManager
from src.managers.device_manager import DeviceManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class VoiceAISystem:
    def __init__(self, groq_api_key: str):
        """음성 AI 시스템 초기화"""
        self.groq_api_key = groq_api_key

        # 관리자 모듈 초기화
        self.device_manager = DeviceManager()
        self.audio_manager = AudioManager()
        self.microphone_manager = MicrophoneManager()
        self.llm_manager = LLMManager(groq_api_key)

        # 상태 변수
        self.is_running = False
        self.conversation_history = []

        logger.info("음성 AI 시스템 초기화 완료")

    def setup_audio_devices(self) -> bool:
        """오디오 디바이스 설정"""
        logger.info("🔧 오디오 디바이스 설정 중...")

        # 1. 최적의 오디오 디바이스 찾기
        best_output, best_input = self.device_manager.find_best_audio_devices()

        # 2. 오디오 출력 설정
        audio_setup_success = False
        if best_output:
            audio_setup_success = self.audio_manager.set_audio_output(best_output)
        else:
            logger.warning("적절한 출력 디바이스를 찾을 수 없습니다.")
            # 기본 USB 스피커 설정 시도
            audio_setup_success = self.audio_manager.setup_preferred_audio_output()

        # 3. 마이크 설정
        mic_setup_success = self.microphone_manager.setup_microphone()

        # 4. 설정 결과 출력
        if audio_setup_success:
            logger.info("✅ 오디오 출력 설정 완료")
        else:
            logger.warning("⚠️ 오디오 출력 설정에 문제가 있을 수 있습니다.")

        if mic_setup_success:
            logger.info("✅ 마이크 설정 완료")
        else:
            logger.warning("⚠️ 마이크 설정에 문제가 있을 수 있습니다.")

        return audio_setup_success and mic_setup_success

    def test_system(self) -> bool:
        """시스템 기본 테스트"""
        logger.info("🧪 시스템 기본 테스트 시작...")

        # Groq API 연결 테스트만 수행
        if not self.llm_manager.test_connection():
            logger.error("Groq API 연결 실패")
            return False

        logger.info("✅ 시스템 기본 테스트 완료")
        return True

    def run_conversation_loop(self):
        """메인 대화 루프"""
        self.is_running = True
        self.microphone_manager.start_listening()

        print("\n" + "="*60)
        print("🤖 음성 AI 시스템이 시작되었습니다!")
        print("종료하려면 'quit', 'exit', '종료'라고 말하거나 Ctrl+C를 누르세요.")
        print("="*60 + "\n")

        # 시작 안내 음성
        self.audio_manager.text_to_speech_and_play(
            "음성 AI 시스템이 준비되었습니다. 무엇을 도와드릴까요?"
        )

        try:
            while self.is_running:
                # 1. 음성 녹음
                print("🎤 음성을 기다리는 중...")
                audio_data = self.microphone_manager.record_audio()

                if not audio_data:
                    print("❌ 음성 녹음 실패. 다시 시도하세요.")
                    continue

                # 2. 음성 처리 (인식 + 응답 생성)
                print("🔄 음성 처리 중...")

                # 오디오 포맷 정보
                audio_format = {
                    'channels': self.microphone_manager.CHANNELS,
                    'sample_width': 2,  # 16-bit
                    'frame_rate': self.microphone_manager.RATE
                }

                # LLM으로 처리
                user_text, ai_response, should_exit = self.llm_manager.process_voice_input(
                    audio_data, audio_format, self.conversation_history
                )

                if not user_text:
                    print("❌ 음성 인식에 실패했습니다.")
                    continue

                print(f"👤 사용자: {user_text}")

                # 종료 처리
                if should_exit:
                    print(f"🤖 AI: {ai_response}")
                    self.audio_manager.text_to_speech_and_play(ai_response)
                    break

                # 3. AI 응답 출력 및 재생
                if ai_response:
                    print(f"🤖 AI: {ai_response}")

                    # 대화 히스토리에 추가
                    self.conversation_history.extend([
                        {"role": "user", "content": user_text},
                        {"role": "assistant", "content": ai_response}
                    ])

                    # 히스토리 크기 제한 (최근 10개 메시지만 유지)
                    if len(self.conversation_history) > 10:
                        self.conversation_history = self.conversation_history[-10:]

                    # 4. 음성 합성 및 재생
                    print("🔊 음성으로 변환 중...")
                    success = self.audio_manager.text_to_speech_and_play(ai_response)

                    if not success:
                        print("⚠️ 음성 재생에 실패했습니다.")

                print("-" * 60)

        except KeyboardInterrupt:
            print("\n👋 사용자에 의해 종료되었습니다.")
        except Exception as e:
            logger.error(f"메인 루프 오류: {e}")
        finally:
            self.stop_system()

    def stop_system(self):
        """시스템 정지"""
        logger.info("시스템 정지 중...")
        self.is_running = False
        self.microphone_manager.stop_listening()

    def cleanup(self):
        """리소스 정리"""
        try:
            self.microphone_manager.cleanup()
            self.audio_manager.cleanup()
            logger.info("✅ 리소스 정리 완료")
        except Exception as e:
            logger.warning(f"리소스 정리 중 오류: {e}")

    def get_system_status(self) -> dict:
        """시스템 상태 정보 반환"""
        return {
            'is_running': self.is_running,
            'conversation_count': len(self.conversation_history) // 2,
            'device_summary': self.device_manager.get_device_summary(),
            'llm_config': self.llm_manager.get_current_config(),
            'audio_device': self.audio_manager.usb_sink_name,
            'mic_device': self.microphone_manager.preferred_device_name
        }


def main():
    """메인 함수"""
    # .env 파일 로드
    load_dotenv()

    # Groq API 키 가져오기
    groq_api_key = os.getenv("GROQ_API_KEY")

    if not groq_api_key:
        print("❌ Groq API 키가 설정되지 않았습니다.")
        print(".env 파일에 GROQ_API_KEY를 설정하세요.")
        return

    # 음성 AI 시스템 초기화
    voice_ai = VoiceAISystem(groq_api_key)

    try:
        # 1. 오디오 디바이스 설정
        print("🔧 시스템 설정 중...")
        if not voice_ai.setup_audio_devices():
            print("⚠️ 오디오 디바이스 설정에 문제가 있지만 계속 진행합니다.")

        # 2. 시스템 기본 테스트
        if not voice_ai.test_system():
            print("⚠️ 시스템 테스트에서 문제가 발견되었지만 계속 진행합니다.")

        # 3. 시스템 상태 출력
        status = voice_ai.get_system_status()
        print(f"\n📊 시스템 상태:")
        print(f"  - 오디오 디바이스: {status['audio_device'] or '기본'}")
        print(f"  - 마이크 디바이스: {status['mic_device'] or '기본'}")
        print(f"  - LLM 모델: {status['llm_config']['llm_model']}")
        print(f"  - Whisper 모델: {status['llm_config']['whisper_model']}")

        # 4. 대화 루프 실행
        voice_ai.run_conversation_loop()

    except Exception as e:
        logger.error(f"시스템 실행 중 오류: {e}")
    finally:
        # 5. 리소스 정리
        voice_ai.cleanup()


if __name__ == "__main__":
    main()