"""Groq 서비스와 구성 가능한 TTS 엔진으로 동작하는 모듈형 음성 AI 런타임."""

import logging
import os
import sys

# 커스텀 모듈 import
from src.managers.voice_interface_manager import VoiceInterfaceManager
from src.managers.llm_manager import LLMManager
from src.managers.stt_manager import STTManager
from src.managers.device_manager import DeviceManager

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class VoiceAISystem:
    def __init__(self):
        # 관리자 모듈 초기화
        self.device_manager = DeviceManager()
        self.voice_manager = VoiceInterfaceManager()
        self.stt_manager = STTManager()
        self.llm_manager = LLMManager()

        # 상태 변수
        self.is_running = False
        self.conversation_history = []

        logger.info("음성 AI 시스템 초기화 완료")

    def setup_audio_devices(self) -> bool:
        """오디오 디바이스 설정"""
        logger.info("🔧 음성 인터페이스 설정 중...")

        # 음성 인터페이스 시스템 준비 상태 확인
        if self.voice_manager.is_ready():
            logger.info("✅ 음성 인터페이스 설정 완료")
            return True
        else:
            logger.warning("⚠️ 음성 인터페이스 설정에 문제가 있습니다.")
            return False

    def test_system(self) -> bool:
        """시스템 기본 테스트"""
        logger.info("🧪 시스템 기본 테스트 시작...")

        # STT 설정 확인
        try:
            stt_info = self.stt_manager.get_current_config()
            logger.info("사용 중인 STT 모델: %s", stt_info["model"])
        except Exception as exc:
            logger.error("STT 설정 확인 실패: %s", exc)
            return False

        # 1. Groq API 연결 테스트
        if not self.llm_manager.test_connection():
            logger.error("Groq API 연결 실패")
            return False

        # 2. 음성 인터페이스 테스트 (선택적)
        logger.info("음성 시스템 빠른 테스트...")
        voice_test_result = self.voice_manager.test_tts("시스템 테스트 중입니다.")
        if voice_test_result:
            logger.info("✅ 음성 시스템 테스트 성공")
        else:
            logger.warning("⚠️ 음성 시스템 테스트 실패")
            return False

        logger.info("✅ 시스템 기본 테스트 완료")
        return True

    def run_conversation_loop(self):
        """메인 대화 루프"""
        self.is_running = True

        print("\n" + "=" * 60)
        print("🤖 음성 AI 시스템이 시작되었습니다!")
        print("종료하려면 'quit', 'exit', '종료'라고 말하거나 Ctrl+C를 누르세요.")
        print("=" * 60 + "\n")

        # 시작 안내 음성
        self.voice_manager.speak(
            "음성 AI 시스템이 준비되었습니다. 무엇을 도와드릴까요?"
        )

        try:
            while self.is_running:
                # 1. 음성 녹음
                print("🎤 음성을 기다리는 중...")
                audio_data = self.voice_manager.listen_and_record()

                if not audio_data:
                    print("❌ 음성 녹음 실패. 다시 시도하세요.")
                    continue

                # 2. 음성 처리 (인식 + 응답 생성)
                print("🔄 음성 처리 중...")

                # 오디오 포맷 정보
                audio_format = {
                    "channels": self.voice_manager.mic_manager.CHANNELS,
                    "sample_width": 2,  # 16-bit
                    "frame_rate": self.voice_manager.mic_manager.RATE,
                }

                # 음성 인식 (STT)
                user_text = self.stt_manager.transcribe(audio_data, audio_format)

                if not user_text:
                    print("❌ 음성 인식에 실패했습니다.")
                    continue

                print(f"👤 사용자: {user_text}")

                # 종료 처리
                if self.llm_manager.check_exit_command(user_text):
                    farewell = "안녕히 가세요!"
                    print(f"🤖 AI: {farewell}")
                    self.voice_manager.speak(farewell)
                    break

                # 3. AI 응답 출력 및 재생
                ai_response = self.llm_manager.generate_response(
                    user_text, self.conversation_history
                )

                if ai_response:
                    print(f"🤖 AI: {ai_response}")

                    # 대화 히스토리에 추가
                    self.conversation_history.extend(
                        [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": ai_response},
                        ]
                    )

                    # 히스토리 크기 제한 (최근 10개 메시지만 유지)
                    if len(self.conversation_history) > 10:
                        self.conversation_history = self.conversation_history[-10:]

                    # 4. 음성 합성 및 재생
                    print("🔊 음성으로 변환 중...")
                    success = self.voice_manager.speak(ai_response)

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

    def cleanup(self):
        """리소스 정리"""
        try:
            self.voice_manager.cleanup()
            logger.info("✅ 리소스 정리 완료")
        except Exception as e:
            logger.warning(f"리소스 정리 중 오류: {e}")

    def get_system_status(self) -> dict:
        """시스템 상태 정보 반환"""
        voice_status = self.voice_manager.get_system_status()

        return {
            "is_running": self.is_running,
            "conversation_count": len(self.conversation_history) // 2,
            "device_summary": self.device_manager.get_device_summary(),
            "stt_config": self.stt_manager.get_current_config(),
            "llm_config": self.llm_manager.get_current_config(),
            "voice_system": voice_status,
        }


def main():
    """메인 함수"""

    # 음성 AI 시스템 초기화
    voice_ai = VoiceAISystem()

    try:
        # 1. 오디오 디바이스 설정
        print("🔧 시스템 설정 중...")
        if not voice_ai.setup_audio_devices():
            print("⚠️ 오디오 디바이스 설정에 문제가 있지만 계속 진행합니다.")

        # 2. (옵션) 시작 전 셀프 테스트
        enable_startup_tests = (
            os.getenv("STARTUP_TESTS", "").lower() in ("1", "true", "yes")
            or "--self-test" in sys.argv
        )
        if enable_startup_tests:
            if not voice_ai.test_system():
                print("⚠️ 시스템 테스트에서 문제가 발견되었지만 계속 진행합니다.")

        # 3. 시스템 상태 출력
        status = voice_ai.get_system_status()
        print(f"\n📊 시스템 상태:")

        voice_status = status["voice_system"]
        mic_info = voice_status["microphone"]
        tts_info = voice_status["tts"]

        print(
            f"  - 음성 시스템: {'✅ 준비됨' if voice_status['ready'] else '❌ 준비되지 않음'}"
        )
        print(f"  - 마이크: {mic_info['device_name'] or '기본 디바이스'}")
        print(f"  - TTS 엔진: {tts_info['current_engine'] or '없음'}")
        print(f"  - LLM 모델: {status['llm_config']['llm_model']}")
        stt_info = status.get("stt_config", {})
        print(f"  - STT 모델: {stt_info.get('model', '알 수 없음')}")

        if len(tts_info["available_engines"]) > 1:
            print(f"  - 사용 가능한 TTS: {', '.join(tts_info['available_engines'])}")

        # 4. 대화 루프 실행
        voice_ai.run_conversation_loop()

    except Exception as e:
        logger.error(f"시스템 실행 중 오류: {e}")
    finally:
        # 5. 리소스 정리
        voice_ai.cleanup()


if __name__ == "__main__":
    main()
