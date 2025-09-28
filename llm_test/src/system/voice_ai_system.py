"""음성 AI 시스템 본체 구현."""

import logging

from src.managers.device_manager import DeviceManager
from src.managers.llm_manager import LLMManager
from src.managers.stt_manager import STTManager
from src.managers.voice_interface_manager import VoiceInterfaceManager

logger = logging.getLogger(__name__)

class VoiceAISystem:
    def __init__(self):
        """음성 AI 시스템을 구성하는 관리자 초기화."""
        self.device_manager = DeviceManager()
        self.voice_manager = VoiceInterfaceManager()
        self.stt_manager = STTManager()
        self.llm_manager = LLMManager()

        self.is_running = False
        self.conversation_history: list[dict] = []

        logger.info("음성 AI 시스템 초기화 완료")

    def setup_audio_devices(self) -> bool:
        """오디오 디바이스를 준비 상태로 만든다."""
        logger.info("🔧 음성 인터페이스 설정 중...")

        if self.voice_manager.is_ready():
            logger.info("✅ 음성 인터페이스 설정 완료")
            return True

        logger.warning("⚠️ 음성 인터페이스 설정에 문제가 있습니다.")
        return False

    def test_system(self) -> bool:
        """주요 구성 요소 연결 상태를 점검."""
        logger.info("🧪 시스템 기본 테스트 시작...")

        try:
            stt_info = self.stt_manager.get_current_config()
            logger.info("사용 중인 STT 모델: %s", stt_info["model"])
        except Exception as exc:
            logger.error("STT 설정 확인 실패: %s", exc)
            return False

        if not self.llm_manager.test_connection():
            logger.error("Groq API 연결 실패")
            return False

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
        """사용자와의 대화 루프를 실행."""
        self.is_running = True

        print("\n" + "=" * 60)
        print("🤖 음성 AI 시스템이 시작되었습니다!")
        print("종료하려면 'quit', 'exit', '종료'라고 말하거나 Ctrl+C를 누르세요.")
        print("=" * 60 + "\n")

        self.voice_manager.speak("음성 AI 시스템이 준비되었습니다. 무엇을 도와드릴까요?")

        try:
            while self.is_running:
                print("🎤 음성을 기다리는 중...")
                audio_data = self.voice_manager.listen_and_record()

                if not audio_data:
                    print("❌ 음성 녹음 실패. 다시 시도하세요.")
                    continue

                print("🔄 음성 처리 중...")

                audio_format = {
                    "channels": self.voice_manager.mic_manager.CHANNELS,
                    "sample_width": 2,
                    "frame_rate": self.voice_manager.mic_manager.RATE,
                }

                user_text = self.stt_manager.transcribe(audio_data, audio_format)

                if not user_text:
                    print("❌ 음성 인식에 실패했습니다.")
                    continue

                print(f"👤 사용자: {user_text}")

                if self.llm_manager.check_exit_command(user_text):
                    farewell = "안녕히 가세요!"
                    print(f"🤖 AI: {farewell}")
                    self.voice_manager.speak(farewell)
                    break

                ai_response = self.llm_manager.generate_response(
                    user_text, self.conversation_history
                )

                if ai_response:
                    response_text = ai_response["text"]
                    response_type = ai_response["type"]

                    logger.info("LLM 응답 type=%s", response_type)

                    match response_type:
                        case "map":
                            # TODO: 지도 관련 응답의 추가 처리 로직 구현
                            logger.info("TODO: 지도 관련 응답 후속 처리")
                        case "info":
                            # TODO: 정보형 응답을 기반으로 한 화면/UI 연동 구현
                            logger.info("TODO: 정보형 응답 후속 처리")
                        case "clarify":
                            # TODO: 추가 질문이 필요한 경우 사용자 상호작용 설계
                            logger.info("TODO: 추가 질문 응답 후속 처리")
                        case _:
                            logger.warning("알 수 없는 응답 type=%s", response_type)

                    print(f"🤖 AI ({response_type}): {response_text}")

                    self.conversation_history.extend(
                        [
                            {"role": "user", "content": user_text},
                            {"role": "assistant", "content": response_text},
                        ]
                    )

                    if len(self.conversation_history) > 10:
                        self.conversation_history = self.conversation_history[-10:]

                    print("🔊 음성으로 변환 중...")
                    success = self.voice_manager.speak(response_text)

                    if not success:
                        print("⚠️ 음성 재생에 실패했습니다.")

                print("-" * 60)

        except KeyboardInterrupt:
            print("\n👋 사용자에 의해 종료되었습니다.")
        except Exception as exc:
            logger.error("메인 루프 오류: %s", exc)
        finally:
            self.stop_system()

    def stop_system(self):
        """대화 루프를 종료 상태로 전환."""
        logger.info("시스템 정지 중...")
        self.is_running = False

    def cleanup(self):
        """사용한 리소스를 정리."""
        try:
            self.voice_manager.cleanup()
            logger.info("✅ 리소스 정리 완료")
        except Exception as exc:
            logger.warning("리소스 정리 중 오류: %s", exc)

    def get_system_status(self) -> dict:
        """현재 시스템 상태 정보를 반환."""
        voice_status = self.voice_manager.get_system_status()

        return {
            "is_running": self.is_running,
            "conversation_count": len(self.conversation_history) // 2,
            "device_summary": self.device_manager.get_device_summary(),
            "stt_config": self.stt_manager.get_current_config(),
            "llm_config": self.llm_manager.get_current_config(),
            "voice_system": voice_status,
        }
