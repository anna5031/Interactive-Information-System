"""Groq 서비스와 구성 가능한 TTS 엔진으로 동작하는 모듈형 음성 AI 런타임."""

import logging
import os
import sys

from src.system.voice_ai_system import VoiceAISystem

# 로깅 설정
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
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
