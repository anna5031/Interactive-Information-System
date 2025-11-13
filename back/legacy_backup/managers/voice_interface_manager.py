"""마이크 입력과 TTS 출력을 통합 관리."""

import os
import logging
from typing import Optional

from .microphone_manager import MicrophoneManager
from .tts_manager import TTSManager
from ..audio_service import VoiceIOService

logger = logging.getLogger(__name__)


class VoiceInterfaceManager:
    """마이크 입력과 TTS 출력을 연결하는 관리자."""

    def __init__(self, preferred_tts_engine: Optional[str] = None):
        """필요한 관리자들을 초기화하고 준비 상태를 확인."""
        self.mic_manager = MicrophoneManager()

        engine_name = (
            preferred_tts_engine if preferred_tts_engine is not None else "auto"
        )
        self.tts_manager = TTSManager(engine_name)

        self.is_initialized = False
        self.audio_service: Optional[VoiceIOService] = None

        logger.info("Voice interface initialisation started")

        # 시스템 초기화
        self._initialize_system()

    def _initialize_system(self) -> bool:
        """마이크와 TTS 하위 시스템을 초기화."""
        try:
            # 마이크 설정
            mic_setup = self.mic_manager.setup_microphone()
            if not mic_setup:
                logger.warning("Microphone setup failed")
                return False

            # TTS 사용 가능 여부 확인
            tts_available = self.tts_manager.is_available()
            if not tts_available:
                logger.warning("TTS subsystem unavailable")
                return False

            self.is_initialized = True
            logger.info("Voice interface ready")

            # 시스템 정보 출력
            self.audio_service = VoiceIOService(
                microphone=self.mic_manager,
                stt=self.stt_manager,
                tts=self.tts_manager,
            )

            self._print_system_info()

            return True

        except Exception as exc:
            logger.error("Voice interface initialisation failed: %s", exc)
            return False

    def _print_system_info(self):
        """시스템 정보 출력"""
        print("🎤🗣️  음성 인터페이스 시스템")
        print("=" * 40)

        # 마이크 정보
        if self.mic_manager.preferred_device_name:
            print(f"🎤 마이크: {self.mic_manager.preferred_device_name}")
        else:
            print("🎤 마이크: 기본 디바이스")

        effective_rate = int(self.mic_manager.get_stream_sample_rate())
        target_rate = int(self.mic_manager.RATE)
        if self.mic_manager.has_samplerate_fallback():
            print(f"   ↳ 장치 샘플레이트 {effective_rate}Hz → STT {target_rate}Hz (재샘플링)")
        else:
            print(f"   ↳ 사용 샘플레이트: {effective_rate}Hz")

        # TTS 정보
        tts_info = self.tts_manager.get_current_engine_info()
        if tts_info:
            engine_label = f"{tts_info['name']} ({tts_info['engine_id']})"
            print(f"🗣️  TTS: {engine_label}")
        else:
            print("🗣️  TTS: 사용 불가")

        # 사용 가능한 TTS 엔진들
        available_engines = self.tts_manager.get_available_engines()
        if len(available_engines) > 1:
            print(f"🔧 사용 가능한 TTS 엔진 ID: {', '.join(available_engines)}")

        print()

    def is_ready(self) -> bool:
        """시스템 준비 상태 확인"""
        return self.is_initialized and self.tts_manager.is_available()

    def test_microphone(self, duration: float) -> bool:
        """
        마이크 테스트

        Args:
            duration: 테스트 시간 (초)

        Returns:
            bool: 테스트 성공 여부
        """
        if not self.is_initialized:
            logger.error("Voice interface has not been initialised")
            return False

        print(f"🧪 마이크 테스트 ({duration}초간 말씀해보세요)")
        return self.mic_manager.test_microphone(duration)

    def test_tts(self, text: str) -> bool:
        """
        TTS 테스트

        Args:
            text: 테스트할 텍스트

        Returns:
            bool: 테스트 성공 여부
        """
        if not self.is_ready():
            logger.error("Voice interface is not ready")
            return False

        print(f"🧪 TTS 테스트: {text}")
        return self.tts_manager.test_priority_chain(text)

    def test_full_system(self) -> bool:
        """
        전체 시스템 테스트 (마이크 + TTS)

        Returns:
            bool: 테스트 성공 여부
        """
        if not self.is_ready():
            logger.error("Voice interface is not ready")
            return False

        print("🔄 전체 시스템 테스트")
        print("=" * 30)

        # 1. 마이크 테스트
        print("1️⃣ 마이크 테스트:")
        mic_result = self.test_microphone(3.0)

        if mic_result:
            print("   ✅ 마이크 테스트 성공")
        else:
            print("   ❌ 마이크 테스트 실패")

        # 2. TTS 테스트
        print("\n2️⃣ TTS 테스트:")
        tts_result = self.test_tts("음성 인터페이스 시스템이 정상적으로 작동합니다.")

        if tts_result:
            print("   ✅ TTS 테스트 성공")
        else:
            print("   ❌ TTS 테스트 실패")

        # 결과
        overall_success = mic_result and tts_result
        print(f"\n📊 전체 테스트 결과: {'✅ 성공' if overall_success else '❌ 실패'}")

        return overall_success

    def speak(self, text: str, save_file: bool = False) -> bool:
        """
        텍스트를 음성으로 변환하고 재생

        Args:
            text: 변환할 텍스트
            save_file: 파일로 저장할지 여부

        Returns:
            bool: 성공 여부
        """
        if not self.is_ready():
            logger.error("TTS subsystem is not ready")
            return False

        if self.audio_service:
            return self.audio_service.speak(text, save_file=save_file)
        return self.tts_manager.speak(text, save_file)

    def listen_and_record(self) -> Optional[bytes]:
        """
        음성 입력 대기 및 녹음

        Returns:
            bytes: 녹음된 오디오 데이터 또는 None
        """
        if not self.is_initialized:
            logger.error("Microphone subsystem is not initialised")
            return None

        if self.audio_service:
            return self.audio_service.record_audio()

        self.mic_manager.start_listening()
        try:
            audio_data = self.mic_manager.record_audio()
            return audio_data
        finally:
            self.mic_manager.stop_listening()

    def get_audio_service(self) -> VoiceIOService:
        if not self.audio_service:
            raise RuntimeError("VoiceIOService is not initialised.")
        return self.audio_service

    def save_audio(
        self, audio_data: bytes, filename: Optional[str] = None
    ) -> Optional[str]:
        """
        오디오 데이터를 파일로 저장

        Args:
            audio_data: 저장할 오디오 데이터
            filename: 파일명 (None이면 자동 생성)

        Returns:
            str: 저장된 파일 경로 또는 None
        """
        if not audio_data:
            return None

        if not filename:
            import time

            timestamp = int(time.time())
            filename = f"recorded_audio_{timestamp}.wav"

        try:
            self.mic_manager.save_audio_to_wav(audio_data, filename)
            return os.path.abspath(filename)
        except Exception as exc:
            logger.error("Failed to persist recorded audio: %s", exc)
            return None

    def switch_tts_engine(self, engine_name: str) -> bool:
        """
        TTS 엔진 변경

        Args:
            engine_name: 변경할 엔진명

        Returns:
            bool: 변경 성공 여부
        """
        return self.tts_manager.switch_engine(engine_name)

    def get_system_status(self) -> dict:
        """시스템 상태 정보 반환"""
        mic_devices, recommended = self.mic_manager.get_audio_devices()
        tts_info = self.tts_manager.get_current_engine_info()
        available_engines = self.tts_manager.get_available_engines()

        return {
            "initialized": self.is_initialized,
            "ready": self.is_ready(),
            "microphone": {
                "device_name": self.mic_manager.preferred_device_name,
                "device_index": self.mic_manager.input_device_index,
                "available_devices": len(mic_devices),
                "recommended_device": recommended,
                "target_sample_rate": int(self.mic_manager.RATE),
                "stream_sample_rate": int(self.mic_manager.get_stream_sample_rate()),
                "samplerate_fallback": self.mic_manager.has_samplerate_fallback(),
            },
            "tts": {
                "current_engine": tts_info["name"] if tts_info else None,
                "available_engines": available_engines,
                "engine_count": len(available_engines),
            },
        }

    def cleanup(self):
        """리소스 정리"""
        try:
            self.mic_manager.cleanup()
            self.tts_manager.cleanup()
            logger.info("음성 인터페이스 관리자 정리 완료")
        except Exception as e:
            logger.error(f"리소스 정리 실패: {e}")

    def __del__(self):
        """소멸자"""
        self.cleanup()
