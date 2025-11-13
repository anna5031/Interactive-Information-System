from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from ..audio import MicrophoneManager, STTManager, TTSManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VoiceIOService:
    microphone: MicrophoneManager
    stt: STTManager
    tts: TTSManager

    def record_audio(self) -> Optional[bytes]:
        self.microphone.start_listening()
        try:
            audio = self.microphone.record_audio()
            if audio:
                logger.info("녹음 길이: %d bytes", len(audio))
            else:
                logger.warning("녹음 데이터를 얻지 못했습니다.")
            return audio
        finally:
            self.microphone.stop_listening()

    def transcribe(self, audio: bytes) -> Optional[str]:
        if not audio:
            return None
        return self.stt.transcribe(audio, self.stt.get_audio_format())

    def speak(self, text: str, save_file: bool = False) -> bool:
        if not text or not text.strip():
            return False
        return self.tts.speak(text.strip(), save_file=save_file)

    def record_and_transcribe(self) -> tuple[Optional[bytes], Optional[str]]:
        audio = self.record_audio()
        if audio:
            text = self.transcribe(audio)
        else:
            text = None
        return audio, text

    def close(self) -> None:
        try:
            self.microphone.stop_listening()
        except Exception:
            pass
        self.tts.cleanup()


def create_voice_service(
    *, preferred_tts_engine: Optional[str] = None
) -> VoiceIOService:
    mic = MicrophoneManager()
    stt = STTManager()
    tts = TTSManager(preferred_engine=preferred_tts_engine)
    return VoiceIOService(microphone=mic, stt=stt, tts=tts)
