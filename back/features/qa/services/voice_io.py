from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import sounddevice as sd

from ..audio import MicrophoneManager, STTManager, TTSManager

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class VoiceIOService:
    microphone: MicrophoneManager
    stt: STTManager
    tts: TTSManager

    def record_audio(self, *, max_idle_seconds: Optional[float] = None) -> Optional[bytes]:
        self.microphone.start_listening()
        try:
            audio = self.microphone.record_audio(max_idle_seconds=max_idle_seconds)
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

    def speak(
        self,
        text: str,
        save_file: bool = False,
        *,
        warmup: bool = False,
        warmup_duration: float = 0.3,
    ) -> bool:
        if not text or not text.strip():
            return False
        if warmup:
            self._warmup_output_device(warmup_duration)
        return self.tts.speak(text.strip(), save_file=save_file)

    def record_and_transcribe(self, *, max_idle_seconds: Optional[float] = None) -> tuple[Optional[bytes], Optional[str]]:
        audio = self.record_audio(max_idle_seconds=max_idle_seconds)
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

    def _warmup_output_device(self, duration: float) -> None:
        if duration <= 0:
            return
        try:
            sample_rate = int(getattr(self.microphone, "RATE", 16000))
        except Exception:
            sample_rate = 16000
        try:
            frames = max(int(sample_rate * duration), 1)
        except Exception:
            frames = int(16000 * duration)
        if frames <= 0:
            return
        silence = np.zeros((frames, 1), dtype=np.float32)
        try:
            sd.play(silence, samplerate=sample_rate)
            sd.wait()
        except Exception:
            logger.debug("출력 장치 워밍업에 실패했지만 계속 진행합니다.")


def create_voice_service(
    *, preferred_tts_engine: Optional[str] = None
) -> VoiceIOService:
    mic = MicrophoneManager()
    stt = STTManager()
    tts = TTSManager(preferred_engine=preferred_tts_engine)
    return VoiceIOService(microphone=mic, stt=stt, tts=tts)
