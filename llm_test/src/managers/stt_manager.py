"""Groq API를 사용하여 구성 기반으로 동작하는 음성 인식 관리자."""

import copy
import logging
import os
import sys
import io
import wave
from typing import Optional

from dotenv import load_dotenv
from groq import Groq

logger = logging.getLogger(__name__)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config")
if CONFIG_PATH not in sys.path:
    sys.path.insert(0, CONFIG_PATH)
try:
    from stt_config import STT_CONFIG as USER_STT_CONFIG
except ImportError as exc:
    raise RuntimeError("config/stt_config.py 를 불러올 수 없습니다.") from exc


def _load_stt_config() -> dict:
    if not isinstance(USER_STT_CONFIG, dict):
        raise RuntimeError("stt_config.STT_CONFIG 형식이 올바르지 않습니다.")

    return copy.deepcopy(USER_STT_CONFIG)


class STTManager:
    def __init__(self):
        load_dotenv()

        self.stt_config = _load_stt_config()

        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

        self.model = self._resolve_model_name("model")
        self.audio_format = self._resolve_audio_format()
        self.parameters = self._resolve_parameters()

        logger.info("STT init complete (model=%s)", self.model)

    def _resolve_model_name(self, config_key: str) -> str:
        if config_key not in self.stt_config:
            raise RuntimeError(f"stt_config 설정에 '{config_key}' 키가 없습니다.")

        config_value = self.stt_config[config_key]
        if not isinstance(config_value, str) or not config_value.strip():
            raise RuntimeError(
                f"stt_config 설정 '{config_key}' 값이 올바르지 않습니다."
            )

        resolved = config_value.strip()
        logger.info("STT 모델 설정: %s", resolved)
        return resolved

    def _resolve_audio_format(self) -> dict:
        if "audio_format" not in self.stt_config:
            raise RuntimeError("stt_config 설정에 'audio_format' 키가 없습니다.")

        config_format = self.stt_config["audio_format"]
        if not isinstance(config_format, dict):
            raise RuntimeError("stt_config.audio_format 설정이 올바르지 않습니다.")

        required_keys = {"channels", "sample_width", "frame_rate"}
        missing = required_keys - config_format.keys()
        if missing:
            raise RuntimeError(
                f"stt_config.audio_format 설정에 누락된 키가 있습니다: {', '.join(sorted(missing))}"
            )

        resolved = {
            "channels": int(config_format["channels"]),
            "sample_width": int(config_format["sample_width"]),
            "frame_rate": int(config_format["frame_rate"]),
        }

        logger.info(
            "STT audio format: channels=%s, sample_width=%s, frame_rate=%s",
            resolved["channels"],
            resolved["sample_width"],
            resolved["frame_rate"],
        )

        return resolved

    def _resolve_parameters(self) -> dict:
        if "parameters" not in self.stt_config:
            raise RuntimeError("stt_config 설정에 'parameters' 키가 없습니다.")

        params = self.stt_config["parameters"]
        if not isinstance(params, dict):
            raise RuntimeError("stt_config.parameters 설정이 올바르지 않습니다.")

        required = {"language", "temperature"}
        missing = required - params.keys()
        if missing:
            raise RuntimeError(
                f"stt_config.parameters 설정에 누락된 키가 있습니다: {', '.join(sorted(missing))}"
            )

        language = params["language"]
        temperature = params["temperature"]

        if not isinstance(language, str) or not language.strip():
            raise RuntimeError("stt_config.parameters.language 값이 올바르지 않습니다.")

        if not isinstance(temperature, (int, float)):
            raise RuntimeError(
                "stt_config.parameters.temperature 값이 숫자가 아닙니다."
            )

        resolved = {"language": language.strip(), "temperature": float(temperature)}
        logger.info(
            "STT parameters: language=%s, temperature=%s",
            resolved["language"],
            resolved["temperature"],
        )
        return resolved

    def transcribe(
        self, audio_data: bytes, audio_format: Optional[dict] = None
    ) -> Optional[str]:
        """설정된 STT 모델로 원시 오디오를 텍스트로 변환."""
        if audio_format is None:
            audio_format = self.audio_format

        try:
            buffer = io.BytesIO()
            with wave.open(buffer, "wb") as wav_file:
                wav_file.setnchannels(audio_format["channels"])
                wav_file.setsampwidth(audio_format["sample_width"])
                wav_file.setframerate(audio_format["frame_rate"])
                wav_file.writeframes(audio_data)

            buffer.seek(0)

            transcription = self.client.audio.transcriptions.create(
                file=("audio.wav", buffer, "audio/wav"),
                model=self.model,
                language=self.parameters["language"],
                temperature=self.parameters["temperature"],
            )

            text = transcription.text.strip()
            logger.info("음성 인식 결과: %s", text)
            return text

        except Exception as exc:
            logger.error("STT 변환 실패: %s", exc)
            return None

    def get_audio_format(self) -> dict:
        return self.audio_format.copy()

    def get_current_config(self) -> dict:
        return {
            "model": self.model,
            "audio_format": self.get_audio_format(),
            "parameters": self.parameters.copy(),
        }
