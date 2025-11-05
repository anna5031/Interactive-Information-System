"""구성 파일을 기반으로 ElevenLabs 음성 합성을 제어하는 어댑터."""

import os
import sys
from typing import Dict, Optional, Any

import requests
from dotenv import load_dotenv

from .audio import play_audio_file

TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "temp"))
os.makedirs(TEMP_DIR, exist_ok=True)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config")
if CONFIG_PATH not in sys.path:
    sys.path.insert(0, CONFIG_PATH)
try:
    from tts_config import ELEVENLABS_CONFIG
except ImportError as exc:
    raise RuntimeError("config/tts_config.py 를 불러올 수 없습니다.") from exc

if not isinstance(ELEVENLABS_CONFIG, dict):
    raise RuntimeError("tts_config.ELEVENLABS_CONFIG 형식이 올바르지 않습니다.")


def _load_config_value(config: Dict, key: str, expected_type):
    if key not in config:
        raise RuntimeError(f"tts_config.ELEVENLABS_CONFIG 에 '{key}' 키가 없습니다.")
    value = config[key]
    if not isinstance(value, expected_type):
        raise RuntimeError(
            f"tts_config.ELEVENLABS_CONFIG.{key} 값이 올바르지 않습니다."
        )
    return value


def _normalize_voice_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Validate and prepare ElevenLabs voice settings from configuration."""

    numeric_keys = {"stability", "similarity_boost", "style", "speed"}
    normalized: Dict[str, Any] = {}

    for key, value in settings.items():
        if key in numeric_keys:
            if value is None:
                normalized[key] = None
            elif isinstance(value, (int, float)):
                normalized[key] = float(value)
            else:
                raise RuntimeError(
                    "tts_config.ELEVENLABS_CONFIG.voice_settings.%s 값은 숫자 또는 null 이어야 합니다." % key
                )
            continue

        if key == "use_speaker_boost":
            if not isinstance(value, bool):
                raise RuntimeError(
                    "tts_config.ELEVENLABS_CONFIG.voice_settings.use_speaker_boost 값은 bool 이어야 합니다."
                )
            normalized[key] = value
            continue

        normalized[key] = value

    return normalized


class ElevenLabsEngine:
    def __init__(self):
        self.name = "ElevenLabs TTS"
        self.engine_id = "elevenlabs"

        load_dotenv()
        self.api_key = os.getenv("ELEVENLABS_API_KEY")

        self.voice_id = _load_config_value(ELEVENLABS_CONFIG, "voice_id", str).strip()
        if not self.voice_id:
            raise RuntimeError(
                "tts_config.ELEVENLABS_CONFIG.voice_id 값이 비어 있습니다."
            )

        self.model_id = _load_config_value(ELEVENLABS_CONFIG, "model_id", str).strip()
        if not self.model_id:
            raise RuntimeError(
                "tts_config.ELEVENLABS_CONFIG.model_id 값이 비어 있습니다."
            )

        voice_settings = _load_config_value(ELEVENLABS_CONFIG, "voice_settings", dict)
        self.voice_settings = _normalize_voice_settings(voice_settings)

        supported_languages = _load_config_value(
            ELEVENLABS_CONFIG, "supported_languages", list
        )
        self.supported_languages = [
            lang.strip().lower()
            for lang in supported_languages
            if isinstance(lang, str) and lang.strip()
        ]
        if not self.supported_languages:
            raise RuntimeError(
                "tts_config.ELEVENLABS_CONFIG.supported_languages 값이 올바르지 않습니다."
            )

        print("📋 ElevenLabs Config 로드:")
        print(f"   voice_id={self.voice_id}")
        print(f"   voice_settings={self.voice_settings}")
        print(f"   model_id={self.model_id}")

        self.available = False

        self._check_api_connection()

    def _check_api_connection(self) -> bool:
        if not self.api_key:
            print("❌ ELEVENLABS_API_KEY is not configured")
            return False

        try:
            headers = {"xi-api-key": self.api_key}
            response = requests.get(
                "https://api.elevenlabs.io/v1/user", headers=headers, timeout=10
            )
            if response.status_code == 200:
                print("✅ ElevenLabs API reachable")
                self.available = True
                return True
            print(f"❌ ElevenLabs API error: {response.status_code}")
            return False
        except Exception as exc:
            print(f"❌ ElevenLabs API connection failed: {exc}")
            return False

    def is_available(self) -> bool:
        return self.available

    def synthesize(
        self, text: str, language: str, output_file: Optional[str] = None
    ) -> bool:
        if not self.is_available():
            print(f"❌ {self.name} 엔진을 사용할 수 없습니다")
            return False

        lang = language.strip().lower()
        if lang not in self.supported_languages:
            print(f"❌ Unsupported TTS language: {language}")
            print(f"Available languages: {', '.join(self.supported_languages)}")
            return False

        try:
            headers = {"xi-api-key": self.api_key, "Content-Type": "application/json"}
            data = {
                "text": text,
                "model_id": self.model_id,
                "voice_settings": self.voice_settings,
            }
            response = requests.post(
                f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}",
                headers=headers,
                json=data,
                timeout=30,
            )

            if response.status_code != 200:
                print(f"❌ ElevenLabs synthesis failed (status {response.status_code})")
                try:
                    print(f"Response body: {response.json()}")
                except Exception:
                    print(f"Response text: {response.text}")
                return False

            if output_file is None:
                output_file = os.path.join(TEMP_DIR, "elevenlabs_output.mp3")
            else:
                output_file = os.path.abspath(output_file)

            os.makedirs(os.path.dirname(output_file) or TEMP_DIR, exist_ok=True)
            with open(output_file, "wb") as file:
                file.write(response.content)

            print(f"💾 Saved audio: {output_file}")
            return True

        except Exception as exc:
            print(f"❌ ElevenLabs synthesis error: {exc}")
            return False

    def play(self, audio_file: str) -> bool:
        return play_audio_file(audio_file)

    def synthesize_and_play(self, text: str, language: str) -> bool:
        temp_file = os.path.join(TEMP_DIR, "temp_elevenlabs.mp3")
        try:
            if self.synthesize(text, language, temp_file):
                result = self.play(temp_file)
            else:
                result = False
        finally:
            if os.path.exists(temp_file):
                try:
                    os.remove(temp_file)
                except OSError:
                    pass
        return result

    def test(self, test_text: str) -> bool:
        print(f"🧪 {self.name} test")
        print(f"   Text: {test_text}")
        return self.synthesize_and_play(test_text, self.supported_languages[0])

    def get_info(self) -> dict:
        return {
            "name": self.name,
            "engine_id": self.engine_id,
            "available": self.available,
            "supported_languages": self.supported_languages,
            "description": "ElevenLabs TTS service",
            "voice_id": self.voice_id,
            "model_id": self.model_id,
            "api_key_set": bool(self.api_key),
        }
