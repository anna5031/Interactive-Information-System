"""tts_config.py 설정을 따르는 gTTS 어댑터."""

import os
import sys
from gtts import gTTS
import pygame
from typing import Optional

TEMP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "temp"))
os.makedirs(TEMP_DIR, exist_ok=True)

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config")
if CONFIG_PATH not in sys.path:
    sys.path.insert(0, CONFIG_PATH)
try:
    from tts_config import GTTS_CONFIG
except ImportError as exc:
    raise RuntimeError("config/tts_config.py 를 불러올 수 없습니다.") from exc

if not isinstance(GTTS_CONFIG, dict):
    raise RuntimeError("tts_config.GTTS_CONFIG 형식이 올바르지 않습니다.")


def _load_language(config_key: str) -> str:
    value = GTTS_CONFIG.get(config_key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"tts_config.GTTS_CONFIG.{config_key} 값이 올바르지 않습니다."
        )
    return value.strip()


def _load_supported_languages() -> list:
    supported = GTTS_CONFIG.get("supported_languages")
    if supported is None:
        return [_load_language("language")]
    if not isinstance(supported, list) or not supported:
        raise RuntimeError(
            "tts_config.GTTS_CONFIG.supported_languages 설정이 올바르지 않습니다."
        )
    normalized = [
        lang.strip().lower()
        for lang in supported
        if isinstance(lang, str) and lang.strip()
    ]
    if not normalized:
        raise RuntimeError(
            "tts_config.GTTS_CONFIG.supported_languages 값이 올바르지 않습니다."
        )
    return normalized


def _load_slow_flag() -> bool:
    slow = GTTS_CONFIG.get("slow", False)
    if not isinstance(slow, bool):
        raise RuntimeError("tts_config.GTTS_CONFIG.slow 값이 boolean이 아닙니다.")
    return slow


class GTTSEngine:
    def __init__(self):
        self.name = "Google TTS"
        self.engine_id = "gtts"

        self.language = _load_language("language")
        self.supported_languages = _load_supported_languages()
        self.slow = _load_slow_flag()

        try:
            pygame.mixer.init()
            self.available = True
            print("✅ GTTSEngine: pygame mixer initialised")
        except Exception as exc:
            print(f"❌ GTTSEngine: pygame 초기화 실패: {exc}")
            self.available = False

    def is_available(self) -> bool:
        return self.available

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> bool:
        if not self.is_available():
            print(f"❌ {self.name} 엔진을 사용할 수 없습니다")
            return False

        lang = (language or self.language).strip().lower()
        if lang not in self.supported_languages:
            print(f"❌ Unsupported TTS language: {lang}")
            print(f"Available languages: {', '.join(self.supported_languages)}")
            return False

        try:
            tts = gTTS(text=text, lang=lang, slow=self.slow)
            if output_file is None:
                output_file = os.path.abspath(os.path.join(TEMP_DIR, "gtts_output.mp3"))
            else:
                output_file = os.path.abspath(output_file)

            os.makedirs(os.path.dirname(output_file) or TEMP_DIR, exist_ok=True)
            tts.save(output_file)
            print(f"💾 Saved audio: {output_file}")
            return True

        except Exception as exc:
            print(f"❌ gTTS synthesis failed: {exc}")
            return False

    def play(self, audio_file: str) -> bool:
        if not self.is_available():
            return False
        if not os.path.exists(audio_file):
            print(f"❌ Audio file not found: {audio_file}")
            return False

        try:
            pygame.mixer.music.load(audio_file)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                pygame.time.wait(100)
            print("✅ Playback complete")
            return True
        except Exception as exc:
            print(f"❌ Playback failed: {exc}")
            return False

    def synthesize_and_play(self, text: str, language: Optional[str] = None) -> bool:
        temp_file = os.path.join(TEMP_DIR, "temp_gtts.mp3")

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
        return self.synthesize_and_play(test_text)

    def get_info(self) -> dict:
        return {
            "name": self.name,
            "engine_id": self.engine_id,
            "available": self.available,
            "supported_languages": self.supported_languages,
            "description": "Google Text-to-Speech 서비스",
        }
