"""gTTS adapter respecting legacy configuration."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from gtts import gTTS

from config.qa.tts_config import GTTS_CONFIG  # type: ignore

from .audio_utils import play_audio_file

TEMP_DIR = Path(__file__).resolve().parents[2] / "temp"
TEMP_DIR.mkdir(parents=True, exist_ok=True)

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
    def __init__(self) -> None:
        self.name = "Google TTS"
        self.engine_id = "gtts"

        self.language = _load_language("language")
        self.supported_languages = _load_supported_languages()
        self.slow = _load_slow_flag()
        self.available = True

    def is_available(self) -> bool:
        return self.available

    def synthesize(
        self,
        text: str,
        language: Optional[str] = None,
        output_file: Optional[str] = None,
    ) -> bool:
        lang = (language or self.language).strip().lower()
        if lang not in self.supported_languages:
            print(f"❌ Unsupported TTS language: {lang}")
            print(f"Available languages: {', '.join(self.supported_languages)}")
            return False

        try:
            tts = gTTS(text=text, lang=lang, slow=self.slow)
            if output_file is None:
                output_file = TEMP_DIR / "gtts_output.mp3"
            else:
                output_file = Path(output_file).resolve()

            output_file.parent.mkdir(parents=True, exist_ok=True)
            tts.save(output_file.as_posix())
            print(f"💾 Saved audio: {output_file}")
            return True

        except Exception as exc:
            print(f"❌ gTTS synthesis failed: {exc}")
            return False

    def play(self, audio_file: str) -> bool:
        return play_audio_file(audio_file)

    def synthesize_and_play(self, text: str, language: Optional[str] = None) -> bool:
        temp_file = TEMP_DIR / "temp_gtts.mp3"

        try:
            result = self.synthesize(text, language, temp_file.as_posix()) and self.play(temp_file.as_posix())
        finally:
            if temp_file.exists():
                try:
                    temp_file.unlink()
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
