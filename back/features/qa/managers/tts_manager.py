"""구성에 따라 동작하는 음성 합성(TTS) 관리자."""

import copy
import os
import sys
from typing import Optional

from ..utils import TTSFactory

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "config")
if CONFIG_PATH not in sys.path:
    sys.path.insert(0, CONFIG_PATH)
try:
    from tts_config import TTS_MANAGER_CONFIG
except ImportError as exc:
    raise RuntimeError("config/tts_config.py 를 불러올 수 없습니다.") from exc

if not isinstance(TTS_MANAGER_CONFIG, dict):
    raise RuntimeError("tts_config.TTS_MANAGER_CONFIG 형식이 올바르지 않습니다.")


def _load_manager_config() -> dict:
    return copy.deepcopy(TTS_MANAGER_CONFIG)


class TTSManager:
    """설정에 맞춰 TTS 엔진 선택과 합성을 처리."""

    def __init__(self, preferred_engine: Optional[str] = None) -> None:
        self.manager_config = _load_manager_config()
        self.factory = TTSFactory()

        self.engine_priority = self._resolve_engine_priority()
        self.language = self._resolve_language()

        default_engine = self._resolve_default_engine()
        self.preferred_engine = (
            preferred_engine if preferred_engine is not None else default_engine
        )
        self.preferred_engine = self.preferred_engine.strip().lower()

        self.current_engine = self._select_current_engine()

    def _select_current_engine(self) -> Optional[str]:
        available_ids = {
            info["engine_id"] for info in self.factory.get_available_engines()
        }

        if self.preferred_engine != "auto" and self.preferred_engine in available_ids:
            return self.preferred_engine

        for engine_id in self.engine_priority:
            if engine_id in available_ids:
                return engine_id

        best = self.factory.select_best_engine()
        if best in available_ids:
            return best
        return None

    def is_available(self) -> bool:
        if not self.current_engine:
            return False
        engine = self.factory.get_engine(self.current_engine)
        return bool(engine and engine.is_available())

    def speak(self, text: str, save_file: bool = False) -> bool:
        if not text or not text.strip():
            return False
        if not self.current_engine:
            print("❌ TTSManager: 사용 가능한 엔진이 없습니다")
            return False

        engine = self.factory.get_engine(self.current_engine)
        if not engine or not engine.is_available():
            print("❌ TTSManager: 현재 엔진이 비활성화되었습니다")
            return False

        try:
            if save_file:
                output_dir = os.path.join(os.path.dirname(__file__), "..", "..", "temp")
                os.makedirs(output_dir, exist_ok=True)
                filename = os.path.join(output_dir, "tts_output.mp3")
                return engine.synthesize(text, self.language, filename)
            return engine.synthesize_and_play(text, self.language)
        except Exception as exc:
            print(f"❌ TTSManager: synthesis failed: {exc}")
            return False

    def get_available_engines(self) -> list:
        return [info["engine_id"] for info in self.factory.get_available_engines()]

    def get_current_engine_info(self) -> Optional[dict]:
        if not self.current_engine:
            return None
        for info in self.factory.get_all_engines():
            if info["engine_id"] == self.current_engine:
                return info
        return None

    def test_priority_chain(self, test_text: str) -> bool:
        previous_engine = self.current_engine
        for engine_id in self.engine_priority:
            engine = self.factory.get_engine(engine_id)
            if not engine or not engine.is_available():
                continue
            self.current_engine = engine_id
            if engine.test(test_text):
                return True
        self.current_engine = previous_engine
        return False

    def test_current_engine(self, test_text: str) -> bool:
        if not self.is_available():
            print("❌ TTS Manager: TTS를 사용할 수 없습니다")
            return False
        engine = self.factory.get_engine(self.current_engine)
        return engine.test(test_text)

    def switch_engine(self, engine_name: str) -> bool:
        engine = self.factory.get_engine(engine_name)
        if not engine:
            print(f"❌ TTS Manager: '{engine_name}' 엔진을 사용할 수 없습니다")
            return False
        if not engine.is_available():
            print(f"❌ TTS Manager: {engine_name} 엔진이 비활성화되어 있습니다")
            return False
        old_engine = self.current_engine
        self.current_engine = engine_name
        old_name = None
        if old_engine:
            old = self.factory.get_engine(old_engine)
            old_name = old.name if old else "없음"
        print(f"🔄 TTS Manager: 엔진 변경 {old_name or '없음'} → {engine.name}")
        return True

    def cleanup(self) -> None:
        print("✅ TTS Manager: 정리 완료")

    def __del__(self) -> None:
        self.cleanup()

    def _resolve_default_engine(self) -> str:
        value = self.manager_config.get("default_engine")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                "tts_config.TTS_MANAGER_CONFIG.default_engine 값이 올바르지 않습니다."
            )
        return value.strip().lower()

    def _resolve_engine_priority(self) -> list[str]:
        priority = self.manager_config.get("engine_priority")
        if not isinstance(priority, (list, tuple)) or not priority:
            raise RuntimeError(
                "tts_config.TTS_MANAGER_CONFIG.engine_priority 설정이 올바르지 않습니다."
            )
        normalized = []
        for engine_id in priority:
            if not isinstance(engine_id, str) or not engine_id.strip():
                raise RuntimeError(
                    "tts_config.TTS_MANAGER_CONFIG.engine_priority 항목이 올바르지 않습니다."
                )
            normalized.append(engine_id.strip().lower())
        return normalized

    def _resolve_language(self) -> str:
        value = self.manager_config.get("language", "ko")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                "tts_config.TTS_MANAGER_CONFIG.language 값이 올바르지 않습니다."
            )
        return value.strip().lower()
