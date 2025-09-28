"""구성에 따라 동작하는 음성 합성(TTS) 관리자."""

import copy
import os
import sys
import pygame
from typing import Optional

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "config")
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


try:
    from ..utils import GTTSEngine  # type: ignore
except Exception:
    GTTSEngine = None

try:
    from ..utils import ElevenLabsEngine  # type: ignore
except Exception:
    ElevenLabsEngine = None


class TTSManager:
    """설정에 맞춰 TTS 엔진 선택과 합성을 처리."""

    def __init__(self, preferred_engine: Optional[str] = None):
        """구성에 지정된 우선순위를 기준으로 엔진을 초기화."""
        self.manager_config = _load_manager_config()

        self._engine_classes = {"gtts": GTTSEngine, "elevenlabs": ElevenLabsEngine}

        self.engine_priority = self._resolve_engine_priority()
        self.language = self._resolve_language()

        default_engine = self._resolve_default_engine()
        self.preferred_engine = (
            preferred_engine if preferred_engine is not None else default_engine
        )
        self.preferred_engine = self.preferred_engine.strip().lower()

        valid_engine_ids = set(self.engine_priority)
        valid_engine_ids.update(
            key for key, cls in self._engine_classes.items() if cls is not None
        )
        if (
            self.preferred_engine != "auto"
            and self.preferred_engine not in valid_engine_ids
        ):
            raise RuntimeError(
                "지원하지 않는 TTS 엔진이 지정되었습니다: %s" % self.preferred_engine
            )

        self.current_engine = None

        # pygame 초기화
        self.pygame_initialized = False
        self._init_pygame()

        # 엔진 초기화
        self._init_engines()

    def _init_pygame(self):
        """pygame 초기화"""
        try:
            pygame.mixer.init()
            self.pygame_initialized = True
            print("✅ TTSManager: pygame mixer initialised")
        except Exception as exc:
            print(f"❌ TTSManager: pygame 초기화 실패: {exc}")
            self.pygame_initialized = False

    def _init_engines(self):
        """TTS 엔진들 초기화"""
        self.engines = {}

        load_order: list[str] = []
        for engine_id in self.engine_priority:
            if engine_id not in load_order:
                load_order.append(engine_id)
        for engine_id in self._engine_classes:
            if engine_id not in load_order:
                load_order.append(engine_id)

        for engine_id in load_order:
            engine_class = self._engine_classes.get(engine_id)
            if engine_class is None:
                continue

            try:
                engine = engine_class()
                self.engines[engine_id] = engine

                if engine.is_available():
                    print(f"✅ TTSManager: {engine.engine_id} ({engine.name}) ready")
                else:
                    print(
                        f"❌ TTSManager: {engine.engine_id} ({engine.name}) unavailable"
                    )

            except Exception as exc:
                print(f"❌ TTSManager: {engine_id} 초기화 실패: {exc}")

        # 현재 엔진 선택
        self._select_current_engine()

    def _select_current_engine(self):
        """현재 사용할 엔진 선택"""
        selected = None

        if self.preferred_engine == "auto":
            for engine_id in self.engine_priority:
                engine = self.engines.get(engine_id)
                if engine and engine.is_available():
                    selected = engine_id
                    break
        else:
            engine = self.engines.get(self.preferred_engine)
            if engine and engine.is_available():
                selected = self.preferred_engine

        if not selected:
            for engine_id, engine in self.engines.items():
                if engine.is_available():
                    selected = engine_id
                    break

        self.current_engine = selected

        if self.current_engine:
            engine = self.engines[self.current_engine]
            print(f"🎯 TTS Manager: 현재 엔진 - {engine.engine_id} ({engine.name})")
        else:
            print("❌ TTS Manager: 사용 가능한 엔진이 없습니다")

    def is_available(self) -> bool:
        """TTS 사용 가능 여부"""
        return self.current_engine is not None and self.pygame_initialized

    def get_current_engine_info(self) -> Optional[dict]:
        """현재 엔진 정보 반환"""
        if not self.current_engine:
            return None

        engine = self.engines[self.current_engine]
        return engine.get_info()

    def speak(self, text: str, save_file: bool = False) -> bool:
        """텍스트를 합성하고 필요하면 재생 전 오디오를 저장."""
        if not self.is_available():
            print("❌ TTSManager: TTS subsystem unavailable")
            return False

        if not text or not text.strip():
            print("❌ TTSManager: text is empty")
            return False

        engine = self.engines[self.current_engine]

        try:
            if save_file:
                timestamp = int(os.path.getmtime(__file__) * 1000) % 1000000
                filename = f"tts_output_{timestamp}.mp3"

                if engine.synthesize(text, self.language, filename):
                    return engine.play(filename)
                else:
                    return False
            else:
                return engine.synthesize_and_play(text, self.language)

        except Exception as exc:
            print(f"❌ TTSManager: synthesis failed: {exc}")
            return False

    def save_audio(self, text: str, filename: Optional[str] = None) -> Optional[str]:
        """합성한 오디오를 재생 없이 디스크에 저장."""
        if not self.current_engine:
            print("❌ TTSManager: no usable engines")
            return None

        if not text or not text.strip():
            print("❌ TTSManager: text is empty")
            return None

        engine = self.engines[self.current_engine]

        if not filename:
            timestamp = int(os.path.getmtime(__file__) * 1000) % 1000000
            filename = f"tts_korean_{timestamp}.mp3"

        try:
            if engine.synthesize(text, self.language, filename):
                return os.path.abspath(filename)
            return None
        except Exception as exc:
            print(f"❌ TTSManager: failed to save audio: {exc}")
            return None

    def _resolve_default_engine(self) -> str:
        value = self.manager_config.get("default_engine")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                "tts_config.TTS_MANAGER_CONFIG.default_engine 값이 올바르지 않습니다."
            )
        return value.strip().lower()

    def _resolve_engine_priority(self) -> list:
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
        value = self.manager_config.get("language")
        if not isinstance(value, str) or not value.strip():
            raise RuntimeError(
                "tts_config.TTS_MANAGER_CONFIG.language 값이 올바르지 않습니다."
            )
        return value.strip()

    def switch_engine(self, engine_name: str) -> bool:
        """
        TTS 엔진 변경

        Args:
            engine_name: 변경할 엔진명 ('gtts' 또는 'elevenlabs')

        Returns:
            bool: 변경 성공 여부
        """
        if engine_name not in self.engines:
            print(f"❌ TTS Manager: '{engine_name}' 엔진을 사용할 수 없습니다")
            available = list(self.engines.keys())
            if available:
                print(f"   사용 가능한 엔진: {', '.join(available)}")
            return False

        old_engine = self.current_engine
        self.current_engine = engine_name

        old_name = self.engines[old_engine].name if old_engine else "없음"
        new_name = self.engines[self.current_engine].name
        print(f"🔄 TTS Manager: 엔진 변경 {old_name} → {new_name}")

        return True

    def get_available_engines(self) -> list:
        """사용 가능한 엔진 목록 반환"""
        return [
            engine_id
            for engine_id, engine in self.engines.items()
            if engine.is_available()
        ]

    def test_priority_chain(self, test_text: str) -> bool:
        """우선순위 순서대로 엔진을 시험해 성공 시까지 진행."""
        previous_engine = self.current_engine

        for engine_id in self.engine_priority:
            engine = self.engines.get(engine_id)
            if not engine or not engine.is_available():
                continue

            self.current_engine = engine_id
            if self.test_current_engine(test_text):
                return True

        self.current_engine = previous_engine
        return False

    def test_current_engine(self, test_text: str) -> bool:
        """
        현재 엔진 테스트

        Args:
            test_text: 테스트할 텍스트

        Returns:
            bool: 테스트 성공 여부
        """
        if not self.is_available():
            print("❌ TTS Manager: TTS를 사용할 수 없습니다")
            return False

        engine = self.engines[self.current_engine]
        print(f"🧪 TTS Manager: {engine.name} 테스트 시작")

        return engine.test(test_text)

    def cleanup(self):
        """리소스 정리"""
        try:
            if self.pygame_initialized:
                pygame.mixer.quit()
                self.pygame_initialized = False
            print("✅ TTS Manager: 정리 완료")
        except Exception as e:
            print(f"❌ TTS Manager: 정리 실패: {e}")

    def __del__(self):
        """소멸자"""
        self.cleanup()
