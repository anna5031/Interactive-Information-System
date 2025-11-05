import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class StubEngine:
    def __init__(self, engine_id: str, available: bool = True) -> None:
        self.engine_id = engine_id
        self.name = f"{engine_id.title()} Engine"
        self._available = available
        self.calls = []

    def is_available(self) -> bool:
        return self._available

    def synthesize_and_play(self, text: str, language: str) -> bool:
        self.calls.append(("play", text, language))
        return True

    def synthesize(self, text: str, language: str, filename: str) -> bool:
        self.calls.append(("save", text, language, filename))
        return True

    def test(self, test_text: str) -> bool:
        self.calls.append(("test", test_text))
        return True

    def get_info(self) -> dict:
        return {
            "engine_id": self.engine_id,
            "name": self.name,
            "description": f"{self.name} description",
        }


class StubFactory:
    def __init__(self, engines: dict[str, StubEngine]) -> None:
        self._engines = engines

    def get_all_engines(self) -> list[dict]:
        return [engine.get_info() for engine in self._engines.values()]

    def get_available_engines(self) -> list[dict]:
        return [
            engine.get_info()
            for engine in self._engines.values()
            if engine.is_available()
        ]

    def select_best_engine(self) -> str | None:
        available = self.get_available_engines()
        return available[0]["engine_id"] if available else None

    def get_engine(self, engine_id: str) -> StubEngine | None:
        return self._engines.get(engine_id)


class TTSManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        engines = {
            "gtts": StubEngine("gtts"),
            "elevenlabs": StubEngine("elevenlabs", available=False),
        }
        self.factory = StubFactory(engines)

        import types
        import importlib.util

        features_pkg = sys.modules.setdefault("features", types.ModuleType("features"))
        features_pkg.__path__ = [str(PROJECT_ROOT / "features")]

        qa_pkg = sys.modules.setdefault("features.qa", types.ModuleType("features.qa"))
        qa_pkg.__path__ = [str(PROJECT_ROOT / "features" / "qa")]

        managers_pkg = sys.modules.setdefault(
            "features.qa.managers", types.ModuleType("features.qa.managers")
        )
        managers_pkg.__path__ = [str(PROJECT_ROOT / "features" / "qa" / "managers")]

        module_name = "features.qa.managers.tts_manager"
        spec = importlib.util.spec_from_file_location(
            module_name, PROJECT_ROOT / "features" / "qa" / "managers" / "tts_manager.py"
        )
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)  # type: ignore[arg-type]
        self.tts_module = module

        factory_patch = patch.object(
            self.tts_module, "TTSFactory", return_value=self.factory
        )
        self.addCleanup(factory_patch.stop)
        factory_patch.start()

        manager_patch = patch.object(
            self.tts_module,
            "_load_manager_config",
            return_value={
                "default_engine": "auto",
                "engine_priority": ["elevenlabs", "gtts"],
                "language": "ko",
            },
        )
        self.addCleanup(manager_patch.stop)
        manager_patch.start()

    def test_speak_uses_available_engine(self) -> None:
        manager = self.tts_module.TTSManager()
        self.assertTrue(manager.is_available())

        success = manager.speak("테스트 음성입니다.")
        self.assertTrue(success)

        gtts_engine = self.factory.get_engine("gtts")
        self.assertIn(("play", "테스트 음성입니다.", "ko"), gtts_engine.calls)

    def test_speak_saves_file_when_requested(self) -> None:
        manager = self.tts_module.TTSManager()
        success = manager.speak("파일 저장 테스트", save_file=True)
        self.assertTrue(success)

        gtts_engine = self.factory.get_engine("gtts")
        save_calls = [call for call in gtts_engine.calls if call[0] == "save"]
        self.assertTrue(save_calls)
        saved_path = save_calls[0][3]
        self.assertTrue(saved_path.endswith(".mp3"))
        self.assertTrue(os.path.basename(saved_path).startswith("tts_output"))

    def test_switch_engine_rejects_unavailable(self) -> None:
        manager = self.tts_module.TTSManager()
        # 현재는 gtts만 사용 가능
        self.assertFalse(manager.switch_engine("elevenlabs"))
        self.assertTrue(manager.switch_engine("gtts"))

    def test_test_priority_chain_fallback(self) -> None:
        manager = self.tts_module.TTSManager()
        result = manager.test_priority_chain("테스트 음성입니다.")
        self.assertTrue(result)

        gtts_engine = self.factory.get_engine("gtts")
        self.assertIn(("test", "테스트 음성입니다."), gtts_engine.calls)


if __name__ == "__main__":
    unittest.main()
