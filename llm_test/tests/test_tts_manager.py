"""가짜 엔진을 이용해 TTSManager 동작을 검증하는 테스트."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


if "pygame" not in sys.modules:
    sys.modules["pygame"] = SimpleNamespace(
        mixer=SimpleNamespace(init=lambda: None, quit=lambda: None)
    )

if "numpy" not in sys.modules:
    sys.modules["numpy"] = SimpleNamespace(
        int16="int16",
        float32="float32",
        float64="float64",
        mean=lambda *args, **kwargs: 0.0,
        max=lambda *args, **kwargs: 0.0,
        abs=abs,
        sqrt=lambda _x: 0.0,
        ndarray=object,
    )

if "sounddevice" not in sys.modules:

    class _DummyInputStream:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            return None

        def stop(self):
            return None

        def close(self):
            return None

        def read(self, *args, **kwargs):
            return ([], False)

    sys.modules["sounddevice"] = SimpleNamespace(
        default=SimpleNamespace(samplerate=None, channels=None, dtype=None),
        query_devices=lambda: [],
        InputStream=_DummyInputStream,
    )

if "dotenv" not in sys.modules:
    sys.modules["dotenv"] = SimpleNamespace(load_dotenv=lambda *args, **kwargs: None)

if "groq" not in sys.modules:

    class _DummyGroq:
        def __init__(self, *args, **kwargs):
            self.chat = SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda *a, **k: SimpleNamespace(choices=[])
                )
            )

    sys.modules["groq"] = SimpleNamespace(Groq=_DummyGroq)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def build_fake_engine(engine_id: str, name: str, available: bool):
    class FakeEngine:
        instances = []

        def __init__(self):
            self.engine_id = engine_id
            self.name = name
            self.calls = []
            self.available = available
            self.__class__.instances.append(self)

        def is_available(self):
            return self.available

        def synthesize_and_play(self, text: str, language: str):
            self.calls.append(("synthesize_and_play", text, language))
            return True

        def synthesize(self, text: str, language: str, output_file: str):
            self.calls.append(("synthesize", text, language, output_file))
            return True

        def play(self, audio_path: str):
            self.calls.append(("play", audio_path))
            return True

        def get_info(self):
            return {
                "name": self.name,
                "engine_id": self.engine_id,
                "available": self.available,
            }

        def test(self, text: str):
            self.calls.append(("test", text))
            return True

    return FakeEngine


BASE_MANAGER_CONFIG = {
    "default_engine": "auto",
    "engine_priority": ["gtts", "elevenlabs"],
    "language": "ko",
}


class TTSManagerTests(unittest.TestCase):
    def _make_manager(self, gtts_available=True, eleven_available=False, config=None):
        from src.managers.tts_manager import TTSManager

        fake_gtts = build_fake_engine("gtts", "Fake gTTS", gtts_available)
        fake_eleven = build_fake_engine(
            "elevenlabs", "Fake ElevenLabs", eleven_available
        )
        fake_gtts.instances = []
        fake_eleven.instances = []

        manager_config = dict(BASE_MANAGER_CONFIG)
        if config:
            manager_config.update(config)

        pygame_patch = mock.patch("src.managers.tts_manager.pygame.mixer")
        gtts_patch = mock.patch("src.managers.tts_manager.GTTSEngine", fake_gtts)
        eleven_patch = mock.patch(
            "src.managers.tts_manager.ElevenLabsEngine", fake_eleven
        )
        config_patch = mock.patch(
            "src.managers.tts_manager._load_manager_config", return_value=manager_config
        )

        pygame_mock = pygame_patch.start()
        pygame_mock.init.return_value = None
        pygame_mock.quit.return_value = None
        self.addCleanup(pygame_patch.stop)

        gtts_patch.start()
        self.addCleanup(gtts_patch.stop)

        eleven_patch.start()
        self.addCleanup(eleven_patch.stop)

        config_patch.start()
        self.addCleanup(config_patch.stop)

        manager = TTSManager()
        return manager, fake_gtts, fake_eleven

    def test_auto_selects_first_available_engine(self):
        manager, fake_gtts, _ = self._make_manager(
            gtts_available=True, eleven_available=False
        )

        self.assertEqual(manager.current_engine, "gtts")
        self.assertTrue(manager.is_available())
        self.assertEqual(manager.get_available_engines(), ["gtts"])
        self.assertEqual(manager.get_current_engine_info()["engine_id"], "gtts")

        engine_instance = manager.engines["gtts"]
        self.assertIs(engine_instance, fake_gtts.instances[0])

    def test_auto_falls_back_to_secondary_engine(self):
        manager, _, fake_eleven = self._make_manager(
            gtts_available=False, eleven_available=True
        )

        self.assertEqual(manager.current_engine, "elevenlabs")
        self.assertTrue(manager.is_available())
        self.assertEqual(manager.get_available_engines(), ["elevenlabs"])
        self.assertIs(manager.engines["elevenlabs"], fake_eleven.instances[0])

    def test_speak_uses_current_engine(self):
        manager, fake_gtts, _ = self._make_manager(
            gtts_available=True, eleven_available=False
        )

        result = manager.speak("안녕하세요")
        self.assertTrue(result)

        engine = fake_gtts.instances[0]
        self.assertIn(
            ("synthesize_and_play", "안녕하세요", manager.language), engine.calls
        )

    def test_speak_with_save_file_calls_synthesize_and_play(self):
        manager, fake_gtts, _ = self._make_manager(
            gtts_available=True, eleven_available=False
        )

        with mock.patch("os.path.getmtime", return_value=1.0):
            result = manager.speak("파일 저장", save_file=True)

        self.assertTrue(result)
        engine = fake_gtts.instances[0]
        synth_calls = [call for call in engine.calls if call[0] == "synthesize"]
        play_calls = [call for call in engine.calls if call[0] == "play"]
        self.assertTrue(synth_calls)
        self.assertTrue(play_calls)

    def test_test_priority_chain_runs_engine_test(self):
        manager, fake_gtts, _ = self._make_manager(
            gtts_available=True, eleven_available=True
        )
        fake_gtts.instances[0].calls.clear()

        self.assertTrue(manager.test_priority_chain("테스트 문장"))
        self.assertIn(("test", "테스트 문장"), fake_gtts.instances[0].calls)


if __name__ == "__main__":
    unittest.main()
