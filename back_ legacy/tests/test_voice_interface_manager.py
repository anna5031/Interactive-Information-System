import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class StubMicrophoneManager:
    def __init__(self) -> None:
        self.setup_called = False
        self.test_calls = []
        self.is_listening = False
        self.preferred_device_name = "Stub Mic"
        self.input_device_index = 1
        self.RATE = 16000
        self._stream_rate = 16000
        self._fallback = False

    def setup_microphone(self, device_index=None) -> bool:  # noqa: ANN001
        self.setup_called = True
        self.input_device_index = device_index if device_index is not None else 1
        return True

    def start_listening(self) -> None:
        self.is_listening = True

    def record_audio(self):
        return b"dummy-bytes"

    def stop_listening(self) -> None:
        self.is_listening = False

    def test_microphone(self, duration: float) -> bool:
        self.test_calls.append(("mic", duration))
        return True

    def save_audio_to_wav(self, audio_data: bytes, filename: str) -> None:  # noqa: D401
        self.test_calls.append(("save", filename, audio_data))

    def get_audio_devices(self):
        return [("0", "Stub Mic")], 0

    def cleanup(self) -> None:
        self.test_calls.append(("cleanup",))

    def get_stream_sample_rate(self) -> float:
        return float(self._stream_rate)

    def has_samplerate_fallback(self) -> bool:
        return bool(self._fallback)


class StubTTSManager:
    def __init__(self, preferred_engine: str | None = None) -> None:
        self.preferred_engine = preferred_engine
        self.calls = []
        self._available = True
        self.engines = ["gtts"]

    def is_available(self) -> bool:
        return self._available

    def speak(self, text: str, save_file: bool = False) -> bool:
        self.calls.append(("speak", text, save_file))
        return True

    def test_priority_chain(self, text: str) -> bool:
        self.calls.append(("test", text))
        return True

    def get_available_engines(self):
        return self.engines

    def get_current_engine_info(self):
        return {"engine_id": "gtts", "name": "Stub TTS"}

    def cleanup(self) -> None:
        self.calls.append(("cleanup",))


def load_voice_interface_module():
    features_pkg = sys.modules.setdefault("features", types.ModuleType("features"))
    features_pkg.__path__ = [str(PROJECT_ROOT / "features")]

    qa_pkg = sys.modules.setdefault("features.qa", types.ModuleType("features.qa"))
    qa_pkg.__path__ = [str(PROJECT_ROOT / "features" / "qa")]

    managers_pkg = sys.modules.setdefault(
        "features.qa.managers", types.ModuleType("features.qa.managers")
    )
    managers_pkg.__path__ = [str(PROJECT_ROOT / "features" / "qa" / "managers")]

    import importlib.util

    module_name = "features.qa.managers.voice_interface_manager"
    spec = importlib.util.spec_from_file_location(
        module_name,
        PROJECT_ROOT / "features" / "qa" / "managers" / "voice_interface_manager.py",
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)  # type: ignore[arg-type]
    return module


class VoiceInterfaceManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.module = load_voice_interface_module()

        mic_patch = patch.object(
            self.module, "MicrophoneManager", side_effect=StubMicrophoneManager
        )
        tts_patch = patch.object(
            self.module, "TTSManager", side_effect=StubTTSManager
        )

        self.addCleanup(mic_patch.stop)
        self.addCleanup(tts_patch.stop)

        self.mic_patch = mic_patch.start()
        self.tts_patch = tts_patch.start()

        self.manager = self.module.VoiceInterfaceManager()
        self.stub_mic = self.manager.mic_manager
        self.stub_tts = self.manager.tts_manager

    def test_initialization_success(self) -> None:
        self.assertTrue(self.manager.is_initialized)
        self.assertTrue(self.manager.is_ready())
        self.assertTrue(self.mic_patch.called)
        self.assertTrue(self.tts_patch.called)

    def test_speak_delegates_to_tts(self) -> None:
        result = self.manager.speak("테스트 음성", save_file=False)
        self.assertTrue(result)
        self.assertIn(("speak", "테스트 음성", False), self.stub_tts.calls)

    def test_listen_and_record_returns_bytes(self) -> None:
        audio = self.manager.listen_and_record()
        self.assertEqual(audio, b"dummy-bytes")

    def test_test_full_system_runs_checks(self) -> None:
        self.manager.test_full_system()
        self.assertIn(
            ("test", "음성 인터페이스 시스템이 정상적으로 작동합니다."),
            self.stub_tts.calls,
        )

    def test_cleanup_invokes_child_cleanup(self) -> None:
        self.manager.cleanup()
        self.assertIn(("cleanup",), self.stub_mic.test_calls)
        self.assertIn(("cleanup",), self.stub_tts.calls)


if __name__ == "__main__":
    unittest.main()
