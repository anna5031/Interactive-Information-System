"""대화 흐름 전반을 검증하는 통합 시뮬레이션 테스트."""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


# --- 외부 의존성 더미 모듈 주입 -------------------------------------------------

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

if "pygame" not in sys.modules:
    sys.modules["pygame"] = SimpleNamespace(
        mixer=SimpleNamespace(init=lambda: None, quit=lambda: None)
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
            self.audio = SimpleNamespace(
                transcriptions=SimpleNamespace(
                    create=lambda *a, **k: SimpleNamespace(text="테스트")
                )
            )

    sys.modules["groq"] = SimpleNamespace(Groq=_DummyGroq)

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# --- 페이크 구현 ------------------------------------------------------------------


class FakeDeviceManager:
    def __init__(self):
        self.get_device_summary_called = False

    def get_device_summary(self):
        self.get_device_summary_called = True
        return {"usb_audio_devices": 1}


class FakeVoiceInterfaceManager:
    instances = []

    def __init__(self):
        self.__class__.instances.append(self)
        self.listen_calls = 0
        self.speak_messages = []
        self.ready = True
        self.mic_manager = SimpleNamespace(CHANNELS=1, RATE=16000)
        self.audio_queue = [b"audio-first", b"audio-exit"]

    def is_ready(self):
        return self.ready

    def test_tts(self, text: str):
        self.speak_messages.append(f"테스트:{text}")
        return True

    def listen_and_record(self):
        self.listen_calls += 1
        if self.audio_queue:
            return self.audio_queue.pop(0)
        return b"audio-empty"

    def speak(self, text: str, save_file: bool = False):
        self.speak_messages.append(text)
        return True

    def get_system_status(self):
        return {
            "initialized": True,
            "ready": True,
            "microphone": {
                "device_name": "Fake Mic",
                "device_index": 0,
                "available_devices": 1,
                "recommended_device": 0,
            },
            "tts": {
                "current_engine": "fake",
                "available_engines": ["fake"],
                "engine_count": 1,
            },
        }

    def cleanup(self):
        return None


class FakeSTTManager:
    def __init__(self):
        self.transcribe_calls = []
        self.transcripts = ["안녕", "quit"]

    def transcribe(self, audio_data, audio_format=None):
        self.transcribe_calls.append((audio_data, audio_format))
        if self.transcripts:
            return self.transcripts.pop(0)
        return "quit"

    def get_current_config(self):
        return {"model": "mock", "audio_format": {}, "parameters": {}}


class FakeLLMManager:
    def __init__(self):
        self.generate_calls = []
        self.test_connection_called = False

    def generate_response(self, user_text, conversation_history=None):
        self.generate_calls.append((user_text, tuple(conversation_history or [])))
        return "응답 메시지"

    def check_exit_command(self, text: str) -> bool:
        return text == "quit"

    def test_connection(self):
        self.test_connection_called = True
        return True

    def get_current_config(self):
        return {"llm_model": "mock", "chat_parameters": {}, "system_prompt": "prompt"}


# --- 테스트 케이스 -----------------------------------------------------------------


class ConversationFlowTests(unittest.TestCase):
    def setUp(self):
        patchers = [
            mock.patch("main.DeviceManager", FakeDeviceManager),
            mock.patch("main.VoiceInterfaceManager", FakeVoiceInterfaceManager),
            mock.patch("main.STTManager", FakeSTTManager),
            mock.patch("main.LLMManager", FakeLLMManager),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)

        from main import VoiceAISystem

        self.system = VoiceAISystem()

    def test_test_system_runs_all_dependencies(self):
        self.assertTrue(self.system.test_system())
        self.assertTrue(self.system.llm_manager.test_connection_called)
        self.assertGreaterEqual(len(self.system.voice_manager.speak_messages), 1)

    def test_conversation_loop_handles_exit_flow(self):
        with mock.patch("builtins.print"):
            self.system.run_conversation_loop()

        voice_messages = self.system.voice_manager.speak_messages
        self.assertTrue(voice_messages[0].startswith("음성 AI 시스템이 준비되었습니다"))
        self.assertEqual(voice_messages[1], "응답 메시지")
        self.assertEqual(voice_messages[-1], "안녕히 가세요!")
        self.assertGreaterEqual(self.system.voice_manager.listen_calls, 2)
        self.assertEqual(len(self.system.llm_manager.generate_calls), 1)
        self.assertEqual(self.system.conversation_history[-1]["content"], "응답 메시지")

    def test_get_system_status_returns_expected_shape(self):
        status = self.system.get_system_status()
        self.assertIn("voice_system", status)
        self.assertIn("llm_config", status)
        self.assertIn("stt_config", status)
        self.assertEqual(status["voice_system"]["tts"]["current_engine"], "fake")


if __name__ == "__main__":
    unittest.main()
