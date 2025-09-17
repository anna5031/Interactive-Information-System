"""Groq 클라이언트를 모킹해 LLMManager를 검증하는 단위 테스트."""

import os
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock


def _ensure_stub_modules():
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
        sys.modules["dotenv"] = SimpleNamespace(
            load_dotenv=lambda *args, **kwargs: None
        )

    if "groq" not in sys.modules:

        class _DummyGroq:
            def __init__(self, *args, **kwargs):
                self.chat = SimpleNamespace(
                    completions=SimpleNamespace(
                        create=lambda *a, **k: SimpleNamespace(choices=[])
                    )
                )

        sys.modules["groq"] = SimpleNamespace(Groq=_DummyGroq)


_ensure_stub_modules()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BASE_LLM_CONFIG = {
    "llm_model": "fake-llm",
    "chat_parameters": {
        "temperature": 0.5,
        "max_tokens": 128,
        "top_p": 1.0,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "stream": False,
    },
}


class LLMManagerTests(unittest.TestCase):
    def _make_manager(self):
        from src.managers.llm_manager import LLMManager

        load_config_patch = mock.patch(
            "src.managers.llm_manager._load_llm_config",
            return_value=BASE_LLM_CONFIG.copy(),
        )
        dotenv_patch = mock.patch("src.managers.llm_manager.load_dotenv")
        groq_patch = mock.patch("src.managers.llm_manager.Groq")
        prompt_patch = mock.patch.object(
            LLMManager, "_load_system_prompt", return_value="system instructions"
        )
        env_patch = mock.patch.dict(
            "os.environ", {"GROQ_API_KEY": "dummy"}, clear=False
        )

        load_config_patch.start()
        self.addCleanup(load_config_patch.stop)
        dotenv_patch.start()
        self.addCleanup(dotenv_patch.stop)
        prompt_patch.start()
        self.addCleanup(prompt_patch.stop)
        env_patch.start()
        self.addCleanup(env_patch.stop)

        mock_groq = groq_patch.start()
        self.addCleanup(groq_patch.stop)

        mock_chat = mock.Mock()
        mock_response = SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="테스트 응답"))]
        )
        mock_chat.completions.create.return_value = mock_response
        mock_client = mock.Mock(chat=mock_chat)
        mock_groq.return_value = mock_client

        manager = LLMManager()
        return manager, mock_chat

    def test_generate_response_uses_configured_client(self):
        manager, mock_chat = self._make_manager()

        result = manager.generate_response(
            "안녕", conversation_history=[{"role": "user", "content": "hi"}]
        )

        self.assertEqual(result, "테스트 응답")
        mock_chat.completions.create.assert_called_once()
        call_kwargs = mock_chat.completions.create.call_args.kwargs
        self.assertIn(
            {"role": "system", "content": "system instructions"},
            call_kwargs["messages"],
        )
        self.assertEqual(call_kwargs["model"], "fake-llm")

    def test_exit_command_detection(self):
        manager, _ = self._make_manager()

        self.assertTrue(manager.check_exit_command("종료해"))
        self.assertTrue(manager.check_exit_command("please exit"))
        self.assertFalse(manager.check_exit_command("계속 진행"))

    def test_set_llm_model_updates_value(self):
        manager, _ = self._make_manager()

        manager.set_llm_model("another-model")
        self.assertEqual(manager.llm_model, "another-model")

    def test_test_connection_handles_success(self):
        manager, mock_chat = self._make_manager()
        mock_chat.completions.create.reset_mock()
        mock_chat.completions.create.return_value = SimpleNamespace(choices=[])

        self.assertTrue(manager.test_connection())
        mock_chat.completions.create.assert_called_once()


if __name__ == "__main__":
    unittest.main()
