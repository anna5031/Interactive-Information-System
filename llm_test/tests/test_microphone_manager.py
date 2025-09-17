"""sounddevice를 패치하여 MicrophoneManager를 검증하는 단위 테스트."""

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
            max=lambda *args, **kwargs: 0,
            abs=abs,
            sqrt=lambda _x: 0.0,
            ndarray=object,
        )

    if "sounddevice" not in sys.modules:

        class _DummyInputStream:
            def __init__(self, *args, **kwargs):
                self.started = False

            def start(self):
                self.started = True

            def stop(self):
                self.started = False

            def close(self):
                self.started = False

            def read(self, *args, **kwargs):
                return ([], False)

        sys.modules["sounddevice"] = SimpleNamespace(
            default=SimpleNamespace(samplerate=None, channels=None, dtype=None),
            query_devices=lambda: [],
            InputStream=_DummyInputStream,
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


BASE_AUDIO_CONFIG = {
    "chunk_size": 1024,
    "dtype": "int16",
    "channels": 1,
    "sample_rate": 16000,
    "silence_threshold": 500,
    "silence_duration": 2.0,
}


class MicrophoneManagerTests(unittest.TestCase):
    def _make_config(self, **overrides):
        config = {
            "selection_mode": "priority",
            "priority_device_names": [],
            "fallback_to_auto": True,
            "auto_priority": [],
            "audio": BASE_AUDIO_CONFIG.copy(),
        }
        config.update(overrides)
        return config

    def _patch_environment(self, config, devices):
        load_config = mock.patch(
            "src.managers.microphone_manager._load_microphone_config",
            return_value=config,
        )
        sd_patch = mock.patch("src.managers.microphone_manager.sd")
        load_config_cm = load_config.start()
        sd_mock = sd_patch.start()
        self.addCleanup(load_config.stop)
        self.addCleanup(sd_patch.stop)

        sd_mock.default = SimpleNamespace(samplerate=None, channels=None, dtype=None)
        sd_mock.query_devices.return_value = devices
        return load_config_cm, sd_mock

    def test_priority_name_match_selects_expected_device(self):
        from src.managers.microphone_manager import MicrophoneManager

        config = self._make_config(priority_device_names=["usb mic"])
        devices = [
            {
                "name": "USB Mic Pro",
                "max_input_channels": 1,
                "default_samplerate": 48000,
            },
            {
                "name": "Internal Mic",
                "max_input_channels": 1,
                "default_samplerate": 44100,
            },
        ]

        self._patch_environment(config, devices)
        manager = MicrophoneManager()
        input_devices, recommended = manager.get_audio_devices()

        self.assertEqual(recommended, 0)
        self.assertEqual(manager.preferred_device_name, "USB Mic Pro")
        self.assertEqual(input_devices, [(0, "USB Mic Pro"), (1, "Internal Mic")])

    def test_auto_priority_applies_when_name_mismatch(self):
        from src.managers.microphone_manager import MicrophoneManager

        config = self._make_config(
            priority_device_names=["nope"],
            auto_priority=[{"label": "USB", "keywords": ["usb"]}],
        )
        devices = [
            {
                "name": "Internal Mic",
                "max_input_channels": 1,
                "default_samplerate": 44100,
            },
            {
                "name": "External USB Interface",
                "max_input_channels": 2,
                "default_samplerate": 48000,
            },
        ]

        self._patch_environment(config, devices)
        manager = MicrophoneManager()
        input_devices, recommended = manager.get_audio_devices()

        self.assertEqual(recommended, 1)
        self.assertEqual(manager.preferred_device_name, "External USB Interface")
        self.assertEqual(input_devices[recommended], (1, "External USB Interface"))

    def test_priority_without_fallback_returns_none(self):
        from src.managers.microphone_manager import MicrophoneManager

        config = self._make_config(
            priority_device_names=["nope"],
            fallback_to_auto=False,
            auto_priority=[{"label": "USB", "keywords": ["usb"]}],
        )
        devices = [
            {
                "name": "Internal Mic",
                "max_input_channels": 1,
                "default_samplerate": 44100,
            },
        ]

        self._patch_environment(config, devices)
        manager = MicrophoneManager()
        _, recommended = manager.get_audio_devices()

        self.assertIsNone(recommended)
        self.assertIsNone(manager.preferred_device_name)

    def test_first_device_used_when_auto_fails(self):
        from src.managers.microphone_manager import MicrophoneManager

        config = self._make_config(
            priority_device_names=["nope"],
            auto_priority=[{"label": "No Match", "keywords": ["bluetooth"]}],
        )
        devices = [
            {
                "name": "Decklink Mic",
                "max_input_channels": 1,
                "default_samplerate": 44100,
            },
            {
                "name": "Fallback Device",
                "max_input_channels": 1,
                "default_samplerate": 44100,
            },
        ]

        self._patch_environment(config, devices)
        manager = MicrophoneManager()
        _, recommended = manager.get_audio_devices()

        self.assertEqual(recommended, 0)
        self.assertEqual(manager.preferred_device_name, "Decklink Mic")


if __name__ == "__main__":
    unittest.main()
