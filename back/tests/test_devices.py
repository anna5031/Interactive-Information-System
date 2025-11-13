import os
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from devices.microphone import check_microphone
from devices.speaker import check_speaker


class TestDeviceChecks(unittest.TestCase):
    @patch("devices.microphone.sd.query_devices")
    def test_check_microphone_ok(self, mock_query_devices):
        mock_query_devices.return_value = [
            {"name": "USB Mic", "max_input_channels": 2},
            {"name": "Line In", "max_input_channels": 0},
        ]

        result = check_microphone()

        self.assertTrue(result.ok)
        self.assertIn("USB Mic", result.detail)
        self.assertIn("devices", result.meta)
        self.assertEqual(result.meta["devices"][0]["index"], 0)
        self.assertEqual(result.meta["devices"][0]["name"], "USB Mic")

    @patch("devices.microphone.sd.query_devices")
    def test_check_microphone_no_devices(self, mock_query_devices):
        mock_query_devices.return_value = [
            {"name": "Line In", "max_input_channels": 0},
        ]

        result = check_microphone()

        self.assertFalse(result.ok)
        self.assertIn("입력 장치를 찾을 수 없습니다.", result.detail)

    @patch("devices.speaker.sd.query_devices")
    def test_check_speaker_ok(self, mock_query_devices):
        mock_query_devices.return_value = [
            {"name": "External Speaker", "max_output_channels": 2},
            {"name": "Dummy", "max_output_channels": 0},
        ]

        result = check_speaker()

        self.assertTrue(result.ok)
        self.assertIn("External Speaker", result.detail)
        self.assertIn("devices", result.meta)
        self.assertEqual(result.meta["devices"][0]["index"], 0)
        self.assertEqual(result.meta["devices"][0]["name"], "External Speaker")

    @patch("devices.speaker.sd.query_devices")
    def test_check_speaker_no_devices(self, mock_query_devices):
        mock_query_devices.return_value = [
            {"name": "Dummy", "max_output_channels": 0},
        ]

        result = check_speaker()

        self.assertFalse(result.ok)
        self.assertIn("출력 장치를 찾을 수 없습니다.", result.detail)


if __name__ == "__main__":
    unittest.main()
