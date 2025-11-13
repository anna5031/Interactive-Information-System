import os
import sys
import time

import numpy as np
import sounddevice as sd  # type: ignore

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from devices import load_device_preferences  # noqa: E402
from devices.microphone import check_microphone  # noqa: E402
from devices.speaker import check_speaker  # noqa: E402


def _select_device(result, priority_names):
    devices = result.meta.get("devices", []) if result.meta else []
    for name in priority_names:
        lowered = name.lower()
        for device in devices:
            if lowered in device["name"].lower():
                return device
    return devices[0] if devices else None


def configure_audio_devices():
    preferences = load_device_preferences()

    mic_result = check_microphone()
    spk_result = check_speaker()

    if not mic_result.ok:
        raise RuntimeError(f"마이크 감지 실패: {mic_result.detail}")
    if not spk_result.ok:
        raise RuntimeError(f"스피커 감지 실패: {spk_result.detail}")

    mic_device = _select_device(mic_result, preferences.microphone_priority_names)
    spk_device = _select_device(spk_result, preferences.speaker_priority_names)

    if mic_device is None or spk_device is None:
        raise RuntimeError("마이크 또는 스피커 장치를 선택할 수 없습니다.")

    sd.default.device = (mic_device["index"], spk_device["index"])

    return mic_device, spk_device


def record_audio(duration: float, samplerate: int = 44100):
    print(f"[녹음] {duration:.1f}초 동안 녹음합니다. 말을 하거나 소리를 내 주세요.")
    sd.default.samplerate = samplerate
    sd.default.channels = 1
    recording = sd.rec(int(duration * samplerate), dtype="float32")
    sd.wait()
    print("[녹음] 완료\n")
    return recording


def play_tone(duration: float, frequency: float = 440.0, samplerate: int = 44100):
    print(f"[재생] {frequency:.0f}Hz 테스트 톤을 {duration:.1f}초 동안 재생합니다.")
    t = np.linspace(0, duration, int(samplerate * duration), endpoint=False)
    tone = 0.2 * np.sin(2 * np.pi * frequency * t)
    sd.play(tone, samplerate=samplerate)
    sd.wait()
    print("[재생] 완료\n")


def play_recording(recording, samplerate: int = 44100):
    print("[재생] 방금 녹음한 오디오를 재생합니다.")
    sd.play(recording, samplerate=samplerate)
    sd.wait()
    print("[재생] 완료\n")


def main():
    mic_device, spk_device = configure_audio_devices()
    print(f"선택된 마이크: {mic_device['index']} - {mic_device['name']}")
    print(f"선택된 스피커: {spk_device['index']} - {spk_device['name']}\n")

    time.sleep(1.0)

    recording = record_audio(duration=3.0)
    play_tone(duration=1.0)
    play_recording(recording)

    print("오디오 라운드트립 테스트가 완료되었습니다.")


if __name__ == "__main__":
    main()
