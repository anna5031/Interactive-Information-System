from __future__ import annotations

import io
import os
import shutil
import subprocess
import sys
import wave
from typing import Iterable

import numpy as np


def ensure_wav_bytes(
    audio_frames: bytes,
    channels: int,
    sample_width: int,
    frame_rate: int,
) -> bytes:
    """Raw PCM 바이트를 WAV 포맷 바이트로 감싼다."""
    if not audio_frames:
        return b""

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(audio_frames)
    return buffer.getvalue()


def normalize_audio_level(
    samples: np.ndarray,
    target_peak: float = 0.9,
) -> np.ndarray:
    """오디오 샘플을 목표 peak 에 맞춰 정규화."""
    if samples.size == 0:
        return samples

    peak = np.max(np.abs(samples))
    if peak == 0:
        return samples

    scale = target_peak / peak
    return np.clip(samples * scale, -1.0, 1.0)


def play_audio_file(path: str) -> bool:
    """외부 플레이어를 사용해 오디오 파일을 재생."""
    if not os.path.exists(path):
        print(f"❌ Audio file not found: {path}")
        return False

    commands: Iterable[list[str]] = []

    if sys.platform == "darwin":
        commands = [["afplay", path]]
    elif sys.platform.startswith("linux"):
        commands = [
            ["mpg123", "-q", path],
            ["ffplay", "-nodisp", "-autoexit", path],
            ["aplay", path],
        ]
    elif sys.platform.startswith("win"):
        commands = [["powershell", "-Command", f"(New-Object Media.SoundPlayer '{path}').PlaySync();"]]

    for cmd in commands:
        if shutil.which(cmd[0]):
            try:
                subprocess.run(cmd, check=False)
                return True
            except Exception:
                continue

    print("⚠️ No supported audio player found (afplay/mpg123/ffplay/aplay).")
    return False
