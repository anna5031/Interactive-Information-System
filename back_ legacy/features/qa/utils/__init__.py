"""Utility helpers and TTS engine adapters for QA feature."""

from .audio import ensure_wav_bytes, normalize_audio_level, play_audio_file
from .gtts_engine import GTTSEngine

try:
    from .elevenlabs_engine import ElevenLabsEngine  # type: ignore
except Exception:  # pragma: no cover
    ElevenLabsEngine = None  # type: ignore

from .tts_factory import TTSFactory

__all__ = [
    "ensure_wav_bytes",
    "normalize_audio_level",
    "play_audio_file",
    "GTTSEngine",
    "ElevenLabsEngine",
    "TTSFactory",
]
