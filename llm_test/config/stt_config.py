"""
STT (Speech-to-Text) 설정 파일
"""

STT_CONFIG = {
    "model": "whisper-large-v3-turbo",
    "parameters": {"language": "ko", "temperature": 0.0},
    "audio_format": {"channels": 1, "sample_width": 2, "frame_rate": 16000},
}
