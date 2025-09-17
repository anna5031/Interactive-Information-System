"""
TTS 설정 파일
"""

# ElevenLabs 설정
ELEVENLABS_CONFIG = {
    "voice_id": "AW5wrnG1jVizOYY7R1Oo",
    "model_id": "eleven_multilingual_v2",
    # "model_id": "eleven_flash_v2_5",
    "voice_settings": {
        "stability": None,
        "similarity_boost": None,
        "style": 0.0,
        "use_speaker_boost": False,
        "speed": 1.0,
    },
    "supported_languages": ["ko", "en", "ja", "zh"],
}

# gTTS 설정
GTTS_CONFIG = {
    "language": "ko",
    "slow": False,
    "supported_languages": ["ko", "en", "ja", "zh"],
}

# TTS 관리자 설정
TTS_MANAGER_CONFIG = {
    "default_engine": "auto",
    "engine_priority": ["elevenlabs", "gtts"],
    # "engine_priority": ["gtts", "elevenlabs"],
    "language": "ko",
}
