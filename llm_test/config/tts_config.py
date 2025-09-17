"""
TTS 설정 파일
"""

# ElevenLabs 설정
ELEVENLABS_CONFIG = {
    "voice_id": "AW5wrnG1jVizOYY7R1Oo",
    "model_id": "eleven_multilingual_v2",
    "voice_settings": {
        "stability": 0.5,
        "similarity_boost": 0.5,
        "style": 0.0,
        "use_speaker_boost": True,
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
    "language": "ko",
}
