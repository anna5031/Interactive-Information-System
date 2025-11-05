"""
마이크 설정 파일
"""

try:  # 사용자 맞춤 우선순위 목록 (git에 포함되지 않음)
    from microphone_priority import PRIORITY_DEVICE_NAMES  # type: ignore
except ImportError:
    PRIORITY_DEVICE_NAMES = []

MICROPHONE_CONFIG = {
    # 마이크 선택 모드: 'auto' 또는 'priority'
    "selection_mode": "priority",
    # priority 모드일 때 우선 선택할 디바이스 이름(부분 일치)
    "priority_device_names": PRIORITY_DEVICE_NAMES,
    # priority 매칭 실패 시 auto 규칙으로 재시도할지 여부
    "fallback_to_auto": True,
    # auto 모드에서 사용할 키워드 우선순위 목록 (상단부터 탐색)
    "auto_priority": [
        {
            "label": "USB 마이크",
            "keywords": ["usb", "microphone", "mic", "headset", "webcam"],
        },
        {"label": "PulseAudio 기본 소스", "keywords": ["pulse", "default"]},
    ],
    # 기본 오디오 설정
    "audio": {
        "chunk_size": 1024,
        "dtype": "int16",
        "channels": 1,
        "sample_rate": 16000,
        "silence_threshold": 500,
        "silence_duration": 2.0,
    },
}
