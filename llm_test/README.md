# 젯슨 오린 나노 음성 AI 시스템

Groq API와 gTTS를 사용한 실시간 음성 대화 시스템입니다.

## 주요 기능

- 🎤 실시간 음성 인식
- 🤖 AI 응답 생성
- 🔊 음성 합성 및 재생
- 🔌 USB 오디오 디바이스 자동 인식 및 설정
- 📁 모듈화된 구조로 관리 용이

## 시스템 요구사항

- Python 3.8+
- NVIDIA Jetson Orin Nano (또는 호환 Linux 시스템)
- USB 스피커/마이크 (권장)
- 인터넷 연결 (Groq API 사용)

## 설치 및 설정

### 1. 패키지 설치

```bash
# 시스템 패키지
sudo apt update
sudo apt install -y pulseaudio pulseaudio-utils alsa-utils ffmpeg

# Python 패키지
pip install -r requirements.txt
```

### 2. 환경 설정

`.env` 파일에 Groq API 키를 설정하세요:

```env
GROQ_API_KEY=your_groq_api_key_here
```

### 3. 실행

```bash
python main.py
```

## 프로젝트 구조

```
jetson_llm/
├── main.py                    # 메인 실행 파일
├── .env                       # 환경 변수 설정
├── requirements.txt           # Python 패키지 의존성
├── src/
│   ├── managers/             # 핵심 관리 모듈들
│   │   ├── audio_manager.py     # 오디오 출력 관리
│   │   ├── microphone_manager.py # 마이크 입력 관리
│   │   ├── llm_manager.py       # LLM/Whisper 관리
│   │   └── device_manager.py    # USB 디바이스 관리
│   ├── config/               # 설정 파일들
│   │   └── base_prompt.md       # 시스템 프롬프트
│   └── utils/                # 유틸리티 도구들
│       └── speaker_test.py      # 스피커 테스트 도구
└── README.md                 # 이 파일
```

## 사용법

1. **시스템 시작**: `python main.py` 실행
2. **음성 입력**: 마이크에 대고 말하기
3. **종료**: "종료", "quit", "exit" 등의 명령어 또는 Ctrl+C

## 트러블슈팅

### 오디오 문제

```bash
# 스피커 테스트
python src/utils/speaker_test.py

# 오디오 디바이스 확인
pactl list short sinks
pactl list short sources
```

### USB 디바이스 인식 문제

- USB 연결 확인: `lsusb`
- PulseAudio 재시작: `pulseaudio -k && pulseaudio --start`

## 설정 커스터마이징

### 시스템 프롬프트 수정

`src/config/base_prompt.md` 파일을 편집하여 AI의 응답 스타일을 변경할 수 있습니다.

### 모델 변경

`.env` 파일에서 모델을 변경할 수 있습니다:

```env
WHISPER_MODEL=whisper-large-v3-turbo
LLM_MODEL=llama-3.1-8b-instant
```

## 라이선스

MIT License