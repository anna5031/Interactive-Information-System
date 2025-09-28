# 음성 AI 시스템

Groq API와 구성 파일 기반 음성 엔진을 결합해 실시간 대화를 제공하는 모듈형 런타임입니다. 마이크/스피커 설정, 음성 인식(STT), LLM 응답 생성, 음성 합성(TTS)을 각각의 매니저로 분리해 유지보수성과 확장성을 확보했습니다.

## 주요 특징

- 🎙️ **구성 기반 음성 인터페이스**: `config/microphone_config.py` 설정과 우선순위 파일을 활용해 환경에 맞는 입력 장치를 자동 선택합니다.
- 🧠 **Groq LLM 연동**: `config/llm_config.py`에서 모델과 채팅 파라미터를 제어하며 대화 히스토리를 기반으로 응답을 생성합니다.
- 🗣️ **다중 TTS 엔진**: `config/tts_config.py`의 우선순위에 따라 ElevenLabs → gTTS 순으로 시도하며 실패 시 자동으로 대체합니다.
- ♻️ **대화 루프와 셀프 테스트**: 시작 시 시스템 상태를 보고하고, 옵션에 따라 장치·API 연결 검사를 수행합니다.
- 🧪 **단위/통합 테스트 세트**: 주요 매니저와 전체 대화 흐름을 검증하는 `tests/` 스위트를 제공합니다.

## 필수 요구사항

- macOS 또는 Linux (PulseAudio/ALSA 환경 권장)
- Python 3.10 이상
- 마이크와 스피커가 연결된 시스템
- Groq API 키 (필수), ElevenLabs API 키 (선택)

### 권장 시스템 패키지 (Ubuntu/Debian 계열)

```bash
sudo apt update
sudo apt install -y pulseaudio pulseaudio-utils alsa-utils ffmpeg
```

## 설치

```bash
git clone <repository>
cd Interactive-Information-System/llm_test
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

## 환경 변수 및 API 키

프로젝트 루트에 `.env` 파일을 생성하고 다음을 설정합니다.

```env
GROQ_API_KEY=your_groq_api_key
# 선택: ElevenLabs를 사용할 때만 필요한 키
ELEVENLABS_API_KEY=your_elevenlabs_api_key

# 옵션: 시작 시 셀프 테스트 실행
STARTUP_TESTS=true
```

`STARTUP_TESTS`를 설정하거나 `python main.py --self-test`로 실행하면 STT/TTS/LLM 연결 점검을 자동으로 수행합니다.

## 구성 커스터마이징

- `config/llm_config.py`: 사용할 Groq 모델과 채팅 파라미터(temperature, max_tokens 등)를 정의합니다.
- `config/stt_config.py`: Whisper 모델, 언어, 오디오 포맷을 지정합니다.
- `config/tts_config.py`: TTS 엔진 우선순위, 언어, ElevenLabs/gTTS 세부 설정을 조정합니다.
- `config/microphone_config.py`: 마이크 자동 탐지 규칙, 샘플링 파라미터, 침묵 감지 임계값을 제어합니다.
- `config/microphone_priority.py.example`: 사용자 장비명에 맞춰 복사·수정하면 개인화된 우선순위를 적용할 수 있습니다 (`microphone_priority.py`는 git에 커밋되지 않습니다).
- `config/base_prompt.md`: LLM 시스템 프롬프트를 텍스트로 관리합니다.

## 실행 방법

```bash
python main.py
```

실행 시 출력되는 시스템 상태에서 선택된 마이크, TTS 엔진, LLM/STT 모델을 확인할 수 있습니다. 종료하려면 마이크에 "종료", "quit" 등을 말하거나 `Ctrl+C`를 누르세요.

## 프로젝트 구조

```
llm_test/
├── main.py                     # 음성 AI 런타임 진입점
├── requirements.txt            # 프로젝트 의존성 목록
├── config/
│   ├── base_prompt.md          # 시스템 프롬프트 텍스트
│   ├── llm_config.py           # Groq 모델 및 채팅 파라미터
│   ├── microphone_config.py    # 마이크 선택/오디오 설정
│   ├── microphone_priority.py.example  # 사용자 우선순위 예시
│   ├── stt_config.py           # Whisper 모델 및 파라미터
│   └── tts_config.py           # TTS 엔진 우선순위와 설정
├── src/
│   ├── system/
│   │   └── voice_ai_system.py          # LLM/STT/TTS를 조합한 대화 엔진
│   ├── managers/
│   │   ├── device_manager.py           # USB 오디오 장치 요약
│   │   ├── llm_manager.py              # Groq API 연동 로직
│   │   ├── microphone_manager.py       # 입력 장치 제어
│   │   ├── stt_manager.py              # 음성 인식 제어
│   │   ├── tts_manager.py              # 합성 엔진 선택/재생
│   │   └── voice_interface_manager.py  # 마이크·TTS 통합 인터페이스
│   └── utils/
│       ├── elevenlabs_engine.py       # ElevenLabs 어댑터
│       ├── gtts_engine.py             # gTTS 어댑터
│       └── tts_factory.py             # 엔진 생성 헬퍼
├── tests/
│   ├── test_env.py                    # 테스트 공용 초기화
│   ├── test_integration_system.py     # 대화 루프 통합 테스트
│   ├── test_llm_manager.py            # LLM 매니저 단위 테스트
│   ├── test_microphone_manager.py     # 마이크 매니저 단위 테스트
│   └── test_tts_manager.py            # TTS 매니저 단위 테스트
└── temp/                              # 임시 오디오/출력 보관
```

## 테스트 실행

오디오/네트워크 의존성이 없는 더미 모듈을 사용하도록 구성되어 있어 로컬에서 안전하게 검증할 수 있습니다.

```bash
python -m unittest discover tests
```

개별 테스트는 `python -m unittest tests.test_llm_manager`와 같이 실행할 수 있습니다.

## 문제 해결

- **오디오 장치가 감지되지 않음**: `python main.py --self-test`로 장치 로그를 확인하고, `pactl list short sinks/sources` 또는 `arecord -l`로 시스템 인식을 점검하세요.
- **TTS가 재생되지 않음**: `config/tts_config.py`의 엔진 우선순위를 확인하고, 필요한 API 키 및 네트워크 접근 권한을 설정합니다.
- **Groq API 오류**: `.env`의 키 값과 네트워크 연결을 확인한 뒤 `tests/test_llm_manager.py`에서 제공하는 모의 테스트를 활용해 로컬 환경을 검증합니다.

## 라이선스

MIT License
