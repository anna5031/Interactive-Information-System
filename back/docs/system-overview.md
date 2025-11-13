# Interactive Information System – Backend Overview

이 문서는 현재 `back/` 코드베이스의 구조, 주요 모듈, 설정 파일, 수동/자동 테스트 방법을 정리한 레퍼런스입니다. 새 Codex 세션에서 빠르게 컨텍스트를 파악할 수 있도록 작성되었습니다.

---

## 1. 최상위 실행 흐름

```
python main.py
 └─ runtime.application.Application
     ├─ DeviceManager.initialize()
     ├─ SessionRunnerFactory.create()
     │   ├─ ExplorationPipeline
     │   └─ QAPipeline
     └─ WebSocketServer.serve_forever()
```

- **DeviceManager** (`devices/manager.py`)는 마이크·스피커·카메라·아두이노를 점검하고, `config/device_preferences.py`에 정의된 우선순위·카메라 소스/프레임 크기(`CAMERA_SOURCE`, `CAMERA_FRAME_SIZE`)로 OS 기본 장치를 설정합니다.
- **SessionRunnerFactory**는 `ExplorationPipeline`(YOLO 탐색)과 `QAPipeline`(음성 QA)을 생성합니다. 플래그 `USE_DUMMY_ARDUINO`, `USE_DUMMY_NUDGE_PASS` 등은 `main.py`에서 Boolean 변환 후 주입됩니다.
- **WebSocketServer** (`websocket/server.py`)는 각 클라이언트 연결마다 `SessionRunner`를 실행해 탐색→QA 루프를 돌립니다.

---

## 2. Exploration Pipeline

위치: `features/exploration/`

구성:
- `pipeline.py` – 카메라 캡처, YOLO pose 추론, 추적/AssistanceClassifier orchestration
- `core/` – 추적(`tracking.py`), AssistanceClassifier(`assistance.py`), FootPointEstimator 등 핵심 로직
- `detection/` – YOLO 결과에서 keypoint 추출
- `geometry/` – `PixelToWorldMapper`
- `io/` – 카메라 캡처(`capture.py`), 모델 래퍼(`model.py`)
- `visual/` – 디버그 오버레이
- `config.py` – Tracking/Detection/Assistance/Mapping dataclass

### Assistance 흐름

1. `AssistanceClassifier.evaluate()`는 정지 조건을 만족한 트랙을 탐색
2. `_should_dismiss()`가 거리·시간·더미 플래그를 검사
3. `dummy_nudge_pass_enabled=True`일 경우 `dummy_nudge_pass_seconds` 경과 시 `"dummy-pass"` 이유로 해제하고 `transition_to_qa=True`를 돌려줍니다.
4. `SessionRunner`는 `ExplorationPipeline._last_assistance_decision.transition_to_qa`를 확인해 QA로 넘어갑니다.

### Mapping 설정

- `features/exploration/geometry/mapping.py`는 `MappingConfig`에서 calibration 경로를 읽고, `PixelToWorldMapper.pixel_to_world()`로 Z 평면 좌표를 반환합니다.
- 캘리브레이션 파일 기본 위치: `features/exploration/calibration/camera_calib.npz`, `camera_extrinsics.npz`
- 평면 Z는 `MappingConfig.floor_z_mm`로 설정 (기본 0)

---

## 3. Voice / QA Pipeline

위치: `features/qa/`

구성:
- `audio/` : `MicrophoneManager`, `STTManager`, `TTSManager`, TTS 엔진(`gtts_engine`, `elevenlabs_engine`) 및 `audio_utils`
- `services/voice_io.py` : `VoiceIOService` – 녹음, STT, TTS를 하나의 서비스로 제공
- `pipeline.py` : 새 `QAPipeline` – 인사(TTS) → 녹음 및 STT → 응답 생성 → 응답 TTS
- `config/qa/` (루트에 위치) : `microphone_config.py`, `microphone_priority.py`, `stt_config.py`, `tts_config.py`

설정 로딩:
- 각 audio 모듈은 `config.qa.<file>`을 직접 import (sys.path 조작 없음)
- 예: `config/qa/microphone_priority.py`에서 `PRIORITY_DEVICE_NAMES = ["AirPods", "MacBook", ...]` 정의

STT/TTS 의존성:
- Groq API (`GROQ_API_KEY`)와 ElevenLabs (`ELEVENLABS_API_KEY`)는 `.env`에 저장
- gTTS를 사용할 경우 네트워크 연결 필요

---

## 4. config / .env

공통 규칙:
- `.env`는 `back/.env` 하나로 관리 (API 키 등)
- `config/` 폴더에는 런타임 설정을 Python 모듈 형태로 저장

주요 파일:
- `config/device_preferences.py` – DeviceManager 우선순위, `CAMERA_SOURCE`, `CAMERA_FRAME_SIZE=(1280, 720)`
- `config/motor_settings.yaml` – 모터/프로젝터 설정 (MotorController, HomographyCalculator)
- `config/qa/*` – 마이크/TTs/STT 설정 (레거시에서 이동)

환경 변수 예시 (`.env`):
```
GROQ_API_KEY=...
ELEVENLABS_API_KEY=...
```

---

## 5. Motor & Homography

위치: `features/motor/`, `features/homography/`

- Motor:
  - `config.py` – `MotorSettings` dataclass (`motor_settings.yaml` 로드)
  - `setpoint.py` – 목표 좌표→각도 계산 (`SetpointCalculator`)
  - `driver.py` – `MotorDriver` 프로토콜, `SerialMotorDriver`, `DummyMotorDriver`
  - `controller.py` – `MotorController.point_to(target_xyz)` 고수준 API
- Homography:
  - `calculator.py` – `HomographyCalculator` 클래스

수동 스크립트:
- `tests/manual_target_nudge.py` – 픽셀→월드→모터→호모그래피 전체 파이프라인을 수동 테스트 (더미/실제 모터 선택 가능)

---

## 6. Tests & Manual Scripts

### Pytest Suites

- `tests/exploration/test_assistance_dummy.py` – 더미 넛지 패스 로직이 QA 전환 신호를 내는지 확인
- `tests/motor/*` – Motor controller/setpoint 유닛 테스트
- `tests/homography/test_calculator.py` – Homography matrix 계산 기본 확인

실행:
```
source .venv/bin/activate
PYTHONPATH=. pytest tests/exploration tests/motor tests/homography
```

### Manual Scripts

- `tests/manual_exploration_stream.py` – 카메라/YOLO 탐색 수동 실행
- `tests/manual_audio_roundtrip.py` – 간단한 오디오 테스트
- `tests/manual_target_nudge.py` – 픽셀 입력 기반 모터/호모그래피 테스트

레거시 저장소(`back_legacy/`) 참고 스크립트:
- `tests/manual_voice_service.py` – VoiceIOService의 TTS/STT/녹음을 직접 실행 (개발 환경에서 레거시 확인용)

---

## 7. 플래그 및 토글

`main.py` 상단:
```python
SHOW_EXPLORATION_OVERLAY = True
USE_DUMMY_ARDUINO = True
USE_DUMMY_NUDGE_PASS = False
```

사용 시:
- **`USE_DUMMY_ARDUINO`**: DeviceManager가 아두이노 점검을 건너뜀
- **`USE_DUMMY_NUDGE_PASS`**: 타겟이 3초(또는 `dummy_nudge_pass_seconds`) 경과 시 거리 조건 없이 QA 단계로 이동
- **`SHOW_EXPLORATION_OVERLAY`**: OpenCV 창 표시 (Q, ESC로 종료)

---

## 8. TODO / Follow-ups

1. **QA Pipeline 고도화**
   - 현재 QAPipeline은 간단한 인사/녹음/응답 플로우만 구현. 레거시 `QAController`(SessionFlowCoordinator, multi-turn) 이식 필요
   - VoiceIOService를 활용해 Groq/RAG 응답을 실제로 생성하고, 다중 응답/종료 조건 구현
2. **Motor/Homography 통합**
   - 현재 MotorController와 HomographyCalculator는 수동 스크립트 수준. 탐색→유도 파이프라인과 연동 필요
3. **Config 일원화**
   - `config/qa`는 Python 모듈 기반. 필요 시 YAML → dataclass 로더 추가 고려
4. **Logging/Monitoring**
   - 장치 선택 실패/성공 로그는 존재하지만, QA/음성 단계의 오류 수집을 위한 추가 메트릭 필요
5. **Docs 유지**
   - 향후 구조 변경 시 본 문서를 업데이트해 Codex 컨텍스트와 실제 코드 간 불일치가 없도록 유지

---

## 9. 참고 Links

- `back/docs/exploration-pipeline-notes.md` – YOLO 탐색 파이프라인 메모
- `back_legacy/features/qa/` – 레거시 QA 시스템 전체 (Controller, managers, utils)
- GitHub Issues / TODOs – 없음 (수기로 관리 중)

이 문서는 2025-11-13 기준 상태를 반영합니다. 이후 변경사항이 발생하면 반드시 업데이트해주세요.
