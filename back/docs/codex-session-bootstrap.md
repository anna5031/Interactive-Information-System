# Codex 새 세션 부트스트랩 가이드

## 1. 목적
- 대화 토큰 한도 때문에 Codex 세션을 새로 열어야 할 때, 필수 컨텍스트를 빠르게 공유하기 위한 요약 문서.
- 현재 백엔드 구조, 실행 방법, QA 플로우, 테스트/디버깅 포인트를 한 장에 정리.
- 새로운 Codex 세션의 첫 메시지에 본 문서를 함께 첨부하면 이후 작업 지시가 수월해진다.

## 2. 현재 백엔드 요약
- 프로젝트 루트: `/Users/jaehyeon/Developer/Interactive-Information-System/back`
- 주요 진입점: `main.py`
- 런타임: Python 3.10 (로컬 venv `.venv` 사용 중)
- 핵심 구성:
  - `app/application.py`: 세션별 의존성 생성 (`SessionFlowCoordinator`, Stubs 등)
  - `websocket/server.py` & `websocket/session.py`: WebSocket 서버 및 세션 루프
  - `features/exploration/motor/homography/.../qa`: 각 기능 모듈 (현재는 더미 스텁 중심)
  - `docs/backend-architecture.md`: 백엔드 설계 상세 문서 (항상 최신 변경 반영)
- 프런트 문맥: `/Users/jaehyeon/Developer/Interactive-Information-System/front/docs/projector-ui-design.md`

## 3. 실행/테스트 방법
```bash
# venv 활성화
source /Users/jaehyeon/Developer/Interactive-Information-System/back/.venv/bin/activate\

# 백엔드 실행
python main.py
```
- QA 오디오 파이프라인 수동 점검:
  - `tests/test_voice_interface_manager.py` (단위/통합 테스트)
  - `tests/manual_voice_roundtrip.py` (마이크→STT→TTS 순환 수동 검증)
  - ElevenLabs API Key는 `.env` 경유로 로드됨 (`ELEVENLABS_API_KEY`)

## 4. 디버그 로그 토글
- `main.py` 상단의 상수 주석을 풀면 각각의 로그가 활성화됨:
  ```python
  # LOG_EXPLORATION = True
  # LOG_MOTOR = True
  # LOG_HOMOGRAPHY = True
  # LOG_COMMANDS = True
  ```
- 주석 해제 상태면 True로 간주되어 콘솔에 세부 로그가 표시된다.

## 5. 현재 QA 플로우 핵심
- `SessionFlowCoordinator`가 Vision/모터 이벤트를 감시해 상태를 전환:
  1. 탐지 신호가 `BACKEND_DETECTION_HOLD_SECONDS` 동안 지속되면 Landing 명령(`start_landing`, `start_nudge`) 큐잉.
  2. 모터가 허용 오차(`alignment_tolerance_deg`) 내에서 `alignment_hold_seconds` 유지되면 `start_qa` 발행.
  3. `start_qa`에 포함된 `context.initialPrompt`는 QA 컨트롤러가 듣기 단계에 들어가기 전에 TTS로 재생된다.
  4. `run_once` 흐름: `start_listening` → STT → `start_thinking` → TTS 출력(`start_speaking`/`stop_speaking`) → `stop_all`.
- QA 완료 후에는 Vision이 타겟을 잃어버렸을 때만 다음 세션이 준비되어 반복 QA가 중첩되지 않는다.

## 6. 더미 Exploration 시나리오
- `features/exploration/stub.py`는 내부 상태 머신(`SCANNING → TRACKING → COOLDOWN`)으로 움직임을 시뮬레이션한다:
  - 초기 `detection_delay` 후에 대상이 나타나고, `tracking_duration` 동안 좌표가 안정적으로 이동한다.
  - `cooldown_duration` 동안 대상이 사라져 후속 QA 테스트가 가능.
- 이를 통해 프런트 연결 없이도 QA 진입/종료 흐름을 반복적으로 검증할 수 있다.

## 7. 문서 & 소스 링크 모음
- 설계 문서: `docs/backend-architecture.md`
- 프런트 설계: `front/docs/projector-ui-design.md`
- QA 음성 구성: `features/qa/` 하위 (`config`, `managers`, `system`, `utils`)
- 탐색 스텁: `features/exploration/stub.py`
- 주요 설정: `app/config.py` (환경변수), `.env` (API 키 등 민감값)

## 8. 새 Codex 세션 시작 시 권장 전달 템플릿
```
1) 백엔드 루트: /Users/jaehyeon/Developer/Interactive-Information-System/back
2) 참고 문서:
   - docs/backend-architecture.md
   - docs/codex-session-bootstrap.md (본 문서)
   - front/docs/projector-ui-design.md
3) 현재 요구사항 요약:
   - SessionFlowCoordinator가 탐색→Landing→QA를 관리
   - QA 시작 전 initialPrompt를 음성으로 재생해야 함
   - 테스트는 tests/test_voice_interface_manager.py 또는 manual_voice_roundtrip.py로 수행
4) 실행 커맨드: source .venv/bin/activate && python main.py
```

---
필요 시 본 문서를 업데이트하고, 변경 사실을 `docs/backend-architecture.md`에도 함께 기록한다.
