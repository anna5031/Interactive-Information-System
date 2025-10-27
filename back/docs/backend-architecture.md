# Projector Guidance Backend – 설계 문서

## 1. 목적
- 프론트엔드(`docs/projector-ui-design.md`)가 기대하는 WebSocket 프로토콜(`homography`, `command`, `ack`, `sync`)을 안정적으로 제공하는 백엔드 서버 설계.
- 카메라 기반 탐색/넛지와 QA 사이클을 오케스트레이션하며, YOLO 추론, 모터 제어, 음성 ASR/TTS, LLM 호출 등 주변 기능을 플러그인처럼 결합할 수 있는 구조 마련.
- 장기적으로는 실제 하드웨어/AI 서비스와 연계하되, 단기적으로는 컴포넌트별 더미 구현으로 개발 속도 확보.

## 2. 시스템 개요
- **실행 엔트리포인트**: `main.py` – 애플리케이션 수명 주기를 관리하고, 상태 머신과 비동기 태스크를 구동.
- **런타임**: Python 3.10+, `asyncio` 기반. CPU/GPU 작업은 별도 스레드/프로세스로 위임하거나 비동기 래퍼 사용.
- **통신**: WebSocket 서버 1개(`ws://<host>:<port>`). 단일 클라이언트(프로젝터 UI)를 상정하지만, 향후 동시 접속 확장을 고려해 세션 관리 계층 분리.
- **상태 축**: 
  - `Exploration` (탐색·넛지 단계)
  - `QA` (질의응답 루프)
- **주요 파이프라인**:
  - Vision → Detection Result Stream → Motor Control & Front Homography
  - SessionFlowCoordinator → Landing Command Script → QA Controller
  - QA Controller → 내부 음성/LLM 파이프라인 → Front Command & Overlay

## 3. 런타임 구성도(논리)
- `Application` (메인 컨테이너)
  - `StateManager`: 현재 상위 상태(`Exploration`/`QA`), 명령 진행 상황, ACK 추적.
  - `EventBus`: 내부 비동기 이벤트 전달(`asyncio.Queue` 기반). 도메인 모듈 간 결합도를 낮춤.
  - `WebSocketGateway`: 메시지 직렬화/역직렬화, 클라이언트 세션 관리, ACK 재전송 로직.
  - `ExplorationPipeline`: 카메라 인입 → YOLO 추론 → 관심 대상 위치/시선 계산.
  - `MotorController`: 탐색 결과/기존 상태를 바탕으로 팬틸트·프로젝터 보정 요청, Homography 계산기와 연동.
- `QAPipeline`: 음성 인식, 프롬프트 구성, LLM/TTS 호출, 상태 전환 명령 발행. 최초 안내 멘트는 `VoiceInterfaceManager`가 직접 TTS로 재생한다.
- `SessionFlowCoordinator`: Vision/모터 이벤트를 받아 Landing 스크립트와 QA 진입을 조율하고, detection hold·정렬 조건을 중앙집중식으로 관리.
  - `Diagnostics`: 로깅, 메트릭, 헬스체크.

## 4. 모듈 책임 및 예상 디렉터리 구조
```
back/
  main.py                     # 엔트리포인트 (asyncio.run)
  app/
    __init__.py
    application.py            # Application 컨테이너 (모듈 wiring)
    config.py                 # 환경변수/설정 로딩
    state.py                  # 상태 머신, CommandTracker, SessionState
    events.py                 # 내부 이벤트 타입 정의 (dataclass/TypedDict)
  websocket/
    __init__.py
    server.py                 # 실제 websockets 서버 구동, 세션 accept
    session.py                # 메시지 핸들러, ACK logic, heartbeat
    schemas.py                # WS 송수신 payload 모델 (pydantic/TypedDict)
  features/
    exploration/
      __init__.py
      stub.py                 # Vision/Motor 더미 데이터 생성 (초기 단계)
    motor/
      __init__.py
      stub.py                 # 팬틸트 제어 더미 구현
    homography/
      __init__.py
      stub.py                 # 더미 homography 행렬 생성기
    qa/
      __init__.py
      stub.py                 # QA 상태 전환용 더미 응답 시퀀스
  services/
    datastore.py              # 선택: 세션 로그, 임시 저장
    task_runner.py            # 백그라운드 워커 추상화 (필요 시)
  docs/
    backend-architecture.md   # (본 문서)
```
- 실제 하드웨어/API 연동은 `driver.py`/`client.py` 레이어에서 구현하고, 상위 `controller.py`는 인터페이스 기반으로 의존.
- 초기 구현에서는 각 `stub.py` 모듈이 더미 데이터를 생성해 `main.py`에서 orchestrate하며, 하드웨어/AI 연동 시 실제 구현으로 교체한다.
- 모든 컨트롤러는 `EventBus`에 이벤트를 발행하고 구독하는 방식으로 상호작용 (예: `VisionResultEvent`, `CommandAckEvent` 등).

## 5. WebSocket 계약 (백엔드 관점)
- 메시지 형식은 프론트 문서와 동일. 재사용 가능한 모델 정의:
  - `HomographyMessage`: `type="homography"`, `matrix`, `timestamp`, `validUntil`.
  - `CommandMessage`: `type="command"`, `commandId`, `sequence`, `action`, `context`.
  - `AckMessage`: `type="ack"`, `commandId`, `action`, `status`, `timestamp`.
  - `SyncMessage`: 재연결 시 현재 상태 및 마지막 명령을 전달.
- `WebSocketGateway` Responsibilities:
  - 연결 수립 시 `SyncMessage` 전송.
  - 내부 이벤트(`HomographyUpdated`, `CommandIssued`) → WS 메시지 변환.
  - 클라이언트 `ack` 수신 시 `StateManager`에 전달.
  - 타임아웃/재전송: `commandId`별 타이머 유지, `status="completed"` 수신 전까지 1초 간격 전송.
  - 헬스체크: 주기적 `ping/pong` 또는 keepalive 메시지.

## 6. 상태 머신 개요
- **전역 상태**
  - `Idle`: 클라이언트 연결 전 혹은 `stop_all` 이후.
  - `Exploration`: YOLO를 통해 도움이 필요한 사람 탐색. 결과에 따라 `start_nudge` 명령을 발행하고 모터 위치 갱신.
  - `QA`: 사용자가 빔으로 안내 받아 정지했다는 가정 하에 QA 루프 시작.
- **전환 규칙 (현재 구현)**  
  1. `Idle` → `Exploration`: 클라이언트 접속 후 `start_landing` 명령이 전달되면 탐색 루프가 시작된다.  
  2. `Exploration` → `Landing`: Vision 이벤트에서 `has_target=True` 상태가 `BACKEND_DETECTION_HOLD_SECONDS` 이상 유지되면 `SessionFlowCoordinator`가 Landing 스크립트를 큐잉한다 (`start_landing` → `start_nudge`).  
  3. `Landing` → `QA`: 모터 상태가 팬/틸트 모두 허용 오차(`alignment_tolerance_deg`) 이내로 `alignment_hold_seconds` 동안 유지되면 `start_qa` 명령과 함께 QA 파이프라인을 기동한다. `initialPrompt`는 QA 시작 전에 음성으로 재생된다.  
  4. `QA` → `Exploration`: QA 컨트롤러가 `stop_all` 명령을 발행하고 ACK가 완료되면 `SessionFlowCoordinator`가 재탐색을 허용한다 (새로운 타겟이 일정 시간 사라져야 재진입).
- `SessionFlowCoordinator`는 Vision/모터 신호를 기반으로 Landing·QA 조건을 평가하며, `CommandManager`는 각 명령의 ACK를 추적한다.

## 7. 탐색(Exploration) 파이프라인
- 카메라 스트림 수신 (RTSP/WebRTC/직접 장치) → 프레임 큐에 push.
- YOLO 추론기(추후 Ultralytics 등 교체 가능)는 가장 높은 우선순위의 사람 후보를 반환.
- `VisionResult` 데이터 모델:
  - `has_target: bool`
  - `target_position: tuple[float, float] | None` (정규화 좌표 0.0~1.0)
  - `gaze_vector: tuple[float, float] | None`
  - `confidence: float | None`
  - `timestamp: float`
- 결과 이벤트는 `EventBus`를 통해 모터 제어/상태 머신에 전달.
- `has_target == False`면 모터는 순회 모드 유지, 프론트에는 `homography` 업데이트만 지속된다.
- 타겟이 감지되면 `SessionFlowCoordinator`가 detection hold 타이머를 시작하고, `MotorController`는 목표 지점을 향해 팬/틸트를 수렴시킨다. 정렬이 완료될 때까지 Landing 상태가 유지된다.
- 더미 `ExplorationStub`은 `SCANNING → TRACKING → COOLDOWN` 상태 머신으로 동작하여 실서비스 전환 시나리오(탐지 지연, 추적 유지, 재탐색)를 재현한다.

## 8. 모터 제어 및 Homography
- `MotorController`는 하드웨어 추상화(`MotorDriver`)를 사용해 각도 명령을 비동기 전송.
- `MotorDriver`는 실제 장치 혹은 시뮬레이터에 대응하는 인터페이스. 개발 초기에는 더미 구현으로 로그만 기록.
- `HomographyGenerator`는 현재 모터 각도, 카메라-프로젝터 캘리브레이션 데이터, 타겟 좌표를 입력받아 3×3 행렬을 산출.
- 행렬 업데이트는 최소 10Hz를 유지하도록 스케줄링하며, 변경 없을 때도 keepalive 용으로 주기 전송.

## 9. QA 파이프라인
- `QAController`는 다음 서브 모듈을 조율:
  1. `ASRClient`: 마이크 입력 캡처 및 음성 인식 트리거. 초기에는 파일 재생/더미 텍스트로 대체 가능.
  2. `LLMClient`: 인식 결과를 기반으로 답변 생성. 외부 API 호출은 비동기 래퍼 또는 쓰레드 풀 사용.
  3. `TTSClient`: 답변을 음성으로 변환하고 재생 제어. 필요 시 별도 프로세스 활용.
- 상태 전환 명령 순서는 프론트 문서 기준(`start_listening` → `start_thinking` → `start_speaking` → `stop_speaking`).
- `start_qa` 명령의 `context.initialPrompt`는 QA 컨트롤러가 듣기 단계에 들어가기 전에 TTS로 재생되어 사용자가 안내 음성을 들을 수 있다.
- 각 단계에서 `CommandMessage` 발행 → ACK 추적 → 내부 작업 완료 시 다음 단계 이벤트 발행.
- 장애 처리: API 실패 시 `CommandMessage`로 오류 문구(`start_error` 등) 또는 `stop_all` 전달, 동시에 `StateManager` 복귀.

## 10. 구성/설정 및 관측성
- `config.py`는 `.env` 또는 환경변수 기반 설정 로딩 (포트, 카메라 URL, YOLO 모델 경로, API 키).
- 로깅: Python `logging` + 구조화 로깅(선택) → 콘솔 + 파일 옵션.
- 메트릭: 개발 초안에서는 `/metrics` HTTP 엔드포인트 대신 표준 로그 기반으로 대응. 향후 `prometheus_client` 추가 가능.
- 헬스체크: 
  - 내부 태스크 감시(`asyncio.TaskGroup`/커스텀 워처).
  - 외부: 단순 HTTP 또는 CLI 헬스체크 스크립트 (`python -m app.healthcheck`).

## 11. 개발 로드맵 (초안)
1. **프로젝트 스캐폴드**
   - `pyproject.toml`, 포에트리/uv 등 패키지 매니저 설정.
   - 기본 패키지 구조, 로깅/설정 헬퍼 작성.
2. **WebSocket 뼈대**
   - `WebSocketGateway` 구현, 더미 이벤트를 주기 전송.
   - 프론트와 연결 테스트 (`dummy_homography_generator/dumpy_server.py` 수준 기능).
3. **StateManager & CommandTracker**
   - 명령 발행/ACK 추적 로직 통합, 상태 전환 이벤트 정의.
4. **탐색 파이프라인 더미**
   - 랜덤/녹화 데이터 기반 Vision 결과 생성 → 모터/행렬 stub 연동.
5. **QA 파이프라인 더미**
   - 정해진 스크립트로 `start_listening` → `start_speaking` 흐름 재현.
6. **실장치/실서비스 연동**
   - 순차적으로 YOLO 모델, 모터 API, ASR/LLM/TTS 교체.
7. **관측/신뢰성 강화**
   - 에러 핸들링, 재시도, 타임아웃, 모니터링 보강.

## 12. 미해결 논의 사항
- YOLO 추론을 GPU 환경에서 실행할지, 외부 서비스로 위임할지 결정 필요.
- 모터/프로젝터 캘리브레이션 데이터 포맷 및 저장 위치.
- 다중 사람 감지 시 우선순위 규칙 (거리, 정면 여부 등).
- QA 파이프라인에서 ASR/LLM/TTS SLA 및 백업 전략.
- 서버 다중 인스턴스 운용 시 상태 동기화 방법 (현재는 단일 노드 가정).

---
본 문서는 초기 설계 기준이며, 구현 진행에 따라 각 모듈 문서화 및 API 스펙을 보강한다. 변경 사항은 `docs/backend-architecture.md`에 지속 반영한다.
