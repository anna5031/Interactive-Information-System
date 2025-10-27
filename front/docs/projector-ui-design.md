# Projector Guidance UI – 설계 문서

## 1. 목적
- 빔 프로젝터를 이용한 안내 시스템 프론트엔드(UI) 설계.
- WebSocket을 통해 백엔드에서 전송하는 상태·영상 보정 정보를 활용해 화면을 전환하고, 프로젝션 키스톤을 동적으로 조정한다.
- 음성 안내, 넛지 유도, 질의응답 등 복수 상태를 가진 키오스크 성격의 애플리케이션을 React 기반으로 구현한다.

## 2. 시스템 개요
- **디스플레이 환경**: 전용 프로젝터/디스플레이 장치. 브라우저 주소창, 기본 UI 없이 키오스크 모드로 실행.
- **프론트엔드 프레임워크**: React 19 + `react-router-dom`의 `MemoryRouter`.
- **실시간 통신**: WebSocket. 백엔드가 단일 스트림으로 상태 전환, 키스톤 매트릭스, 보조 컨텐츠 메시지를 push.
- **내부 상태 제어**: WebSocket 메시지 → 전역 상태(React state 또는 컨텍스트) → 라우터 전환 및 화면 렌더링.

## 3. 라우팅 구조
- Router: React Router `MemoryRouter`. 주소창/URL 상태에 의존하지 않는 프로젝터 UI 환경에 최적화.
- 경로 매핑
  - `/` : 랜딩/대기 화면 (기본 상태 및 연결 끊김 시 복귀).
  - `/nudge` : 넛지 유도 화면.
  - `/guidance` : 음성 안내 진행 화면.
  - `/qa` : 질의응답/추가 정보 화면.
  - 필요 시 `/error` 등 확장 가능.
- 상태 전환 로직
  - 서버에서 오는 `command.action`을 `src/config/appConfig.js`의 `actionRouteMap`으로 매핑.
  - `useWebSocketController` 훅이 `navigate()`를 호출해 라우트를 전환.
  - 모든 라우트 렌더링은 `MemoryRouter` 내부 컴포넌트로 처리 (URL 노출 없음).

## 4. WebSocket 메시지 규격 (초안)
```
공통 필드:
- type: "homography" | "command" | "overlay" | "ping" | ...
- timestamp: number (UTC epoch ms)
- sequence: number (선택) – 메시지 순서 추적
- meta: object (선택) – 확장용

type === "homography":
- matrix: number[3][3]
- validUntil: number (선택) – 적용 만료 시각

type === "command":
- action: string – 실행할 명령/상태 전환을 직접 표현 (예: `"start_nudge"`, `"start_guidance"`, `"enter_idle"`, `"show_overlay"`).
- context: object – 목적지, 음성 스크립트 ID 등 부가 정보
- priority (선택)

현재 정의된 주요 `action` (2025-10-24 기준)
- `start_landing` : 초기 랜딩 화면 진입.
- `start_nudge` : 넛지 화면 전환 요청. `context` 없음. **클라이언트가 OK(`status="completed"`) ACK을 보낼 때까지 1초 간격으로 재전송**. ACK 수신 후 homography 스트림만 유지.
- `start_guidance` : 음성 안내 화면으로 전환.
- `start_qa` : QA 모드 진입. `context.initialPrompt`(초기 안내 문구) 포함. **클라이언트 OK ACK 이전에는 1초 간격 재전송**.
- `start_listening` : QA 진행 중, 음성 수신 상태 진입(마이크 활성). `context.message`로 화면에 표기할 문구(예: "듣는 중...")를 전달할 수 있음. 수신 즉시 `ack`.
- `stop_listening` : 마이크 입력 종료. 파라미터 없음. 수신 즉시 `ack`.
- `start_thinking` : 서버가 응답 생성 중(사용자 입력 처리 완료). `context.message`에 화면에 표기할 상태 문구(예: "생각 중...")를 포함할 수 있음. 수신 즉시 `ack`.
- `start_speaking` : 서버가 음성 출력 중. `context.message`에 말풍선에 노출할 텍스트 포함. 수신 즉시 `ack`.
- `stop_speaking` : 말하기 단계 종료. 파라미터 없음. 수신 즉시 `ack`.
- `stop_all` : 모든 동작 중단, 랜딩 화면 복귀.

type === "overlay" (추후 확장):
- contentType: "map" | "image" | "text" | ...
- content: object – 지도 데이터, 텍스트, 이미지 경로 등

type === "ping" | "pong":
- 연결 유지용. 필요 시 latency 측정
```
- `type: "command"` 메시지를 수신한 클라이언트는 명령을 적용하기 시작할 때 `type: "ack"` (`status: "received"`)을 송신하고, 실제 화면/상태 전환이 완료되면 `status: "completed"`로 다시 응답해 백엔드가 종료 시점을 인지하도록 한다.  
- 백엔드는 `status: "completed"` ACK을 받을 때까지 동일한 `command` 메시지를 약 1초 간격으로 재전송해 누락 상황에 대비한다.
- 재연결 시 백엔드가 `type: "sync"`로 현재 상태를 주도적으로 전송, 프론트는 이를 초기 라우트로 반영.

## 5. 상태 관리 전략
- `useWebSocketController` 훅이 WebSocket 연결/재연결, 명령 재전송 대응, ACK 전송을 담당.
- WebSocket 연결 상태(`connecting` → `connected` → `disconnected`)와 현재 적용 중인 `command` 정보를 Context(`AppStateProvider`)로 공유.
- `disconnectGraceMs`(기본 10초) 동안 연결 복구를 시도한 뒤 실패하면 자동으로 `/` 상태로 복귀.
- Homography 행렬은 `latestHomography` 상태로 노출되고, 화면별로 변환 적용.
- 명령 ACK 정책
  - `start_nudge`, `start_qa`는 클라이언트 `ack(status="completed")` 수신 전까지 서버가 1초 간격으로 재전송.  
  - QA 상태 명령 (`start_listening`, `stop_listening`, `start_thinking`, `start_speaking`, `stop_speaking`)은 수신 즉시 `ack(status="completed")` 전송.

## 6. 화면 구성 개요
- 공통: `src/styles/ScreenLayout.module.css`의 레이아웃 클래스를 사용.
- 랜딩(`/`): 연결 상태 텍스트, 준비 메시지.
- 넛지(`/nudge`):
  - `NudgeScreen`이 `latestHomography.matrix` 값을 CSS `matrix3d`로 변환, 미리보기 박스에 적용.
  - `NudgeArrows` 컴포넌트가 화살표 모션을 렌더 (파라미터 기반 count 지원).
  - 정보 라벨 영역은 기본 “Information” 텍스트, 필요 시 추가 필드 확장.
- 안내(`/guidance`): 현재 단계/메시지 표시.
- 질의응답(`/qa`):
  - 화면 구성: ChatGPT Voice 스타일의 말풍선 애니메이션 + 단일 텍스트 메시지.
  - `start_qa` 수신 시 `initialPrompt` 문구를 표시하며, 서버는 동일 문구를 스피커로 재생.
  - 상태 명령(`start_listening`, `stop_listening`, `start_thinking`, `start_speaking`, `stop_speaking`)에 따라 말풍선 애니메이션이 부드럽게 전환된다.
  - `start_listening`과 `start_thinking`이 전달하는 `context.message`(예: "듣는 중...", "생각 중...") 및 `start_speaking.context.message`가 순서대로 화면 텍스트를 갱신하며, `stop_speaking` 이후에도 마지막 문구를 유지한다.

## 7. 실행 환경 고려 사항
- 브라우저는 키오스크 모드로 실행(Chrome `--kiosk` 등).  
- 앱 시작 시 자동으로 `/` 경로 → WebSocket 연결 시도.  
- 백엔드 가동 전에도 랜딩 화면이 안정적으로 표시되도록 타임아웃/재시도 로직 추가.  
- 로그 출력: 키오스크 환경에서는 콘솔 접근이 어려울 수 있으므로 원격 로깅(예: Sentry) 고려.
- 주요 프론트엔드 파라미터(WS 호스트/포트, 재연결 지연, 그레이스 타임 등)는 `src/config/appConfig.js`에서 중앙 관리.

## 8. 향후 TODO
- 안내/넛지 UI 디자인 시안 반영.
- `overlay` 메시지 포맷 확정 후 QA 화면 확장.
- ACK 누락/재전송 로깅 고도화.
- 실제 키스톤 렌더링(캔버스/웹GL) 도입 여부 검토.
- 자동 테스트: WebSocket 시뮬레이션 스크립트, 상태 전환 단위 테스트.

---
본 문서는 현재 구현 상태를 반영한 설계 자료이며, 백엔드 메시지 스펙 및 UI 디자인이 확정될 때 지속 갱신한다.

## 9. 구현 코드 구조 개요

### 최상위 구성
- `src/App.jsx`  
  - `MemoryRouter` 기반 앱 루트.  
  - `useWebSocketController` 훅을 초기화하고 전역 상태(Context)로 제공.  
  - 각 라우트(`/`, `/nudge`, `/guidance`, `/qa`)에 화면 컴포넌트 연결.
- `src/index.js`  
  - CRA 진입점. `App` 렌더링.

### 설정 및 유틸
- `src/config/appConfig.js`  
  - WebSocket 접속 정보(호스트/포트), 재연결/그레이스 타임, `actionRouteMap` 정의.  
  - `resolveWebSocketUrl`, `resolveRouteForAction` 함수 제공.
- `src/utils/homography.js`  
  - 3×3 homography 행렬을 CSS `matrix3d` 문자열로 변환 (`homographyToCssMatrix`).  
  - 기본 변환 상수(`IDENTITY_MATRIX3D`).

### 상태 관리
- `src/state/AppStateContext.jsx`  
  - React Context/Provider.  
  - `useAppState` 훅으로 화면 컴포넌트가 WebSocket 상태와 커맨드 정보를 구독.
- `src/hooks/useWebSocketController.jsx`  
  - WebSocket 연결/재연결/재전송 로직.  
  - 커맨드 수신 시 `ACK(received/completed)` 전송, `navigate` 콜백으로 라우트 전환.  
  - `latestHomography`, `connectionStatus`, `currentScreenCommand`, `qaState`(상태/표시 문구/마지막 명령 등)을 노출.

### 화면 컴포넌트
- 공통 스타일: `src/styles/ScreenLayout.module.css`.
- `src/components/screens/LandingScreen.jsx`  
  - 연결 상태 표시. 커맨드 없음.
- `src/components/screens/NudgeScreen.jsx`  
  - `latestHomography`를 CSS transform으로 반영.  
  - `NudgeArrows` 컴포넌트와 정보 라벨 구성.  
  - JSON 데이터 패널로 원본 메시지 확인.
  - 스타일: `src/components/screens/NudgeScreen.module.css`.
- `src/components/screens/GuidanceScreen.jsx`  
  - 단계/메시지 표시.
- `src/components/screens/QaScreen.jsx`  
  - 말풍선 애니메이션(`VoiceBubble`)과 `qaState.displayMessage` 텍스트만을 렌더링.  
  - 스타일: `QaScreen.module.css`.

### 넛지 전용 부품
- `src/components/nudge/NudgeArrows.jsx`  
  - `count` 프로퍼티로 화살표 개수 제어 (기본 3).  
  - 각 화살표는 인덱스 기반 `animation-delay`를 적용.  
  - 스타일: `NudgeArrows.module.css` (CSS 애니메이션 `nudgeChevronPulse` 정의).
- 넛지 화면 스타일(`NudgeScreen.module.css`)은 미리보기 프레임/라벨/데이터 영역을 담당.

### QA 전용 부품 (예정)
- `src/components/qa/VoiceBubble.jsx` (신규 예정)  
  - ChatGPT Voice 시안에 맞춘 말풍선/상태 UI.  
  - `status` 값(`listening`, `thinking`, `speaking` 등)에 따라 동일한 원 구성 요소의 크기·위치·애니메이션을 부드럽게 변경.  
  - 스타일: `VoiceBubble.module.css`로 분리, CSS 변수와 키프레임으로 연속적인 전환 구현.

### 기타
- `docs/projector-ui-design.md` (현재 문서)  
  - 요구사항, 메시지 스펙, 구조, 코드 개요 정리.
- 테스트/빌드  
  - CRA `npm run build`로 빌드 검증.  
  - 실제 환경 점검 시 `dummy_homography_generator/dumpy_server.py`를 실행해 WebSocket 시뮬레이션 가능 (`start_nudge` 명령은 context 없이 전송).

## 10. 조정 포인트 및 파라미터

- **WebSocket 설정 조정**:  
  - `appConfig.websocket.host/port/reconnectDelayMs/disconnectGraceMs`에서 한번에 관리.  
  - 프론트에서 포트를 변경해야 할 경우 해당 파일만 수정.

- **명령→화면 라우팅 규칙**:  
  - `appConfig.commands.actionRouteMap`에 키 추가/수정. 예) `"start_idle": "/"`.  
  - 새로운 화면 추가 시 이 맵과 `App.jsx` 라우트 정의를 함께 업데이트.

- **넛지 화살표 개수**:  
  - `NudgeArrows`는 `<NudgeArrows count={4} />`처럼 사용할 수 있음.  
  - `count` 값을 바꾸면 화살표가 자동으로 증가/감소하며 애니메이션도 같은 간격으로 적용.  
  - default 값 3은 넛지 화면에서 import 시 기본 적용.

- **넛지 라벨/메시지**:  
  - 현재는 “Information” 고정 텍스트. 향후 커맨드 `context`에 `infoTitle`, `infoHint` 등을 추가하면 `NudgeScreen`에서 조건부 렌더링으로 확장 가능.

- **Homography 적용**:  
  - `homographyToCssMatrix`가 3×3 행렬을 CSS `matrix3d`에 매핑.  
  - 셋째 행이 원근 투영을 제공하므로 서버에서 행렬을 바꾸면 프론트 미리보기도 즉시 반영.

- **ACK/명령 재전송**:  
  - `useWebSocketController`는 명령을 `ackStatusRef`로 추적하고, 서버가 동일 명령을 재전송하면 `received` ACK를 반복 송신.  
  - 명령이 완료되면 `completeCommand`가 `status: "completed"` ACK 전송 후 내부 상태 초기화.

- **리팩터링 고려**:  
  - UI 복잡도가 높아지면 상태 머신(XState 등) 도입 검토.  
  - 키스톤 실사 적용 시 Canvas/WebGL Layer를 별도 컴포넌트로 분리.

## 11. 백엔드 런타임 시나리오 (실서비스 기준)

1. **초기 연결**
   - 서버는 클라이언트 접속 직후부터 `homography` 메시지를 10Hz로 지속 송신 (ACK 불필요).
   - 초기 상태로 `start_landing` 전송 → 클라이언트 `ack(status="completed")` 수신 후 다음 단계 진행.

2. **넛지 유도**
   - `start_nudge` 전송 (`context` 없음). 클라이언트에서 OK 사인을 보내기 전까지 1초 간격으로 재전송.
   - 클라이언트가 `ack(status="completed")`를 보내면 넛지 화면이 준비된 것으로 간주하고, 이어서 homography 스트림만 지속 송신.

3. **QA 모드 진입 및 음성 지원**
   - 안내 방송 전 `start_qa` 전송 (`context.initialPrompt`: 스피커로 재생할 초기 가이드 문구).  
     → 클라이언트 OK ACK 수신 전까지 1초 간격으로 재전송.  
     → 클라이언트는 같은 문구를 화면에 표시.
   - 서버는 초기 안내 문구를 스피커로 재생. 재생 완료 시점에 `start_listening` 전송(마이크 활성화). → 즉시 ACK.
   - 사용자의 음성이 종료되면(음성 인식 타임아웃/종료 이벤트) `stop_listening` 전송 후 곧바로 `start_thinking` 전송 (예: `{ message: "생각 중..." }`). → 즉시 ACK.
   - 모델 답변이 준비되면 음성 합성 직전에 `start_speaking` 전송 (`context.message`: 화면에 띄울 답변 요약). → 즉시 ACK.
   - 음성 출력이 끝나면 `stop_speaking` 전송 → 즉시 ACK. 이후 다시 `start_listening`으로 루프.

4. **세션 종료**
   - 모든 안내를 종료하거나 오류 발생 시 `stop_all` 전송 → ACK 수신 후 `/` 상태 복귀 확인.

5. **일반 규칙**
   - 각 `command`에는 `commandId`와 `sequence`를 부여해 중복 수신 대비.
   - 클라이언트 ACK 포맷: `{"type": "ack", "action": "<원본 액션>", "status": "received" | "completed", "commandId": "...", "timestamp": ...}`.
   - 서버는 `status: "completed"` ACK을 받으면 해당 명령의 재전송을 중단.
   - Homography, 음성 상태 등 실시간 스트림 데이터는 ACK 없이 지속 송신.
   - 마이크 녹음, 음성 인식(ASR), 응답 생성, 음성 합성(TTS) 등 음성 관련 처리는 백엔드에서 수행하며, 단계 전환 시점에 맞춰 위 명령을 클라이언트로 전달.

---
위 내용을 바탕으로 개발을 재개할 경우, 설계 문서 및 코드 구조 개요를 먼저 확인한 후, 추가 요구사항에 따라 `appConfig` 설정, 명령 핸들링, UI 컴포넌트 확장을 진행한다. Codex와의 협업 시 본 문수정 사항을 참고해 동일한 기준으로 작업을 이어갈 수 있다.

---
본 문서는 현재 구현 상태와 최신 요구사항을 반영하며, 백엔드 메시지 스펙 및 UI 디자인이 확정될 때 지속 갱신 예정이다.
