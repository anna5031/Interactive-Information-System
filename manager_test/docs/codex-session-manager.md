# Codex 새 세션 React 부트스트랩 가이드

## 1. 목적
- 관리자 React 앱(도면 관리 단계 1/2)을 다시 열 때 필수 컨텍스트를 빠르게 공유하기 위한 요약 문서.
- 현재 프런트 구조, 실행 방법, 저장 데이터 포맷, 2단계 설계 기준을 한 장에 정리.
- 새로운 Codex 세션 시작 시 본 문서를 함께 전달하면 컨텍스트 회복이 쉬워진다.

## 2. 현재 프런트 요약
- 프로젝트 루트: `/Users/jaehyeon/Developer/Interactive-Information-System/manager_test`
- 런타임: Node 20.x + npm (CRA 기반)
- 주요 스크립트
  ```bash
  npm start       # 개발 서버 (http://localhost:3000)
  npm test        # CRA 테스트 러너 (watch)
  npm run build   # 프로덕션 빌드
  ```
- 라우팅 구조 (`src/router/AppRouter.jsx`)
  - `/admin/upload` : 업로드 & 저장된 도면 목록 (Step 1 진입)
  - `/admin/editor` : YOLO 결과 검수/수정 (Step 1 편집)
  - `/admin/review` : Step 1 결과 확인 (JSON 미리보기/복사)
  - `/admin/step-two/:stepOneId` : 그래프 메타데이터 입력 (Step 2)
- 인증 컨텍스트: `useAuth` (기존 로그인 흐름 유지)
- 글로벌 상태: `FloorPlanProvider` (`src/utils/floorPlanContext.jsx`)

## 3. 1단계(업로드·검수) 설계 기준
- `uploadFloorPlan(file)` (`src/api/floorPlans.jsx`)
  - 더미 `yolo.txt`, `wall.txt`를 fetch → 박스/라인/포인트 생성.
  - 업로드 이미지 Base64 URL로 메모리에 저장.
- `saveAnnotations({...})`
  - 박스/라인/포인트를 직렬화 후 `src/api/stepOneResults.js` 통해 localStorage에 JSON 저장.
  - 저장 레코드 스키마
    ```json
    {
      "id": "step_one_<uuid>",
      "fileName": "step_one_<uuid>.json",
      "createdAt": "ISO8601",
      "yolo": {"raw": string, "text": string, "boxes": []},
      "wall": {"raw": string, "text": string, "lines": []},
      "door": {"raw": string, "text": string, "points": []},
      "metadata": {"fileName": string, "imageUrl": string|null}
    }
    ```
- Review 페이지는 JSON 미리보기/복사만 제공 (다운로드는 메인 카드에서 수행).
- 업로드 메인 (`AdminUploadPage`)
  - localStorage 에서 Step 1 결과 목록을 가져와 카드 그리드로 표시 (`StepOneResultCard`).
- 카드: 미리보기 이미지, 저장 시각, 항목 카운트, “1단계 수정”, “2단계 진행”, “JSON 다운로드”.
- “1단계 수정” 선택 시 해당 저장본을 다시 편집 단계로 로드하며, 재저장하면 기존 레코드를 덮어쓴다.

## 4. 2단계(그래프 메타데이터) 설계 기준
- 진입: `/admin/step-two/:stepOneId`
  - localStorage에서 선택한 Step 1 결과 로드.
  - Canvas (`StepTwoCanvas`)에 Step 1에서 확정된 박스/라인/포인트를 그대로 표시 (읽기 전용).
  - 캔버스/폼 상호 선택 동기화: 우측 리스트 or 좌측 도면 클릭 시 `selectedEntity` 업데이트.
- 기본 정보 입력(Stage: `base`)
  - Room: 방 이름/호수 입력.
  - Door: 문 종류(미닫이/여닫이/기타) 및 기타시 텍스트 입력.
- 상세 정보(Stage: `details`)
  - 기본 정보 하단에서 추가 Key-Value 필드를 동적으로 관리 (`KeyValueEditor`).
  - 카드 헤더는 사용자가 입력한 방 이름/호수(또는 문 종류)를 표시.
- 저장(Stage: `review`)
  - `saveStepTwoResult(payload)` (`src/api/stepTwoResults.js`) → localStorage 저장 + JSON 다운로드/복사 제공.
  - 저장 스키마
    ```json
    {
      "id": "step_two_<uuid>",
      "sourceFloorplan": "step_one_<uuid>",
      "rooms": [ { nodeId, base: {name, number, displayLabel}, meta: [], geometry } ],
      "doors": [ { nodeId, base: {type, customType?}, meta: [], geometry } ],
      "preview": { "imageUrl": string|null },
      "createdAt": "ISO8601"
    }
    ```

## 5. 상태 & 저장 계층
- `src/utils/floorPlanContext.jsx`
  - 세션 상태(localStorage 동기화): 업로드 진행 단계, 현재 편집 데이터, `stepOneResult`.
  - 레거시 필드(`savedYoloText` 등)는 초기 로드시 자동 마이그레이션.
- `src/api/stepOneResults.js`, `src/api/stepTwoResults.js`
  - 단순 localStorage CRUD. 서버 연동 시 이 레이어를 API 호출로 대체 예정.

## 6. 실행/디버깅 팁
- CRA Hot Reload 사용. `npm start` 상태에서 파일 수정 → 즉시 반영.
- ESLint 경고 대부분은 CRA 기본 규칙 (no-unused-vars 등). 저장 전 불필요 변수 제거.
- Canvas 좌표는 0~1 정규화 값 기반. Step 2에서 미니뷰를 더 고도화하려면 `StepTwoCanvas` 참고.
- 더미 텍스트 파일 수정 시 `public` fetch 경로 확인 필요 (현재 `src/dummy` 번들 사용).

## 7. 향후 작업 TODO 참고
- 서버 연동 시: `saveStepOneResult`, `saveStepTwoResult`를 REST 호출로 교체.
- Step 2 Canvas 확대/검색 UX 개선.

## 8. 코드 구성 요약
- **src/router/AppRouter.jsx**: 인증 여부에 따라 관리자 라우팅 구성.
- **src/utils/floorPlanContext.jsx**: 업로드/검수 워크플로우 전역 상태 + localStorage 연동.
- **src/api/floorPlans.jsx**: 더미 YOLO/wall parsing, Step 1 저장 로직.
- **src/api/stepOneResults.js / stepTwoResults.js**: 로컬 저장 레이어.
- **src/pages/AdminUploadPage.jsx**: 업로드 카드 그리드 & Step 2 진입.
- **src/components/stepOne/StepOneResultCard.jsx**: 저장된 도면 카드 UI.
- **src/pages/AdminEditorPage.jsx / FloorPlanEditorPage.jsx**: 1단계 편집 플로우.
- **src/pages/AnnotationReviewPage.jsx**: Step 1 JSON 미리보기.
- **src/pages/AdminStepTwoPage.jsx**: 2단계 전체 플로우(Stage 관리, 폼, 저장).
- **src/components/stepTwo/StepTwoCanvas.jsx**: Step 1 박스/포인트를 읽기 전용 Canvas로 표시.
- **src/components/annotations/**: AnnotationCanvas 및 상호작용 훅 (Step 1 편집/Step 2 프리뷰 공용).

---
필요 시 본 문서를 업데이트하고, 변경 사실을 커밋 메시지 또는 추가 설계 문서에 함께 남긴다.
