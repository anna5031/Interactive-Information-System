# Exploration Pipeline Notes

## 1. 구현 개요
- YOLO11n pose 모델(`yolo11n-pose.pt`)을 사용하여 실시간 관절 추정.
- 관절 기반 anchor 및 몸통 축을 결합해 발 접점을 계산하고, EMA로 smoothing.
- CentroidTracker · AssistanceClassifier로 움직임/정지 판정 후, stationary → 도움 대상 → 쿨다운을 레거시와 동일하게 유지.
- 디버그 오버레이는 `SHOW_EXPLORATION_OVERLAY` 토글로 제어하며, 발/중심 포인트를 빨간 점으로 표시.
- Web UI 없이도 `tests/manual_exploration_stream.py`로 로컬 카메라 또는 동영상 파일을 테스트.

## 2. 주요 설정 파일
- `features/exploration/config.py`
  - `DetectionConfig`: `confidence_threshold`, `iou_threshold`, `keypoint_confidence_threshold`
  - `TrackingConfig`: 거리 임계치, 속도 smoothing 등
  - `AssistanceConfig`: 정지 시간, 쿨다운 시간
- `config/device_preferences.py`: 마이크/스피커 우선순위 및 `CAMERA_SOURCE`

## 3. 발 좌표 계산 전략 (혼합)
1. 발목이 양쪽 모두 보이면 y값이 낮은(바닥에 가까운) 발 + 이전 프레임 속도 비교로 체중이 실린 발 선택.
2. 발목 누락 시, hip→knee 벡터 기반으로 다리 방향 연장.
3. 그마저 없으면 바운딩 박스 하단을 fallback.
4. 몸통축(hip→어깨/코)을 아래로 내려 바닥 교차점과 0.7:0.3 비율로 혼합.
5. EMA smoothing + 최대 이동량 제한. 걷기 여부(`track.walking`) 저장.

## 4. Assistance 로직
- 정지 조건(`stationary_speed_threshold`, `stationary_duration_seconds`)을 충족하면 `_active_track_id`로 유지.
- 움직이거나 프레임에서 사라지면 `cooldown_seconds` 동안만 유지 후 해제.
- 도움 대상(`TRACK TARGET`)은 빨간 박스, 정지된 사람은 초록 박스, 나머지는 노란색 박스.

## 5. 로그 및 디버그 오버레이
- `track=… foot_point=(x, y) walking=Y/N` 형태로 발 좌표 로그 출력.
- `result.plot(boxes=False)` + 커스텀 박스로 상태 색상 구분.
- 센터 anchor/발 좌표는 굵은 빨간 점으로 화면에 표시.
- 텍스트 라벨은 ID만 유지(필요 시 `pipeline._draw_track_overlays`에서 제거 가능).

## 6. 수동 테스트 스크립트
```bash
python tests/manual_exploration_stream.py --source 0 --conf 0.3 --iou 0.6 --kp 0.2
```
- `SOURCE` / `MAX_FRAMES` 상수로 기본값 설정 가능.
- 동영상 파일 경로도 `--source` 인자로 전달 가능.

## 7. 레거시 대비 차이
- 구조 분리(`features/exploration/` 패키지) 및 설정 클래스 도입.
- 발 좌표 계산 방식과 오버레이는 레거시보다 명확히 분리돼 있으며, 향후 smoothing/보행 분석을 더 쉽게 확장 가능.
- Assistance 흐름은 레거시와 동일하지만, 상태 표시(빨강/초록/노랑)가 더 단순해졌습니다.

## 8. 향후 TODO
- 발 좌표 보정(발목과 발바닥 차이) 상수 튜닝
- walking 상태를 QA 흐름(예: 안내 대기)과 연동
- EMA 파라미터 및 detection/TrackingConfig 최적화
- 좌표 smoothing 결과를 세션 이벤트로 노출해 프론트엔드와 연동
