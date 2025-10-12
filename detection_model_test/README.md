# 인터랙티브 객체 탐지 비교 도구

Streamlit 기반의 웹 애플리케이션으로, **YOLO**와 **RF-DETR** 모델을 한 화면에서 비교하며 객체 탐지를 수행할 수 있습니다. 로컬에 저장된 테스트 이미지 혹은 업로드한 이미지를 대상으로 모델별 추론 결과와 시각화를 제공합니다.

- 좌측 사이드바에서 모델 종류(YOLO / RF-DETR), 체크포인트, confidence, IoU, 사용 장치를 손쉽게 변경
- 원본 이미지와 탐지 결과 이미지를 나란히 확인
- 필터링된 탐지 결과를 표 형태로 검토

---

## 주요 폴더 구조

```
├── app.py                # Streamlit 진입점
├── config.py             # 경로/슬라이더 기본값 등 공통 설정
├── core/
│   └── model_registry.py # model 폴더 내 체크포인트 자동 탐색
├── inference/            # 백엔드별 추론 래퍼
│   ├── base.py
│   ├── yolo_backend.py
│   └── rfdetr_backend.py
├── ui/                   # Streamlit UI 구성 요소
├── utils/                # 이미지 로딩 및 시각화 유틸
├── test_image/           # 기본 제공 테스트 이미지
└── requirements.txt      # 의존성 목록
```

> 중요: `.gitignore`에 의해 `model/` 폴더는 Git에 포함되지 않습니다. 로컬에서 직접 생성하고 모델 가중치를 배치해야 합니다.

---

## 환경 준비

1. **Python 환경**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Windows는 .venv\Scripts\activate
   ```
2. **필수 패키지 설치**
   ```bash
   pip install -r requirements.txt
   ```
   - GPU 사용 시 `torch`, `torchvision`이 환경에 맞게 설치되어 있어야 합니다.
   - RF-DETR 백엔드는 `rfdetr`, YOLO 백엔드는 `ultralytics` 패키지를 사용합니다.

---

## 모델 가중치 배치

앱은 `model/` 디렉터리 아래에서 체크포인트를 자동으로 검색합니다.

```
model/
├── yolo/      # *.pt (YOLO 가중치)
│   ├── roboflow.pt
│   └── …
└── rf-detr/   # *.pth (RF-DETR 가중치)
    ├── cubi/checkpoint_best_ema.pth
    └── …
```

- 필요 폴더가 없다면 직접 생성한 뒤 가중치 파일을 복사하세요.
- 새 모델을 추가하면 앱 재실행 없이도 사이드바 모델 목록이 갱신됩니다.

---

## 실행 방법

```bash
streamlit run app.py
```

앱이 실행되면 브라우저에서 다음을 수행할 수 있습니다.

- `test_image/` 폴더의 샘플 이미지를 자동으로 로딩
- 사이드바에서 모델/장치/threshold 조정
- 필요 시 이미지 파일 업로드 후 해당 이미지로만 추론
- 탐지 결과 표에서 confidence와 bounding box 좌표 확인

---

## 커스터마이징 팁

- `config.py`에서 confidence, IoU, 경로 등의 기본값 조정 가능
- `test_image/` 폴더에 샘플 이미지를 추가하면 자동 인식
- `utils/visualization.py`를 수정해 박스 스타일이나 라벨 표현 방식을 바꿀 수 있음

---

## 문제 해결

- **모델 목록이 비어있는 경우**: `model/` 아래에 적절한 확장자의 가중치 파일이 있는지 확인하세요.
- **GPU가 감지되지 않는 경우**: 사이드바의 장치를 `auto` 대신 원하는 장치(`cuda`, `mps`, `cpu`)로 직접 선택하세요.
- **패키지 오류**: `pip install -r requirements.txt`가 정상적으로 완료되었는지 재확인하거나, CUDA 환경에 맞는 PyTorch를 별도로 설치합니다.

