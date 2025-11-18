# Back RAG System

`back/rag_system`은 음성 인터페이스 중심 QA 서비스를 위해 설계된 신규 RAG 파이프라인입니다. 텍스트 문서·실내 지도 JSON·LangGraph 워크플로우를 활용합니다.

## 구성 요소

- `data/source_documents/`: `.txt` 문서를 보관하는 디렉터리. 인덱스 구축 시 청크로 분할되어 임베딩 파일로 저장됩니다.
- `data/indoor_map.json`: 경로 안내 보조용 실내 지도 메타데이터(샘플).
- `text_ingest.py`: 텍스트 문서를 읽고 청크를 생성합니다.
- `embeddings.py`: 기존 Groq RAG에서 사용하던 ONNX 임베딩 모델(`nlpai-lab/KURE-v1`)을 로딩합니다.
- `vector_index.py`: 단일 `.npz` 파일과 JSON 메타데이터로 구성된 경량 벡터 인덱스.
- `retriever.py`: BM25 + 코사인 유사도 앙상블 검색, 상대적 임계값 필터링을 지원합니다.
- `guardrails.py`, `answer_generator.py`: Groq `openai/gpt-oss-120b` 모델을 Structured Output으로 활용하는 1차/2차 LLM.
- `graph.py`: LangGraph 기반 파이프라인. 컨텍스트 메모리, 가드레일, 검색, 답변 생성, 내비게이션 트리거, 음성 재질문 경로를 포함합니다.
- `service.py`: 질문 스트림을 처리하는 상위 서비스. 7초 타임아웃 및 세션 초기화 로직을 포함합니다.
- `cli.py`: 콘솔에서 인덱스 구축 및 QA 테스트를 수행하는 실행 진입점.

## 빠른 시작

<!-- ```bash
# 환경 준비
python -m venv .venv && source .venv/bin/activate
pip install -r back/rag_system/requirements.txt -->

# ONNX 임베딩 모델 준비

python back/rag_system/convert_to_onnx.py

# 텍스트 인덱스 재구축

python -m back.rag_system.cli --build-index

# QA 세션 실행 (콘솔 입력 기반)

python -m back.rag_system.cli --interactive
python -m back.rag_system.cli --question "김철수 교수님 연구실 위치 알려줘"

```

> 음성 인식 모델과 실제 네비게이션 API는 아직 미구현 상태이며, `speech_interface.py` / `navigation.py`에서 더미 동작으로 대체됩니다.
```
