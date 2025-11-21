# Back RAG System

`back/rag_system`은 음성 인터페이스 중심 QA 서비스를 위해 설계된 신규 RAG 파이프라인입니다. 텍스트 문서·실내 지도 JSON·LangGraph 워크플로우를 활용하며, 모든 핵심 모델은 Hugging Face에서 받아 로컬 ONNX 형식으로 실행합니다.

## 구성 요소

- `data/source_documents/`: `.txt` 문서를 보관하는 디렉터리. 인덱스 구축 시 청크로 분할되어 임베딩 파일로 저장됩니다.
- `data/indoor_map.json`: 경로 안내 보조용 실내 지도 메타데이터(샘플).
- `text_ingest.py`: 텍스트 문서를 읽고 청크를 생성합니다.
- `convert_to_onnx.py`: Hugging Face 모델을 받아 `nlpai-lab/KURE-v1` 임베딩과 `Dongjin-kr/ko-reranker` 재정렬 모델을 ONNX로 변환합니다.
- `embeddings.py`: 변환된 KURE 임베딩 모델을 로딩합니다.
- `vector_index.py`: 단일 `.npz` 파일과 JSON 메타데이터로 구성된 경량 벡터 인덱스.
- `retriever.py`: 정보 검색 전용 벡터 파이프라인 + 추천 검색 전용 BM25/벡터 RRF + Ko-Reranker 재정렬을 담당합니다.
- `guardrails.py`, `answer_generator.py`: Groq `openai/gpt-oss-120b` 모델을 Structured Output으로 활용하는 1차/2차 LLM.
- `graph.py`: LangGraph 기반 파이프라인. 컨텍스트 메모리, 가드레일, 검색, 답변 생성, 내비게이션 트리거, 음성 재질문 경로를 포함합니다.
- `service.py`: 질문 스트림을 처리하는 상위 서비스. 7초 타임아웃 및 세션 초기화 로직을 포함합니다.
- `cli.py`: 콘솔에서 인덱스 구축 및 QA 테스트를 수행하는 실행 진입점.

## 빠른 시작

```bash
# 1. 환경 준비
python -m venv .venv && source .venv/bin/activate
pip install -r back/rag_system/requirements.txt

# 2. Hugging Face 모델을 로컬 ONNX로 변환
#    사전 준비:
#      1) Hugging Face 계정을 만들고 Access Token 발급
#      2) 터미널에서 `huggingface-cli login` 실행 후 토큰 입력 (또는 HUGGINGFACE_HUB_TOKEN 환경 변수 설정)
#    변환 대상:
#      - nlpai-lab/KURE-v1 (임베딩 모델)
#      - Dongjin-kr/ko-reranker (Cross-Encoder 재정렬 모델)
#    아래 스크립트는 위 두 모델을 자동으로 다운로드→ONNX 변환→`back/rag_system/onnx_model`, `back/rag_system/onnx_reranker`에 저장합니다.
python back/rag_system/convert_to_onnx.py

# 3. 텍스트 인덱스 재구축
python -m back.rag_system.cli --build-index

# 4. QA 세션 실행 (콘솔 입력 기반)
python -m back.rag_system.cli --interactive
python -m back.rag_system.cli --question "김철수 교수님 연구실 위치 알려줘"
```
