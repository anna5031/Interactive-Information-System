# 대학교 기계공학과 RAG QA 모듈

음성 인터페이스 QA 파이프라인에서 바로 사용할 수 있도록 정리한 기계공학과 전용 RAG 서비스입니다.  
FastAPI 서버는 제거했으며, `QAController`에서 직접 불러 사용할 수 있는 `RAGQAService` 와 개발용 CLI 실행기를 제공합니다.

## 1. 사전 준비

- Python 3.10 이상
- Groq API 키 → `GROQ_API_KEY` 환경 변수로 등록  
  (예: `export GROQ_API_KEY="sk_groq_xxx"`)
- Hugging Face 모델 다운로드 권한 (필요 시 토큰)

## 2. 의존성 설치

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install "optimum[onnxruntime]" transformers  # ONNX 변환용 (최초 1회)
```

> `torch`, `langchain`, `chromadb` 설치에는 시간이 걸릴 수 있습니다.

## 3. 임베딩 모델 & 벡터 DB 준비

1. ONNX 임베딩 모델 생성
   ```bash
   python convert_to_onnx.py
   ```
   - Hugging Face `nlpai-lab/KURE-v1` 모델을 내려 받아 `onnx_model/`에 저장합니다.
2. 최초 실행 시 `n7_professor_lectures_seminar.json` 데이터를 이용해 ChromaDB(`chroma_db/`)가 자동으로 채워집니다.  
   초기화를 원하면 `chroma_db/`를 삭제한 뒤 다시 실행하세요.

## 4. 개발용 CLI 실행

```bash
# 단일 질문 실행
python -m rag_service.app --question "김철수 교수님 연구실 위치 알려줘"

# 대화형 모드 + 처리 로그 출력
python -m rag_service.app --interactive --show-log

# 단계별 로그를 실시간으로 확인하고 Raw JSON까지 출력
python -m rag_service.app --question "열역학 수업 언제 있나요?" --stream-log --dump-json
```

옵션 설명:
- `--show-log`: LangGraph 처리 로그를 응답 후에 요약 형태로 보여줍니다.
- `--stream-log`: 처리 중간 로그를 실시간으로 출력합니다.
- `--show-documents`: 상위 3개의 참조 문장을 함께 표시합니다.
- `--dump-json`: 내부 딕셔너리 결과를 JSON으로 출력합니다.
- `--include-navigation-warmup`: 내비게이션 모듈까지 선행 로드합니다.

## 5. QAController 연동 가이드

```python
from rag_service import RAGQAService

service = RAGQAService()
await service.ensure_ready()
result = await service.query("김철수 교수님 연락처 알려줘")
print(result.answer)
```

`QAController`에서는 `RAGQAService` 인스턴스를 주입하거나, 생성자 기본값을 사용해 바로 연결할 수 있습니다.  
`emit_processing_log=True`로 호출하면 처리 로그가 콘솔에 출력되어 디버깅에 유용합니다.

## 6. 자주 발생하는 문제

- **`convert_to_onnx.py` 실행 없이 시작** → `onnx_model/model.onnx`를 찾지 못해 초기화가 실패합니다. 스크립트를 먼저 실행하세요.
- **`GROQ_API_KEY` 미설정** → Groq API 호출이 401 오류로 실패합니다. 환경 변수를 다시 확인하세요.
- **Hugging Face 모델 다운로드 실패** → 프록시/방화벽 또는 인증 문제일 수 있으니 네트워크 상태와 토큰을 확인하세요.

## 7. 변경 이력

주요 변경 사항은 `CHANGELOG.md`에 기록됩니다.

## 8. 테스트 안내

```bash
pytest rag_service/tests
```
