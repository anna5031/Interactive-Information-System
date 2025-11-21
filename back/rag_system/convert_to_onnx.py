from pathlib import Path

from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForSequenceClassification
from transformers import AutoTokenizer

EMBEDDING_MODEL = "nlpai-lab/KURE-v1"
EMBEDDING_DIR = Path(__file__).parent / "onnx_model"
RERANKER_MODEL = "Dongjin-kr/ko-reranker"
RERANKER_DIR = Path(__file__).parent / "onnx_reranker"


def convert_embedding_model() -> None:
    """기존 임베딩 모델을 ONNX로 변환."""
    print(f"'{EMBEDDING_MODEL}' 임베딩 모델을 ONNX로 변환합니다...")
    model = ORTModelForFeatureExtraction.from_pretrained(EMBEDDING_MODEL, export=True)
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL)
    EMBEDDING_DIR.mkdir(exist_ok=True)
    model.save_pretrained(EMBEDDING_DIR)
    tokenizer.save_pretrained(EMBEDDING_DIR)
    print(f"✅ 임베딩 모델이 '{EMBEDDING_DIR}'에 저장되었습니다.")


def convert_reranker_model() -> None:
    """Ko-Reranker 모델을 ONNX로 변환."""
    print(f"'{RERANKER_MODEL}' Ko-Reranker 모델을 ONNX로 변환합니다...")
    model = ORTModelForSequenceClassification.from_pretrained(RERANKER_MODEL, export=True)
    tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    RERANKER_DIR.mkdir(exist_ok=True)
    model.save_pretrained(RERANKER_DIR)
    tokenizer.save_pretrained(RERANKER_DIR)
    print(f"✅ Ko-Reranker 모델이 '{RERANKER_DIR}'에 저장되었습니다.")


def convert_all_models() -> None:
    convert_embedding_model()
    convert_reranker_model()
    print("모든 ONNX 변환이 완료되었습니다. 애플리케이션 실행 준비가 되었습니다.")


if __name__ == "__main__":
    convert_all_models()
