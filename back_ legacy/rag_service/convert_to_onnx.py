from pathlib import Path
from optimum.onnxruntime import ORTModelForFeatureExtraction
from transformers import AutoTokenizer

def convert_model_to_onnx():
    """Hugging Face 모델을 ONNX 형식으로 변환하고 로컬에 저장."""
    
    model_name = "nlpai-lab/KURE-v1"
    onnx_path = Path(__file__).parent / "onnx_model"

    print(f"'{model_name}' 모델을 ONNX로 변환 시작...")
    
    model = ORTModelForFeatureExtraction.from_pretrained(model_name, export=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    onnx_path.mkdir(exist_ok=True)
    model.save_pretrained(onnx_path)
    tokenizer.save_pretrained(onnx_path)

    print(f"✅ 모델이 성공적으로 ONNX 형식으로 변환되어 '{onnx_path}'에 저장되었습니다.")
    print("이제 주 애플리케이션을 실행할 수 있습니다.")

if __name__ == "__main__":
    convert_model_to_onnx()