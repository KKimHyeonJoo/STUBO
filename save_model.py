# save_model.py
from sentence_transformers import SentenceTransformer
import os

def download_and_save(model_name: str, save_subdir: str):
    save_path = os.path.join("models", save_subdir)

    # ✅ 이미 저장돼 있다면 건너뛰기
    if os.path.exists(save_path) and os.listdir(save_path):
        print(f"✅ 이미 존재함, 건너뜀: {save_path}")
        return

    os.makedirs(save_path, exist_ok=True)
    
    print(f"📦 {model_name} 다운로드 중...")
    model = SentenceTransformer(model_name)
    model.save(save_path)
    print(f"✅ 저장 완료: {save_path}\n")

# 🔽 저장할 모델 목록
models_to_download = {
    "jhgan/ko-sroberta-multitask": "jhgan-ko-sroberta-multitask",
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS": "snunlp-kr-sbert-v40k",
    "BM-K/KoSimCSE-roberta-multitask": "kosimcse-roberta-multitask"
}

if __name__ == "__main__":
    for model_name, subdir in models_to_download.items():
        download_and_save(model_name, subdir)