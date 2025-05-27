import os
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import pickle

# 1. 데이터 불러오기
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_PATH = os.path.join(ROOT_DIR, "data", "questions_clean.json")

# 1. 데이터 불러오기
with open(DATA_PATH, "r", encoding="utf-8") as f:
    questions = json.load(f)

# 2. 임베딩 모델 로드 (한국어 전용 SBERT 추천)
model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 3. 질문 문장 임베딩
corpus = [q["문제"] for q in questions]
corpus_embeddings = model.encode(corpus, convert_to_numpy=True)

# 4. FAISS 벡터 인덱스 저장
dimension = corpus_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(corpus_embeddings)

# 5. 인덱스와 원본 데이터 저장
faiss.write_index(index, "index.faiss")
with open("questions.pkl", "wb") as f:
    pickle.dump(questions, f)

print("✅ 벡터 인덱스 저장 완료!")
