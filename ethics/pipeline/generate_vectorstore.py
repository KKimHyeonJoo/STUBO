import os
import json
import pickle
import faiss
from sentence_transformers import SentenceTransformer

# 🔧 설정
DATA_PATH = "/Users/chaewon/Desktop/STUBO/ethics/data/jsonl/parsing.jsonl"
INDEX_PATH = "/Users/chaewon/Desktop/STUBO/ethics/pipeline/index.faiss"
META_PATH = "/Users/chaewon/Desktop/STUBO/ethics/pipeline/questions_metadata.pkl"
MODEL_NAME = "jhgan/ko-sbert-nli"

# 1. 데이터 불러오기
with open(DATA_PATH, "r", encoding="utf-8") as f:
    raw_data = [json.loads(line) for line in f]

# 2. 문항별 임베딩 텍스트 구성
texts_to_embed = []
metadata_list = []

for item in raw_data:
    title = item["explanation_title"].strip()
    question = item["question"].strip()

    if item["context"].startswith("[IMAGE:"):
        # 🖼️ 이미지 문항
        context_part = item["context_explanation"].strip()
    else:
        # 📄 텍스트 문항
        context_part = item["context"].strip()

    embedding_text = f"{title}\n{context_part}\n{question}"
    texts_to_embed.append(embedding_text)
    metadata_list.append(item)

print(f"✅ 총 {len(texts_to_embed)}개 문항 임베딩 준비 완료!")

# 3. 모델 로딩 & 임베딩
print(f"🔍 모델 로딩 중: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)
embeddings = model.encode(texts_to_embed, convert_to_numpy=True)
dimension = embeddings.shape[1]

# 4. FAISS 인덱스 저장
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)
faiss.write_index(index, INDEX_PATH)
print(f"📦 FAISS 인덱스 저장 완료 → {INDEX_PATH}")

# 5. 메타데이터 저장
with open(META_PATH, "wb") as f:
    pickle.dump(metadata_list, f)
print(f"📝 메타데이터 저장 완료 → {META_PATH}")
