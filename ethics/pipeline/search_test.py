import pickle
import faiss
from sentence_transformers import SentenceTransformer

# 🔧 설정
INDEX_PATH = "/Users/chaewon/Desktop/STUBO/ethics/pipeline/vectorstore_KoSimCSE/index.faiss"
META_PATH = "/Users/chaewon/Desktop/STUBO/ethics/pipeline/vectorstore_KoSimCSE/questions_metadata.pkl"
MODEL_NAME = "jhgan/ko-sbert-nli" # 테스트용으로 다른 모델로도 변경 가능
TOP_K = 5

# 1. 모델 로딩
print(f"🔍 모델 로딩 중: {MODEL_NAME}")
model = SentenceTransformer(MODEL_NAME)

# 2. 벡터스토어 로딩
print("📦 FAISS 인덱스 로딩 중...")
index = faiss.read_index(INDEX_PATH)

# 3. 메타데이터 로딩
print("📑 메타데이터 로딩 중...")
with open(META_PATH, "rb") as f:
    metadata_list = pickle.load(f)

# 4. 사용자 쿼리 입력
query = input("\n❓ 검색할 쿼리를 입력하세요: ").strip()
query_embedding = model.encode([query])
D, I = index.search(query_embedding, k=TOP_K)

# 5. 결과 출력
print(f"\n🔎 Top-{TOP_K} 검색 결과:")
for rank, idx in enumerate(I[0], start=1):
    item = metadata_list[idx]
    print(f"\n--- [{rank}] --------------------------------")
    print(f"📘 제목: {item['explanation_title']}")
    print(f"❓ 문제: {item['question']}")
    print(f"📄 제시문 해설: {item['context_explanation'][:150]}...")
    print(f"✅ 정답: {item['answer']}")
    print(f"💬 해설: {item['answer_explanation'][:150]}...")

print("\n✅ 검색 완료!")
