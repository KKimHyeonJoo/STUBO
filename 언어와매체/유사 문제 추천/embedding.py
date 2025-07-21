import os
import json
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings

# STEP 1: JSON 데이터 로드
BASE_PATH = "C:/Users/user/STUBO/언어와매체/save_json_tagged"
all_data = []

for filename in os.listdir(BASE_PATH):
    if filename.endswith(".json"):
        with open(os.path.join(BASE_PATH, filename), "r", encoding="utf-8") as f:
            data = json.load(f)

            all_data.extend(data)

print(f"총 {len(all_data)}개 문항 로드 완료")

# STEP 2: LangChain 문서 변환
# 문제만 임베딩하고, 나머지 정보는 metadata로 저장
documents = []
for item in all_data:
    content = item['문제']
    metadata = {
        "문제유형": item.get("type"),
        "중세국어유무" : item.get("med", False),
        "태그": item.get("tags", []),
        "년": item.get("년"),
        "월": item.get("월"),
    }
    documents.append(Document(page_content=content, metadata=metadata))

print("LangChain 문서 변환 완료")

# STEP 3: 임베딩 모델 로딩 (LangChain wrapper)
embedding_model = SentenceTransformerEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# STEP 4: FAISS 인덱스 생성 및 저장 (LangChain용)
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("outputs")

print("FAISS 인덱스 + 메타데이터 저장 완료 (LangChain 용)")