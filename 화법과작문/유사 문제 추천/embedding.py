# 최종 json 파일에서 임베딩 모델 생성(문제만 임베딩 진행, 나머지는 metadata로 저장)

import os
import json
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.embeddings import SentenceTransformerEmbeddings

# STEP 1: JSON 데이터 로드
BASE_PATH = "/Users/chaewon/Desktop/STUBO/화법과작문/save_json_with_answers"
all_data = []

for filename in os.listdir(BASE_PATH):
    if filename.endswith("_with_answers.json"):
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
        "년도": item.get("년도"),
        "월": item.get("월"),
        "문제유형": item.get("문제유형"),
        "답": item.get("답"),
        "해설": item.get("해설"),
        "지문": item.get("지문")
    }
    documents.append(Document(page_content=content, metadata=metadata))

print("LangChain 문서 변환 완료")

# STEP 3: 임베딩 모델 로딩 (LangChain wrapper)
embedding_model = SentenceTransformerEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# STEP 4: FAISS 인덱스 생성 및 저장 (LangChain용)
vectorstore = FAISS.from_documents(documents, embedding_model)
vectorstore.save_local("outputs")

print("FAISS 인덱스 + 메타데이터 저장 완료 (LangChain 용)")