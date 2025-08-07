import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

# 🔐 환경변수 불러오기
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY")

# ✅ 벡터스토어 경로
faiss_dir = Path("subject_literature/faiss_index_examine")

# ✅ embedding 모델 로딩
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

# ✅ 벡터스토어 로딩
vectorstore = FAISS.load_local(str(faiss_dir), embeddings, allow_dangerous_deserialization=True)

# ✅ 내부 docstore 접근
docstore = vectorstore.docstore._dict

print(f"[INFO] 문서 수: {len(docstore)}")

# ✅ 문서 내용 출력
for i, (doc_id, doc) in enumerate(docstore.items()):
    print("문서 내용:", doc.page_content)
    print("메타데이터:", doc.metadata)