import pickle
import faiss
import os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from openai import OpenAI  # 최신 방식
from typing import Dict

# 최신 openai 클라이언트 초기화
client = OpenAI(api_key="OPENAPI_KEY")
class InMemoryDocstore:
    def __init__(self, _dict: Dict[int, Document]):
        self._dict = _dict

    def search(self, search: int) -> Document:
        return self._dict[search]

    def add(self, texts: Dict[int, Document]) -> None:
        self._dict.update(texts)

    def delete(self, ids: list) -> None:
        for i in ids:
            self._dict.pop(i, None)

# Hugging Face embedding 모델 설정
MODEL_NAME = "jhgan/ko-sbert-nli"
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

# FAISS vectorstore 로딩
VECTOR_DIR = "./pipeline/vectorstore_kosbert"
INDEX_PATH = f"{VECTOR_DIR}/index.faiss"
META_PATH = f"{VECTOR_DIR}/questions_metadata.pkl"
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# 문서 리스트 생성
docs = []
for m in metadata:
    title = m["explanation_title"].strip()
    question = m["question"].strip()
    if m["context"].startswith("[IMAGE:"):
        context_part = m["context_explanation"].strip()
    else:
        context_part = m["context"].strip()
    full_text = f"{title}\n{context_part}\n{question}"
    docs.append(Document(page_content=full_text, metadata=m))

# dense_k5 retriever 구성
docstore = InMemoryDocstore({i: doc for i, doc in enumerate(docs)})
index_to_docstore_id = {i: i for i in range(len(docs))}
vectorstore = FAISS(
    embedding_function=embedding,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# GPT-4o 호출 함수
def query_gpt4o(question: str):
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([f"문서 {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    prompt = (
        "아래 문서들을 참고하여 질문에 대한 정확하고 간결한 답변을 생성해줘.\n\n"
        f"{context}\n\n질문: {question}\n답변:"
    )

    # 최신 방식으로 GPT-4o 호출
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "너는 한국어로 답변하는 윤리 선생님이야."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return response.choices[0].message.content.strip()

# 예시 실행
if __name__ == "__main__":
    q = input("질문을 입력하세요: ").strip()
    answer = query_gpt4o(q)
    print("\n🧠 GPT-4o의 답변:\n")
    print(answer)
