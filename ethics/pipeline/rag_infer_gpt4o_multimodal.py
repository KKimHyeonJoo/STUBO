import os
import pickle
import faiss
import base64
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
from openai import OpenAI
from typing import Dict
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key="OPENAPI_KEY")

# ✅ InMemoryDocstore 구현
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

# ✅ base64 인코딩 함수
def encode_image_to_base64(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_image}"

# ✅ 문서 로딩 및 FAISS 로드
MODEL_NAME = "jhgan/ko-sbert-nli"
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)

VECTOR_DIR = "./pipeline/vectorstore_kosbert"
INDEX_PATH = f"{VECTOR_DIR}/index.faiss"
META_PATH = f"{VECTOR_DIR}/questions_metadata.pkl"
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

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

docstore = InMemoryDocstore({i: doc for i, doc in enumerate(docs)})
index_to_docstore_id = {i: i for i in range(len(docs))}
vectorstore = FAISS(
    embedding_function=embedding,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

# ✅ 멀티모달 RAG 질의 함수
def multimodal_query_gpt4o(question: str, image_path: str):
    relevant_docs = retriever.invoke(question)
    context = "\n\n".join([f"문서 {i+1}:\n{doc.page_content}" for i, doc in enumerate(relevant_docs)])
    base64_image = encode_image_to_base64(image_path)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": f"다음은 수능 모의고사 문제입니다. 문서들과 이미지를 참고하여, '{question}'\n\n문서들:\n{context}"},
                {
                    "type": "image_url",
                    "image_url": {"url": base64_image},
                },
            ],
        }
    ]

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        temperature=0.2,
    )

    return response.choices[0].message.content.strip()

# 실행
if __name__ == "__main__":
    image_file_path = "./data/test_image/test6.png"  # 업로드한 이미지 파일
    question_text = "이 문제의 정답과 해설 알려줘."
    answer = multimodal_query_gpt4o(question_text, image_file_path)
    print("\n🧠 GPT-4o의 멀티모달 응답:\n")
    print(answer)
