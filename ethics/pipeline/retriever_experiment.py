import pickle, faiss, os
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from typing import Dict

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

# 설정
VECTOR_DIR = "./pipeline/vectorstore_kosbert"
INDEX_PATH = f"{VECTOR_DIR}/index.faiss"
META_PATH = f"{VECTOR_DIR}/questions_metadata.pkl"
MODEL_NAME = "jhgan/ko-sbert-nli"
TOP_K = 5
QUERIES = [
    "석가모니가 말한 고통의 원인은 무엇인가?",
    "공자와 맹자의 인간관은 어떻게 다른가?",
    "사회계약론의 주요 사상가 두 명을 설명하라"
]

# 모델 및 벡터 로딩
embedding = HuggingFaceEmbeddings(model_name=MODEL_NAME)
model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)
with open(META_PATH, "rb") as f:
    metadata = pickle.load(f)

# LangChain용 문서 구성
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

# FAISS용 docstore 및 매핑 설정
docstore = InMemoryDocstore({i: doc for i, doc in enumerate(docs)})
index_to_docstore_id = {i: i for i in range(len(docs))}

# Vectorstore 생성 (최신 langchain-compatible 방식)
vectorstore = FAISS(
    embedding_function=embedding,
    index=index,
    docstore=docstore,
    index_to_docstore_id=index_to_docstore_id,
)

# Retriever 2종 구성
retrievers = {
    "dense_k5": vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
    "bm25_k5": BM25Retriever.from_documents(docs)
}

# 결과 저장
os.makedirs("pipeline", exist_ok=True)
with open("pipeline/retriever_results.txt", "w", encoding="utf-8") as f:
    for name, retriever in retrievers.items():
        f.write(f"\n=== 🔍 Retriever: {name} ===\n")
        for query in QUERIES:
            f.write(f"\n[Query] {query}\n")
            results = retriever.get_relevant_documents(query)
            for i, doc in enumerate(results):
                content = doc.page_content[:150].replace('\n', ' ')
                f.write(f" ({i+1}) {content}...\n")
