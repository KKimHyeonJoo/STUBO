# 유사 문제 추천 모델 구성
# OCR로 입력 받은 이미지에서 텍스트 추출 -> 임베딩 모델로 임베딩 -> 유사 문제 검색
from PIL import Image
import easyocr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# 1. EasyOCR로 이미지에서 문제 텍스트 추출
def ocr_image_to_text(image_path):
    # EasyOCR 초기화 (한국어 + 영어)
    reader = easyocr.Reader(['ko', 'en'])
    
    # OCR 실행
    results = reader.readtext(image_path)
    
    # 텍스트 추출 및 정리
    extracted_text = ""
    for (bbox, text, confidence) in results:
        if confidence > 0.5:  # 신뢰도가 50% 이상인 텍스트만 사용
            extracted_text += text + " "
    
    print(f"\n[EasyOCR 추출 텍스트]\n{extracted_text.strip()}")
    print(f"총 {len(results)}개 텍스트 블록 발견")
    
    return extracted_text.strip()

# 2. 임베딩 모델 및 벡터스토어 로드
embedding_model = SentenceTransformerEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
vectorstore = FAISS.load_local("outputs", embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# 3. 이미지 파일 경로 입력
image_path = "/Users/chaewon/Desktop/STUBO/화법과 작문/테스트/유사 문제 추천 테스트/테스트 데이터/테스트_7.png"  # 실제 이미지 파일 경로로 변경

# 4. OCR → 유사 문제 검색
extracted_text = ocr_image_to_text(image_path)
similar_docs = retriever.get_relevant_documents(extracted_text)

print("\n=== 유사 문제 리스트 ===")
for i, doc in enumerate(similar_docs, 1):
    meta = doc.metadata
    print(f"{i}. {meta.get('년도', '연도 정보 없음')}년 {meta.get('월', '월 정보 없음')}월")
    print(doc.page_content)
    if meta.get("답"):
        print("정답:", meta["답"])
    if meta.get("해설"):
        print("해설:", meta["해설"])
    print("-" * 60)