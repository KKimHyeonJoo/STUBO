# 유사 문제 추천 모델 구성 (문제 유형별 필터링 + 유사도 검색)
# 이미지 → OCR → GPT 유형 분류 → 유형별 필터링 → 유사도 검색
import openai
import easyocr
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
import os

# OpenAI API 설정
client = openai.OpenAI(api_key="  ")

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

# 2. GPT로 문제 유형 분류
def classify_question_type(question_text):
    prompt = f"""
수능 국어 화법과 작문 문제는 다음의 5가지 유형 중 하나로 분류됩니다:

1. 의사소통 상황 이해형
2. 담화/작문 방식 분석형
3. 자료 활용·적용형
4. 협력적 의사소통형
5. 수정·보완 판단형

다음 문제의 유형을 위 기준에 따라 판단하세요.
출력은 반드시 숫자 하나만! (예: 2)

문제 텍스트:
{question_text}
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": prompt}
            ],
            temperature=0.0,
            max_tokens=10
        )
        
        question_type = response.choices[0].message.content.strip()
        print(f"\n[GPT 유형 분류 결과] 유형: {question_type}")
        return question_type
    except Exception as e:
        print(f"유형 분류 실패: {e}")
        return None

# 3. 유형별 필터링 + 유사도 검색
def search_similar_by_type(question_text, question_type, vectorstore, k=5):
    print(f"\n=== 유형 {question_type} 필터링 + 유사도 검색 ===")
    
    # 숫자 유형을 실제 JSON 텍스트 유형으로 매핑
    type_mapping = {
        "1": "의사소통 상황 이해형",
        "2": "담화/작문 방식 분석형", 
        "3": "자료 활용·적용형",
        "4": "협력적 의사소통형",
        "5": "수정·보완 판단형"
    }
    
    # 유형별 필터링 조건
    type_filters = {
        "1": {"문제유형": "의사소통 상황 이해형"},
        "2": {"문제유형": "담화/작문 방식 분석형"},
        "3": {"문제유형": "자료 활용·적용형"},
        "4": {"문제유형": "협력적 의사소통형"},
        "5": {"문제유형": "수정·보완 판단형"}
    }
    
    filter_condition = type_filters.get(question_type, {})
    
    try:
        # 유형별 필터링된 검색
        if filter_condition:
            docs = vectorstore.similarity_search(
                question_text, 
                k=k,
                filter=filter_condition
            )
            mapped_type = type_mapping.get(question_type, question_type)
            print(f"유형 '{mapped_type}' 필터링 결과: {len(docs)}개")
        else:
            # 필터링 없이 전체 검색
            docs = vectorstore.similarity_search(question_text, k=k)
            print(f"전체 검색 결과: {len(docs)}개")
        
        return docs
    except Exception as e:
        print(f"유형별 검색 실패: {e}")
        # 필터링 실패 시 전체 검색으로 fallback
        return vectorstore.similarity_search(question_text, k=k)

# 4. 메인 실행 코드
def main():
    # 임베딩 모델 및 벡터스토어 로드
    print("임베딩 모델 로딩 중...")
    embedding_model = SentenceTransformerEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
    
    print("벡터스토어 로딩 중...")
    try:
        vectorstore = FAISS.load_local("outputs", embedding_model, allow_dangerous_deserialization=True)
        print("✅ 벡터스토어 로드 성공")
    except Exception as e:
        print(f"❌ 벡터스토어 로드 실패: {e}")
        return
    
    # 이미지 파일 경로
    image_path = "/Users/chaewon/Desktop/STUBO/화법과 작문/테스트/유사 문제 추천 테스트/테스트 데이터/테스트_14.png"
    
    print(f"\n{'='*60}")
    print(f"테스트 이미지: {os.path.basename(image_path)}")
    print(f"{'='*60}")
    
    # 1. OCR로 텍스트 추출
    extracted_text = ocr_image_to_text(image_path)
    
    if not extracted_text.strip():
        print("❌ OCR 결과가 비어있습니다.")
        return
    
    # 2. GPT로 문제 유형 분류
    question_type = classify_question_type(extracted_text)
    
    if not question_type:
        print("❌ 유형 분류 실패. 전체 검색으로 진행합니다.")
        question_type = None
    
    # 3. 유형별 필터링 + 유사도 검색
    similar_docs = search_similar_by_type(extracted_text, question_type, vectorstore, k=3)
    
    # 4. 결과 출력
    print(f"\n=== 유사 문제 추천 결과 (상위 {len(similar_docs)}개) ===")
    for i, doc in enumerate(similar_docs, 1):
        meta = doc.metadata
        print(f"\n{i}. {meta.get('년도', '연도 정보 없음')}년 {meta.get('월', '월 정보 없음')}월")
        print(f"유형: {meta.get('문제유형', '유형 정보 없음')}")
        print(f"{doc.page_content}...")
        if meta.get("답"):
            print(f"정답: {meta['답']}")
        if meta.get("해설"):
            print(f"해설: {meta['해설']}...")
        print("-" * 60)

if __name__ == "__main__":
    main()