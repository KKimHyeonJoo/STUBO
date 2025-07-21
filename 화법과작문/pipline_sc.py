# 유사 문제 추천과 답변 해설을 동시에 하는 최종 모델
import openai
import os
import base64
import easyocr
import pytesseract
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# GPT client 설정
client = openai.OpenAI(api_key="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")  # API 키 입력 필요

# EasyOCR 설정
reader = easyocr.Reader(['ko', 'en'])

# Embedding model & vectorstore 로딩
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  # 현재 py 기준 절대 경로
outputs_path = os.path.join(os.path.dirname(BASE_DIR), "outputs")       # 상위 디렉토리의 outputs 폴더

embedding_model = SentenceTransformerEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
vectorstore = FAISS.load_local(outputs_path, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def easyocr_text(image_path):
    result = reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(result)

def pytesseract_text(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang="kor+eng")

def analyze_problem(context_image_path, question_image_path):
    print("\n===== [1] GPT 정답/해설 생성 =====")
    context_text = easyocr_text(context_image_path)
    question_text = easyocr_text(question_image_path)

    c_b64 = encode_image(context_image_path)
    q_b64 = encode_image(question_image_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"다음은 OCR로 추출한 지문 텍스트입니다:\n{context_text}\n\n"
                        f"다음은 OCR로 추출한 문제와 선택지 텍스트입니다:\n{question_text}\n\n"
                        "위 두 이미지와 OCR 텍스트를 참고하여 문제의 정답과 해설을 알려줘. "
                        "정답 번호와 해설을 분리해서 알려줘. (예: 정답: ③)"
                        "모든 선지에 대한 해설을 제공해줘."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{c_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{q_b64}"}},
                ]
            }
        ],
        max_tokens=5000,
    )
    print("\n[정답/해설 GPT 응답]\n")
    print(response.choices[0].message.content)

    print("\n===== [2] 유사 문제 추천 =====")
    full_text = pytesseract_text(question_image_path)

    similar_docs = retriever.get_relevant_documents(full_text)
    for i, doc in enumerate(similar_docs, 1):
        meta = doc.metadata
        print(f"\n{i}. {meta.get('년도', '연도 정보 없음')}년 {meta.get('월', '월 정보 없음')}월")
        print(doc.page_content)
        if meta.get("답"): print("정답:", meta["답"])
        if meta.get("해설"): print("해설:", meta["해설"])
        print("-" * 60)

# 사용 예시:
analyze_problem("/Users/chaewon/Desktop/STUBO/화법과작문/output_images/2021-03-화작_p10.png", "/Users/chaewon/Desktop/STUBO/화법과작문/output_images/2021-03-화작_42.png")
