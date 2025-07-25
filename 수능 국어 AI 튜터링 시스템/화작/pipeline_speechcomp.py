# -*- coding: utf-8 -*-
"""pipeline_speechcomp.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1sGEBK0mgK3UxnDkuqjc9YtHCGTNq-oOj
"""

import os

if __name__ == "__main__":  # 이 파일이 직접 실행될 때만
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except:
        pass

# 유사 문제 추천과 답변 해설을 동시에 하는 최종 모델
import openai
import os
import base64
import easyocr
import pytesseract
import re
from PIL import Image
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings

# GPT client 설정
client = openai.OpenAI(api_key="  ")  # API 키 입력 필요

# 경로 설정
BASE_DIR = "/content/drive/MyDrive/Colab Notebooks/TAVE 프로젝트_STUBO/수능 국어 AI 튜터링 시스템/화작"
DATA_DIR = os.path.join(BASE_DIR, "data")
VECTOR_DB_PATH = os.path.join(BASE_DIR, "outputs")  # FAISS vector DB가 저장된 폴더

# [5] EasyOCR Reader
reader = easyocr.Reader(['ko', 'en'])

# [6] 임베딩 모델 & 벡터 스토어 로딩
embedding_model = SentenceTransformerEmbeddings(model_name="snunlp/KR-SBERT-V40K-klueNLI-augSTS")
vectorstore = FAISS.load_local(VECTOR_DB_PATH, embedding_model, allow_dangerous_deserialization=True)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# [7] 이미지 인코딩
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# [8] OCR 함수
def easyocr_text(image_path):
    result = reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(result)

def pytesseract_text(image_path):
    img = Image.open(image_path)
    return pytesseract.image_to_string(img, lang="kor+eng")

def extract_problem_number(text: str) -> str:
    match = re.match(r"\s*(\d+)[\.\)]?\s*문제", text)
    return match.group(1) if match else "번호 없음"

def analyze_problem(context_image_path, question_image_path, top_k=2):
    result_dict = {}

    # 🔧 내부 설정
    base_img_dir = "/content/drive/MyDrive/Colab Notebooks/TAVE 프로젝트_STUBO/수능 국어 AI 튜터링 시스템/화작/data/output_images"
    page_map = {
        "p9": [35, 36, 37],
        "p10": [38, 39, 40, 41, 42],
        "p11": [43, 44, 45],
    }

    # [1] GPT 정답/해설 생성
    context_text = easyocr_text(context_image_path)
    question_text = easyocr_text(question_image_path)

    c_b64 = encode_image(context_image_path)
    q_b64 = encode_image(question_image_path)

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"다음은 OCR로 추출한 지문 텍스트입니다:\n{context_text}\n\n"
                        f"다음은 OCR로 추출한 문제와 선택지 텍스트입니다:\n{question_text}\n\n"
                        "위 두 이미지와 OCR 텍스트를 참고하여 문제의 정답과 해설을 알려줘. "
                        "정답 번호와 해설을 분리해서 알려줘. (예: 정답: ③) "
                        "모든 선지에 대한 해설을 제공해줘."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{c_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{q_b64}"}},
                ]
            }
        ],
        max_tokens=5000,
    )

    gpt_response = response.choices[0].message.content
    full_text = pytesseract_text(question_image_path)

    # [2] 유사 문제 추천
    similar_docs = retriever.get_relevant_documents(full_text, k=top_k)
    similar_problems = []

    for i, doc in enumerate(similar_docs, 1):
        meta = doc.metadata
        year = str(meta.get("년도", "연도 없음"))
        month = str(meta.get("월", "월 없음")).zfill(2)
        question_text = doc.page_content

        # 문제 번호 추출
        match = re.match(r"\s*(\d+)[\.\)]?", question_text)
        problem_number = match.group(1) if match else "번호없음"

        # 이미지 파일명 추론
        problem_filename = f"{year}-{month}-화작_{problem_number}.png"
        passage_filename = meta.get("passage_img", "")

        # 지문 추론
        if not passage_filename and problem_number.isdigit():
            for page, numbers in page_map.items():
                if int(problem_number) in numbers:
                    passage_filename = f"{year}-{month}-화작_{page}.png"
                    break

        # 전체 경로 생성
        passage_img_path = os.path.join(base_img_dir, passage_filename) if passage_filename else ""
        problem_img_path = os.path.join(base_img_dir, problem_filename)

        similar_problems.append({
            "index": i,
            "year": year,
            "month": month,
            "answer": meta.get("답", None),
            "explanation": meta.get("해설", None),
            "content": question_text,
            "problem_number": problem_number,
            "passage_img_path": passage_img_path,
            "problem_img_path": problem_img_path,
        })

    return {
        "problem_number": extract_problem_number(full_text),
        "gpt_response": gpt_response,
        "similar_problems": similar_problems,
    }