import os
import re
import json
import cv2
import numpy as np
import unicodedata
import easyocr
from openai import OpenAI
from PIL import Image

# ✅ 설정
image_dir = "C:/Users/user/STUBO/언어와매체/output_images/"

client = OpenAI(api_key="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")
reader = easyocr.Reader(['ko', 'en'], gpu=True)

# ✅ 유니코드 정규화
def normalize_filename(fn):
    return unicodedata.normalize('NFC', fn)


# ✅ 텍스트 추출
def extract_text_with_underlines(image_path):
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)

    results = reader.readtext(gray, detail=True)
    full_text = " ".join([text for (_, text, _) in results])
    return re.sub(r'\b1[\.\)]', '①', full_text)\
             .replace('2.', '②')\
             .replace('3.', '③')\
             .replace('4.', '④')\
             .replace('5.', '⑤')\
             .strip()


# ✅ GPT 프롬프트
text_prompt = '''
다음은 국어 언어와 매체 문제입니다. 지문과 문제(선택지 포함)를 꼼꼼히 읽고 다음을 수행하세요:

1. 질문 조건을 정확히 반영해 정답을 선택하세요.
2. 반드시 ①~⑤ 중 하나만 골라 [정답] ③ 형식으로 답하세요.
3. 지문에 근거한 해설을 [해설]로 3~5문장 쓰세요.

아래 이미지는 문학 문제 하나의 '질문 문장 + ①~⑤ 선택지'가 포함된 이미지야.
    만약 <보기> 문장이 존재한다면 질문 앞에 위치하며, 반드시 포함해서 출력해.

    형식은 아래와 같이 출력해:

    (질문과 <보기> 내용. <보기>가 없다면 생략)
    ① ...
    ② ...
    ③ ...
    ④ ...
    ⑤ ...

    ❗ 절대 설명이나 부가 텍스트를 추가하지 말고 형식 그대로 출력해.

[지문]
{passage}

[문제]
{question}

결과는 다음 형식으로 출력하세요:

[정답] ③
[해설] … (여기에 근거 설명)

'''

# ✅ GPT 실행 및 파싱
def ask_gpt(prompt_text):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.5
    )
    return resp.choices[0].message.content

    

# ✅ 매핑
def get_passage_mapping(year, month):
    # 지문 공유 범위 정의
    p9_38 = (year, month) in [(2022, "03")]
    p10_42 = (year, month) in [(2022, "09"), (2024, "09")]

    mapping = {}
    if p9_38:
        mapping.update({i: "p9" for i in range(38, 40)})
        mapping.update({i: "p10" for i in range(40, 43)})
        mapping.update({i: "p11" for i in range(43, 46)})
    elif p10_42:
        mapping.update({i: "p10" for i in range(40, 43)})
        mapping.update({i: "p11" for i in range(43, 46)})
    else:
        mapping.update({i: "p10" for i in range(40, 44)})
        mapping.update({i: "p11" for i in range(44, 46)})

    return mapping

def solve_question_with_images(passage_img_path, question_img_path):
    # 이미지 경로 유효성 확인
    if not os.path.exists(passage_img_path):
        print(f"❌ 지문 이미지 없음: {passage_img_path}")
        return
    if not os.path.exists(question_img_path):
        print(f"❌ 문제 이미지 없음: {question_img_path}")
        return

    # OCR 텍스트 추출
    passage_text = extract_text_with_underlines(passage_img_path)
    question_text = extract_text_with_underlines(question_img_path)

    # GPT 프롬프트 생성 및 요청
    prompt = text_prompt.format(passage=passage_text, question=question_text)

    try:
        gpt_output = ask_gpt(prompt)
        print("\n📗 GPT 응답:")
        print(gpt_output)
    except Exception as e:
        print(f"⚠️ GPT 실패: {e}")

solve_question_with_images(
    "C:/Users/user/STUBO/언어와매체/output_images/2023-09-언매_p9.png",
    "C:/Users/user/STUBO/언어와매체/output_images/2025-06-언매_35.png"
)
