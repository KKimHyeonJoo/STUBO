import os
os.environ["OPENAI_API_KEY"] = "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"

import base64
from PIL import Image
from openai import OpenAI


"""## Easy OCR"""

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
image_dir = "/content/drive/MyDrive/classified_images/비문학"
client = OpenAI()
reader = easyocr.Reader(['ko', 'en'], gpu=True)

# ✅ 유니코드 정규화
def normalize_filename(fn):
    return unicodedata.normalize('NFC', fn)

def extract_text_with_underlines(image_path):
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"이미지 경로 없음: {image_path}")
    
    # PIL 로드 후 numpy 배열로 변환
    pil_img = Image.open(image_path).convert("RGB")
    img = np.array(pil_img)

    # OpenCV와 호환되도록 BGR로 변환
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

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
다음은 국어 비문학 문제입니다. 지문과 문제(선택지 포함)를 꼼꼼히 읽고 다음을 수행하세요:

1. 질문 조건을 정확히 반영해 정답을 선택하세요.
2. 반드시 ①~⑤ 중 하나만 골라 [정답] ③ 형식으로 답하세요.
3. 지문에 근거한 해설을 [해설]로 3~5문장 쓰세요.

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
        temperature=0.3
    )
    return resp.choices[0].message.content

def parse_gpt_output(output):
    a = re.search(r'\[정답\]\s*([①-⑤])', output)
    e = re.search(r'\[해설\](.*)', output, re.DOTALL)
    return (a.group(1) if a else None), (e.group(1).strip() if e else "")

# ✅ 매핑
def get_mapping(year, month):
    special = (year, month) in [(2024, "06"), (2024, "09"), (2024, "수능"), (2025, "06"), (2025, "09")]
    return {
        "p1": [1, 2, 3],
        "p2": list(range(4, 8)) if special else list(range(4, 10)),
        "p3": list(range(8, 12)) if special else list(range(10, 14)),
        "p4": list(range(12, 18)) if special else list(range(14, 18))
    }

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
    r"C:/Users/user/STUBO/언어와매체/output_images/2024-수능-언매_35.png", # 사용자 직접 입력
    r"C:/Users/user/STUBO/언어와매체/output_images/2024-수능-언매_p9.png" # 사용자 직접 입력
    
)