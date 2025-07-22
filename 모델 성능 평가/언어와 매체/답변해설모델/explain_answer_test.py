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
image_dir = "C:/Users/user/STUBO/langmed/output_images/"

client = OpenAI(api_key="   ")
reader = easyocr.Reader(['ko', 'en'], gpu=True)

# ✅ 유니코드 정규화
def normalize_filename(fn):
    return unicodedata.normalize('NFC', fn)

# ✅ 정답지 (JSON에서 로드한 대신 코드에 직접 포함)
true_answers = {
    "2022-03": {35: "①",
    36: "①",
    37: "④",
    38: "④",
    39: "①",
    40: "⑤",
    41: "③",
    42: "⑤",
    43: "③",
    44: "②",
    45: "②"},

    "2022-06": {35: "③",
    36: "②",
    37: "①",
    38: "⑤",
    39: "③",
    40: "②",
    41: "⑤",
    42: "④",
    43: "④",
    44: "①",
    45: "③"},

    "2022-09": {35: "③",
    36: "③",
    37: "④",
    38: "⑤",
    39: "④",
    40: "⑤",
    41: "②",
    42: "②",
    43: "①",
    44: "③",
    45: "①"
    },

    "2022-수능": {35: "②",
    36: "④",
    37: "①",
    38: "④",
    39: "①",
    40: "②",
    41: "①",
    42: "③",
    43: "⑤",
    44: "④",
    45: "③"
    },

    "2023-03": {35: "④",
    36: "③",
    37: "①",
    38: "④",
    39: "⑤",
    40: "②",
    41: "④",
    42: "⑤",
    43: "①",
    44: "②",
    45: "③"
    },

    "2023-06": {35: "③",
    36: "⑤",
    37: "②",
    38: "④",
    39: "④",
    40: "②",
    41: "①",
    42: "⑤",
    43: "⑤",
    44: "③",
    45: "①"
    },

    "2023-09": {35: "④",
    36: "③",
    37: "⑤",
    38: "②",
    39: "①",
    40: "⑤",
    41: "②",
    42: "⑤",
    43: "①",
    44: "③",
    45: "④"
    },

    "2023-수능": {35: "④",
    36: "④",
    37: "①",
    38: "③",
    39: "④",
    40: "②",
    41: "①",
    42: "⑤",
    43: "③",
    44: "③",
    45: "⑤"
    },

    "2024-03": {35: "①",
    36: "④",
    37: "④",
    38: "②",
    39: "③",
    40: "⑤",
    41: "④",
    42: "④",
    43: "①",
    44: "⑤",
    45: "②"
    },

    "2024-06": {35: "④",
    36: "⑤",
    37: "①",
    38: "①",
    39: "③",
    40: "②",
    41: "②",
    42: "④",
    43: "②",
    44: "⑤",
    45: "⑤"
    },

    "2024-09": {35: "④",
    36: "③",
    37: "④",
    38: "①",
    39: "⑤",
    40: "①",
    41: "③",
    42: "②",
    43: "④",
    44: "④",
    45: "②"
    },

    "2024-수능": {35: "⑤",
    36: "④",
    37: "③",
    38: "①",
    39: "②",
    40: "④",
    41: "③",
    42: "⑤",
    43: "②",
    44: "①",
    45: "③"
    },

    "2025-03": {35: "②",
    36: "④",
    37: "④",
    38: "③",
    39: "⑤",
    40: "①",
    41: "④",
    42: "③",
    43: "⑤",
    44: "③",
    45: "③"
    },

    "2025-06": {35: "④",
    36: "②",
    37: "⑤",
    38: "①",
    39: "③",
    40: "⑤",
    41: "①",
    42: "④",
    43: "①",
    44: "④",
    45: "②"
    },
    
}

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
        temperature=0.3
    )
    return resp.choices[0].message.content

def parse_gpt_output(output):
    a = re.search(r'\[정답\]\s*([①-⑤])', output)
    e = re.search(r'\[해설\](.*)', output, re.DOTALL)
    return (a.group(1) if a else None), (e.group(1).strip() if e else "")

# ✅ 오답 저장
def log_wrong_answer(qid, gpt_answer, true_answer, passage, question, explanation):
    entry = {
        "문제ID": qid,
        "GPT_정답": gpt_answer,
        "실제정답": true_answer,
        "지문": passage,
        "문제": question,
        "GPT_해설": explanation
    }
    

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

# ✅ 실행 대상
target_sets = [
    (2022, "03"), (2022, "06"), (2022, "09"), (2022, "수능"),
    (2023, "03"), (2023, "06"), (2023, "09"), (2023, "수능"),
    (2024, "03"), (2024, "06"), (2024, "09"), (2024, "수능"),
    (2025, "03"), (2025, "06")
]

wrong = 0
count = 0
# ✅ 실행 루프
for year, month in target_sets:
    exam_key = f"{year}-{month}"
    if exam_key not in true_answers:
        print(f"⚠️ 정답 없음: {exam_key}")
        continue

    # 문제번호별 지문 파일 매핑
    passage_mapping = get_passage_mapping(year, month)

    # 지문 OCR 미리 처리
    passage_cache = {}
    for pcode in set(passage_mapping.values()):
        p_path = normalize_filename(f"{year}-{month}-언매_{pcode}.png")
        p_img = os.path.join(image_dir, p_path)
        if os.path.exists(p_img):
            passage_cache[pcode] = extract_text_with_underlines(p_img)
        else:
            passage_cache[pcode] = ""
            print(f"⚠️ 지문 이미지 없음: {p_img}")

    # 문제 35~45 순차 처리
    for qn in range(35, 46):
        q_path = normalize_filename(f"{year}-{month}-언매_{qn}.png")
        q_img = os.path.join(image_dir, q_path)
        if not os.path.exists(q_img):
            print(f"❌ 문제 이미지 없음: {q_img}")
            continue

        question = extract_text_with_underlines(q_img)
        pcode = passage_mapping.get(qn)
        passage = passage_cache.get(pcode, "") if pcode else ""

        prompt = text_prompt.format(passage=passage, question=question)
        print(f"\n📘 [{exam_key} / 문제 {qn}]")

        try:
            gpt_output = ask_gpt(prompt)
            gpt_ans, explanation = parse_gpt_output(gpt_output)
            true_ans = true_answers[exam_key].get(qn)

            print(f"GPT: {gpt_ans}, 정답: {true_ans}")
            if gpt_ans != true_ans:
                print("❌ 오답 기록")
                wrong += 1
                count += 1
                qid = f"{exam_key}-{qn}"
                log_wrong_answer(qid, gpt_ans, true_ans, passage, question, explanation)
            else:
                count += 1
        except Exception as e:
            print(f"⚠️ GPT 실패: {e}")

print(f"✅ 총 문제 개수: {count}")
print(f"✅ 오답 개수: {wrong}")
print("✅ 정답률 : {(count - wrong) / count:.2%}" if count > 0 else "✅ 총 문제 개수: 0")