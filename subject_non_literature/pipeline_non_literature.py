import os
import re
import json
import cv2
import base64
import unicodedata
import easyocr
from config import OPENAI_CLIENT as client
from config import NON_LITERATURE_DATA, NON_LITERATURE_IMAGE_DIR
from PIL import Image
from IPython.display import display
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
# from langchain.embeddings import HuggingFaceEmbeddings

# ✅ 설정
reader = easyocr.Reader(['ko', 'en'], gpu=True)

MODEL_ROOT = "/app/models"  # 컨테이너 내에서의 경로

models = {
    "roberta": SentenceTransformer(os.path.join(MODEL_ROOT, "jhgan-ko-sroberta-multitask")),
    "snunlp": SentenceTransformer(os.path.join(MODEL_ROOT, "snunlp-kr-sbert-v40k")),
    "kosimcse": SentenceTransformer(os.path.join(MODEL_ROOT, "kosimcse-roberta-multitask")),
}

model = models["kosimcse"]


# ✅ 유니코드 정규화
def normalize_filename(fn):
    return unicodedata.normalize('NFC', fn)

# ✅ 텍스트 추출
def extract_text_with_underlines(image_path):
    img = cv2.imread(image_path)
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
       ① 오답인 이유 ...
       ② 오답인 이유 ...
       ④ 오답인 이유 ...
       ⑤ 오답인 이유 ...
'''

# ✅ GPT 질의
def ask_gpt(prompt_text):
    resp = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt_text}],
        temperature=0.3
    )
    return resp.choices[0].message.content

# ✅ GPT 태깅용 프롬프트
tag_prompt = '''
아래는 국어 영역 비문학 문제입니다.
이 문제에 대해 다음 정보를 태그로 생성해 주세요(최대 6개):
- 출제 영역(비문학)
- 장르('인문-언어', '인문-철학', '사회-경제', '기술-컴퓨터', '사회-법', '과학-생물', '인문-역사', '과학-화학', '인문-심리', '예술-미술', '과학-물리', '사회-정치', '예술-음악', '예술-영화')
- 문제 유형(단일문제/복합문제)
- 주요 주제 또는 키워드

지문이 (가), (나)로 이루어지면 복합문제입니다.

태그는 쉼표로 구분해서 출력해 주세요.
예) 비문학, 사회-경제, 복합문제, 공공재, 정책 딜레마, 지방 정부 재정 지원
'''

def tag_from_image(image_path):
    with open(image_path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode("utf-8")

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "너는 국어 문제 태깅 전문가야."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 국어 지문 이미지를 보고 태깅해 줘:\n" + tag_prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ]
            }
        ],
        temperature=0.2
    )

    tags = response.choices[0].message.content.strip().split(",")
    tags = [t.strip() for t in tags if t.strip()]
    return {
        "passage": "[이미지 기반 지문 생략]",
        "genre": tags[1],
        "question_type": "복합문제" if "복합" in tags[2] else "단일문제",
        "keywords": tags[3:]
    }

# ✅ 유사 문제 추천
# ✅ 올바른 예시: 파일 이름까지 포함해야 함
json_file_path = os.path.join(NON_LITERATURE_DATA, "non-literature_cleaned.json")
with open(json_file_path, encoding="utf-8") as f:
    data = json.load(f)

passages = [item["passage"] for item in data]
embeddings = model.encode(passages, convert_to_tensor=False)

def recommend_for_external_problem(target_problem, top_n=3):
    target_passage = target_problem["passage"]
    target_keywords = target_problem["keywords"]
    target_type = target_problem["question_type"]
    target_genre = target_problem["genre"]

    target_embedding = model.encode(target_passage, convert_to_tensor=False)
    passage_similarities = cosine_similarity([target_embedding], embeddings)[0]

    target_kw_string = " ".join(target_keywords)
    target_kw_embedding = model.encode(target_kw_string, convert_to_tensor=False)

    results = []
    for idx, item in enumerate(data):
        if target_genre not in item["genre"]:
          continue
        item_kw_string = " ".join(item["keywords"])
        item_kw_embedding = model.encode(item_kw_string, convert_to_tensor=False)
        keyword_sim = cosine_similarity([target_kw_embedding], [item_kw_embedding])[0][0]
        passage_sim = passage_similarities[idx]
        final_score = 0.5 * keyword_sim + 0.5 * passage_sim
        if item["question_type"] == target_type:
            final_score += 0.2

        results.append({
            "year": item["year"],
            "month": item["month"],
            "pNum": item["pNum"],
            "start_Qnum": item.get("start_Qnum"),
            "end_Qnum": item.get("end_Qnum"),
            "score": round(final_score, 4),
            "preview": item["passage"][:100]
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]

# ✅ 유사 문제 이미지 출력
def show_problem_image_set(similar_problems, image_base=str(NON_LITERATURE_IMAGE_DIR)):
    for i, p in enumerate(similar_problems, 1):
        year, month, pNum = p["year"], p["month"], p["pNum"]
        print(f"\n--- 유사 문제 {i} --- \n\t{year}-{month}\n")
        img_path = os.path.join(image_base, f"{year}-{month}-국어_p{pNum}.png")
        if os.path.exists(img_path):
            display(Image.open(img_path))
        for q in range(p.get("start_Qnum", 0), p.get("end_Qnum", 0) + 1):
            q_path = os.path.join(image_base, f"{year}-{month}-국어_{q}.png")
            if os.path.exists(q_path):
                display(Image.open(q_path))

# ✅ 통합 실행 함수
def solve_and_recommend(passage_img_path, question_img_path):
    if not os.path.exists(passage_img_path) or not os.path.exists(question_img_path):
        print("❌ 지문 또는 문제 이미지가 존재하지 않습니다.")
        return

    passage_text = extract_text_with_underlines(passage_img_path)
    question_text = extract_text_with_underlines(question_img_path)

    prompt = text_prompt.format(passage=passage_text, question=question_text)
    print("📘 GPT 답변 및 해설:")
    try:
        gpt_output = ask_gpt(prompt)
        print(gpt_output)
    except Exception as e:
        print(f"⚠️ GPT 실패: {e}")
        return

    print("\n\n🔖 유사 문제 추천:")
    try:
        external_problem = tag_from_image(passage_img_path)
        similar_problems = recommend_for_external_problem(external_problem)
        if similar_problems:
            show_problem_image_set(similar_problems)
        else:
            print("추천된 유사 문제가 없습니다.")
    except Exception as e:
        print(f"⚠️ 유사 문제 추천 실패: {e}")

def attach_image_paths(problem, base_path):
    year, month, pNum = problem["year"], problem["month"], problem["pNum"]

    problem["passage_img"] = os.path.join(base_path, f"{year}-{month}-국어_p{pNum}.png")

    problem["problem_imgs"] = []
    for q in range(problem.get("start_Qnum", 0), problem.get("end_Qnum", 0) + 1):
        img_path = os.path.join(base_path, f"{year}-{month}-국어_{q}.png")
        if os.path.exists(img_path):
            problem["problem_imgs"].append((q, img_path))
    return problem

def pipeline_non_literature(passage_img_path, question_img_path, top_k=10):
    # 1. 이미지 OCR
    passage_text = extract_text_with_underlines(passage_img_path)
    question_text = extract_text_with_underlines(question_img_path)

    # 2. GPT 응답
    prompt = text_prompt.format(passage=passage_text, question=question_text)
    try:
        gpt_output = ask_gpt(prompt)
    except Exception as e:
        gpt_output = f"⚠️ GPT 응답 실패: {e}"

    # 3. 태깅 및 유사 문제 추천
    # ✅ ✅ 주의: '.../img/비문학' ❌ → '.../img' ✅
    image_base_path = str(NON_LITERATURE_IMAGE_DIR)

    try:
        # GPT로 지문 태깅
        tags = tag_from_image(passage_img_path)

        # 태깅 정보를 기반으로 유사 문제 추천
        similar_raw = recommend_for_external_problem(tags, top_n=top_k)

        # 추천된 문제에 이미지 경로 붙이기
        similar = [attach_image_paths(p, image_base_path) for p in similar_raw]

    except Exception as e:
        similar = []
        gpt_output += f"\n\n⚠️ 유사 문제 추천 실패: {e}"

    return gpt_output, similar