import os
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import InMemoryStore
from langchain_core.documents import Document
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.vectorstores import FAISS
import uuid
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
import json
import unicodedata
import re
import base64
from IPython.display import display
from PIL import Image
from openai import OpenAI

input_dir = r"C:/Users/user/STUBO/언어와매체/save_json_tagged/"
os.environ["OPENAI_API_KEY"] = "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"


# 파일 이름 정렬
file_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])

# 1. 모델과 데이터 준비
print("모델과 데이터 불러오는 중...")
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

data = []
for file_name in file_list:
    file_path = os.path.join(input_dir, file_name)
    with open(file_path, encoding="utf-8") as f:
        problems = json.load(f)
        data.extend(problems)

passages = [item["지문"] for item in data]
embeddings = model.encode(passages, convert_to_tensor=False)


MEDIA_RELATED_TAGS = {
    "정보 제시 이해", "매체 구성 요소 분석", "매체 표현 방식 이해", "매체의 목적 분석",
    "매체의 효과 분석", "매체의 시각적 요소 이해", "매체의 청각적 요소 이해",
    "매체의 상징적 의미 이해", "매체의 사회적 맥락 이해", "대화 내용 반영",
    "비판적 사고", "자료 분석", "적절하지 않은 것 찾기", "문장 요소 분석",
    "게시판 구성 이해", "인터뷰 분석", "시각 정보 해석", "그래프 이해", "도표 분석"
}

def get_problem_type(tag_list):
    return "매체" if any(tag in MEDIA_RELATED_TAGS for tag in tag_list) else "언어"


for file_name in file_list:
    full_path = os.path.join(input_dir, file_name)
    with open(full_path, "r", encoding="utf-8") as f:
        problems = json.load(f)
    
    for prob in problems:
        # tags 필드를 문자열에서 리스트로 변환 (쉼표 기준)
        tags = prob.get("tags", [])
        if isinstance(tags, str):
            tags = [tag.strip() for tag in tags.split(",") if tag.strip()]
        prob["tags"] = tags

        # 유형 추가
        prob["type"] = get_problem_type(tags)



'''
def normalize_filename(name):
    # 유니코드 정규화, 공백/특수문자/언더스코어/하이픈 제거, 소문자 변환
    name = unicodedata.normalize('NFC', name)
    name = name.replace(' ', '').replace('_', '').replace('-', '').lower()
    return name
'''

def extract_number_from_question(qtext):
    match = re.match(r"^\s*(\d+)\.", qtext)
    return match.group(1) if match else ""


def recommend_for_external_problem(target_problem, top_n=3):
    target_tags = target_problem.get("tags", [])
    target_kw_embedding = model.encode(" ".join(target_tags), convert_to_tensor=False)

    target_passage = target_problem.get("지문", "")
    has_target_passage = bool(target_passage.strip())
    if has_target_passage:
        target_embedding = model.encode(target_passage, convert_to_tensor=False)

    # 1. 문제 유형 판별 (매체 포함 여부)
    target_type = "매체" if any("매체" in tag for tag in target_tags) else "언어"

    results = []
    for item in data:
        item_tags = item.get("tags", [])
        item_passage = item.get("지문", "")
        has_item_passage = bool(item_passage.strip())

        if not item_tags:
            continue  # 태그가 없으면 제외

        # 2. 유형이 다르면 스킵
        item_type = "매체" if any("매체" in tag for tag in item_tags) else "언어"
        if item_type != target_type:
            continue

        # 태그 유사도
        item_kw_embedding = model.encode(" ".join(item_tags), convert_to_tensor=False)
        keyword_sim = cosine_similarity([target_kw_embedding], [item_kw_embedding])[0][0]

        # 지문 유사도
        if has_item_passage:
            item_embedding = model.encode(item_passage, convert_to_tensor=False)
            passage_sim = cosine_similarity([target_embedding], [item_embedding])[0][0]
            final_score = 0.5 * passage_sim + 0.5 * keyword_sim
        else:
            passage_sim = None
            final_score = keyword_sim

        results.append({
            "year": item.get("년", ""),
            "month": item.get("월", ""),
            "문제" : item.get("문제", ""),
            "id_str": f"{item.get('년', '??')}학년도 {item.get('월', '??')} 지문 {extract_number_from_question(item.get('문제', ''))}번",
            "score": round(final_score, 4),
            "embedding_sim": round(passage_sim, 4) if passage_sim is not None else None,
            "keyword_cosine": round(keyword_sim, 4),
            "preview": item_passage[:100]
        })

    return sorted(results, key=lambda x: x["score"], reverse=True)[:top_n]




def tag_from_image(image_path):
    with open(image_path, "rb") as image_file:
        base64_image = base64.b64encode(image_file.read()).decode("utf-8")
    
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "너는 국어 문제 이미지 분석 전문가야."},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text":
                     """
                        아래는 국어 영역 언어와 매체 문제입니다.
                        유형이 '언어' 인 경우, 언어 관련 태그를, '매체'인 경우 매체 관련 태그를 생성해 주세요.

                        유형이 '언어'인 경우에는, 언어 관련 태그를 5개 생성하고, 
                        유형이 '매체'인 경우에는, 매체 관련 태그를 3개, 문제의 출제 의도 태그를 2개씩 생성하여 총 5개의 태그를 생성해주세요.
                        태그는 쉼표로 구분해서 출력해 주세요.
                        
                        언어 관련 태그의 후보에는 '훈민정음 이해', '중세 국어 변화', '단어 변화 양상', '모음 조화 이해', '시간 표현 이해', '부사어', '조사어', '서술어 자릿수', '사잇소리 표기 분석', '동음이의어 분석', '다의어 분석', 
                        '자음 분석', '음운의 변동', '간접 인용', '종결 어미', '발화 형식 비교', '관형사절', '시간적 관계 이해', '사동 및 피동 표현', '주동 및 능동 표현', '문장 성분 분석' 등이 있고,
                        
                        위에서 언급한 언어 관련 태그도 예시일 뿐, 이 중에서만 찾으려고 하지말고, 다른 태그의 후보도 적극적으로 다양하게 찾으세요.

                        매체 관련 태그의 후보에는 '정보 제시 이해', '매체 구성 요소 분석', '매체 표현 방식 이해', '매체의 목적 분석', '매체의 효과 분석', '매체의 시각적 요소 이해', '매체의 청각적 요소 이해', '매체의 상징적 의미 이해', '매체의 사회적 맥락 이해' 등이 있습니다.
                        그리고 매체 관련 태그의 출제 의도는 '매체 구성 이해', '대화 내용 반영', '정보 제공 방식 분석', '비판적 사고', '자료 분석', '문장 요소 분석', '적절하지 않은 것 찾기' 등이 있습니다.

                        매체 관련 태그의 후보, 출제 의도도 위에서 적어준 것은 예시일 뿐, 절대 이 중에서만 찾으려고 하지말고, 적극적으로 다양하게 찾으세요.
                        언어 관련한 문제 출력 예시) 안긴문장 분석, 서술어 자릿수, 의미 관계 파악
                        매체 관련한 문제 출력 예시) 게시판 구성 이해, 대화 내용 반영, 정보 제공 방식 분석, 비판적 사고, 자료 분석
                            {text}
                     """},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        temperature=0.5
    )

    content = response.choices[0].message.content.strip()
    print("GPT Vision 응답 원본:\n", content)

    # 태그를 리스트로 분리
    tag_list = [tag.strip() for tag in content.split(",") if tag.strip()]

    # 유형 판별: 매체 관련 태그 중 하나라도 포함되면 매체
    problem_type = "매체" if any(tag in MEDIA_RELATED_TAGS for tag in tag_list) else "언어"

    return {
        "tags": tag_list,
        "type": problem_type,
        "지문": "[이미지 기반 지문 생략]",
    }

# ✅ 매핑
def get_passage_mapping(year, month):
    # 예외 정의
    p9_range = {
        (2022, "03"): range(38, 40),  # 38~39
    }

    p10_range = {
        (2022, "09"): range(40, 43),
        (2024, "09"): range(40, 43),
    }

    # 기본값
    p9_default = range(35, 37)
    p10_default = range(40, 44)

    y, m = year, month

    mapping = {}

    # p9
    p9 = p9_range.get((y, m), p9_default)
    for i in p9:
        mapping[i] = "p9"

    # p10
    p10 = p10_range.get((y, m), p10_default)
    for i in p10:
        mapping[i] = "p10"

    # p11
    p11_start = max(p10) + 1
    for i in range(p11_start, 46):
        mapping[i] = "p11"

    return mapping



def show_problem_image_set(similar_problems, image_base="C:/Users/user/STUBO/언어와매체/output_images/"):
    for i, p in enumerate(similar_problems, 1):
        year = p.get('year', '정보 없음')
        month = p.get('month', '정보 없음')
        qtext = p.get('문제', '') or ''
        number = extract_number_from_question(qtext)

        if year == '정보 없음' or month == '정보 없음' or not number:
            print(f"\n[{i}] ⚠️ 정보 없음: {year}-{month} / 번호: {number}")
            continue

        # 월을 항상 두 자리로 보정
        if month != '정보 없음':
            month_str = str(month).zfill(2)
        else:
            month_str = '정보 없음'

        if year == '정보 없음' or month_str == '정보 없음' or not number:
            print(f"\n[{i}] ⚠️ 정보 없음: {year}-{month_str} / 번호: {number}")
            continue

        number = int(number)
        mapping = get_passage_mapping(int(year), month_str)
        passage_key = mapping.get(number, None)

        print(f"\n--- 유사 문제 {i} ---")
        print(f"{year}-{month_str} / {number}번")

        # 지문 먼저 출력
        if passage_key:
            passage_img_path = os.path.join(image_base, f"{year}-{month_str}-언매_{passage_key}.png")
            print(passage_img_path)
            if os.path.exists(passage_img_path):
                print(f"[지문: {passage_key}]")
                Image.open(passage_img_path).show()
            else:
                print(f"[지문 {passage_key} 이미지 없음]")

        # 문제 이미지 출력
        problem_img_path = os.path.join(image_base, f"{year}-{month_str}-언매_{number}.png")
        print(problem_img_path)
        if os.path.exists(problem_img_path):
            print(f"[문제: {number}]")
            Image.open(problem_img_path).show()
        else:
            print(f"[문제 {number} 이미지 없음]")


# 1. 테스트할 이미지 파일명
image_name = "2024-06-언매_40.png"

# 2. 폴더 경로
folder_path = "C:/Users/user/STUBO/언어와매체/유사기출"
image_path = os.path.join(folder_path, image_name)

# 3. 존재 여부 확인
if not os.path.exists(image_path):
    print(f"❌ 이미지가 존재하지 않습니다: {image_path}")
else:
    print(f"\n=== 이미지: {image_name} ===")
    img = Image.open(image_path)
    display(img)

    # 4. GPT Vision으로 태깅
    external_problem = tag_from_image(image_path)

    # 5. 유사 문제 추천
    print("\n\n[유사 문제 추천 결과]")
    similar_problems = recommend_for_external_problem(external_problem)

    # 6. 추천된 문제 이미지 표시
    if similar_problems:
        show_problem_image_set(similar_problems)
    else:
        print("최근에 출제된 유사한 기출 문제가 없습니다.")