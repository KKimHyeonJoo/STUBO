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
from PIL import Image
from openai import OpenAI

input_dir = r"C:/Users/user/STUBO/언어와매체/save_json_tagged/"

# 파일 이름 정렬
file_list = sorted([f for f in os.listdir(input_dir) if f.endswith(".json")])

# 평가용 파일 지정 (예: 2025-06-언매.json)
eval_filename = ["2025-03-언매.json", "2025-06-언매.json"]


MEDIA_RELATED_TAGS = {
    "정보 제시 이해", "매체 구성 요소 분석", "매체 표현 방식 이해", "매체의 목적 분석",
    "매체의 효과 분석", "매체의 시각적 요소 이해", "매체의 청각적 요소 이해",
    "매체의 상징적 의미 이해", "매체의 사회적 맥락 이해", "대화 내용 반영",
    "비판적 사고", "자료 분석", "적절하지 않은 것 찾기", "문장 요소 분석",
    "게시판 구성 이해", "인터뷰 분석", "시각 정보 해석", "그래프 이해", "도표 분석"
}

def get_problem_type(tag_list):
    return "매체" if any(tag in MEDIA_RELATED_TAGS for tag in tag_list) else "언어"

train_set = []
eval_set = []

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

    if file_name in eval_filename:
        eval_set.extend(problems)
    else:
        train_set.extend(problems)


models = {
    "ko-sbert-nli": SentenceTransformer("jhgan/ko-sbert-nli"),
    "kr-sbert-augsts": SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS"),
    "kosimcse": SentenceTransformer("BM-K/KoSimCSE-roberta-multitask")
}


os.environ["OPENAI_API_KEY"] = "   "


def normalize_filename(name):
    # 유니코드 정규화, 공백/특수문자/언더스코어/하이픈 제거, 소문자 변환
    name = unicodedata.normalize('NFC', name)
    name = name.replace(' ', '').replace('_', '').replace('-', '').lower()
    return name



def recommend_for_external_problem(target_problem, train_set, model, top_n=3):
    target_tags = target_problem.get("tags", [])
    target_kw_embedding = model.encode(" ".join(target_tags), convert_to_tensor=False)

    target_passage = target_problem.get("지문", "")
    has_target_passage = bool(target_passage.strip())
    if has_target_passage:
        target_embedding = model.encode(target_passage, convert_to_tensor=False)

    # 1. 문제 유형 판별 (매체 포함 여부)
    target_type = "매체" if any("매체" in tag for tag in target_tags) else "언어"

    results = []
    for item in train_set:
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

    # 쉼표로 분리된 태그 문자열을 리스트로 변환
    tag_list = [tag.strip() for tag in content.split(",") if tag.strip()]

    # 유형 판별: 문맥상 매체 관련 태그가 하나라도 있으면 매체로 간주
    if any(tag in MEDIA_RELATED_TAGS for tag in tag_list):
        qtype = "매체"
    else:
        qtype = "언어"

    return {
        "tags": tag_list,
        "type": qtype,
        "지문": "[이미지 기반 지문 생략]",
    }


def extract_number_from_question(qtext):
    match = re.match(r"^\s*(\d+)\.", qtext)
    return match.group(1) if match else ""

eval_set_normalized = {
    normalize_filename(f"{item.get('년', '')}-{item.get('월', '')}-언매_{extract_number_from_question(item.get('문제', ''))}"): item
    for item in eval_set
}

# ===== 4. 이미지 폴더 반복 처리 + 평가 =====
folder_path = "C:/Users/user/STUBO/언어와매체/유사기출"
image_files = sorted([f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])

scores = {name: {"top1": 0, "top2": 0, "top3": 0} for name in models}

for idx, image_name in enumerate(image_files, 1):
    image_name_norm = normalize_filename(image_name)
    image_path = os.path.join(folder_path, image_name)
    print(f"\n=== [{idx}/{len(image_files)}] 이미지: {image_name} ===")
    #gt = [normalize_id(x) for x in eval_set_normalized.get(image_name_norm, {}).get("ground_truth", [])]

    # STEP 1: GPT Vision 태깅
    external_problem = tag_from_image(image_path)
    
    # STEP 2: 모델별 추천 및 평가
    for model_name, model in models.items():
        print(f"\n[{model_name}] 추천 결과:")
        recommendations = recommend_for_external_problem(external_problem, train_set, model=model)

        for rank, rec in enumerate(recommendations):
            print(f"{rec['id_str']} | 점수: {rec['score']} (임베딩: {rec['embedding_sim']}, 키워드: {rec['keyword_cosine']})")
            print(f"  지문 미리보기: {rec['preview']}\n")
            

# ===== 5. 정확도 통계 출력 =====
for name in scores:
    s = scores[name]
    s["total"] = s["top1"] + s["top2"] + s["top3"]

df = pd.DataFrame(scores).T
print("\n=== 모델별 Top-k 정확도 ===")
print(df[["top1", "top2", "top3", "total"]])


# 현재 파일명 정규화
image_name_normalized = normalize_filename(image_name)


