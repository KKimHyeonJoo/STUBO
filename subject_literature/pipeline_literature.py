import os
import openai
import json
import base64
import io
import re
import mimetypes
import unicodedata
from PIL import Image
from typing import List
from config import OPENAI_CLIENT as client
from config import LITERATURE_DATA, LITERATURE_IMAGE_DIR, LITERATURE_FAISS_EXAMINE
from langchain_openai import OpenAIEmbeddings

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda

"""# **<Retriever (답변,해설 / 유사문제 추천)> 호출**"""

# 벡터스토어 경로
VECTORSTORE_PATH = str(LITERATURE_FAISS_EXAMINE)


# retriever 불러오기 함수
def get_retriever():
    if not os.path.exists(VECTORSTORE_PATH):
        raise FileNotFoundError(f"❌ 벡터스토어가 존재하지 않습니다: {VECTORSTORE_PATH}")

    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.load_local(VECTORSTORE_PATH, embeddings, allow_dangerous_deserialization=True)

    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 5,
            "score_threshold": 0.5
        }
    )
    return vectorstore, retriever

# retriever 인스턴스 생성
vectorstore_literature, retriever_literature = get_retriever()

'''
# 🔹 경로 설정
index_path = str(LITERATURE_FAISS_SIMILAR)
embedding_model = OpenAIEmbeddings()

# 🔹 기존 FAISS 인덱스 로드
retriever_literature_recommend = FAISS.load_local(index_path, embedding_model, allow_dangerous_deserialization=True)
'''

"""# **<이미지(지문/문항)에서 텍스트 추출>**
- 지문 이미지에서 텍스트 추출할때 비용 문제 발생

### **지문 텍스트 추출**
이미지 5등분 -> gpt-4o로 텍스트 추출 -> gpt-4o가 추출한 텍스트와 이미지를 비교하면 특수문자 삽입 -> gpt-4o가 특수문자 검토 -> gpt-4o가 지문 범위([A], [B] 등) 삽입
"""



# ✅ 이미지 → base64
def image_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buffer.getvalue()).decode()}"

# ✅ 이미지 수직 분할
def split_image_vertically(image: Image.Image, parts: int = 5) -> List[Image.Image]:
    width, height = image.size
    part_height = height // parts
    return [
        image.crop((0, i * part_height, width, height if i == parts - 1 else (i + 1) * part_height))
        for i in range(parts)
    ]

# ✅ GPT에게 OCR 시키는 함수
def gpt_ocr_text(image: Image.Image) -> str:
    base64_img = image_to_base64(image)

    system_prompt = """
    너는 수능 국어 문학의 지문 OCR 텍스트 추출 전문가야.

    아래 이미지를 보고 **OCR 텍스트를 그대로 추출**해.
    텍스트 추출만 하고, 절대 가공하거나 설명하지 마.

    📌 반드시 지킬 것:
    - 줄바꿈은 이미지에 보이는 그대로 살려야 해.
    - 띄어쓰기, 특수문자, 괄호, 마침표 등 모든 문장 부호도 그대로 유지해야 해.
    - 해석이나 부연 설명 없이 **순수한 OCR 결과만 출력**해야 해.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {
                            "url": base64_img,
                            "detail": "high"
                        }}
                    ]
                }
            ],
            temperature=0,
            max_tokens=16000
        )
        output = response.choices[0].message.content.strip()
        if not output or "죄송하지만" in output:
            raise ValueError("GPT OCR 실패 또는 결과 없음")
        return output
    except Exception as e:
        return f"[❌ GPT OCR 실패]: {str(e)}"

# ✅ GPT-4o로 특수기호&괄호 삽입
def refine_text_with_gpt(image: Image.Image, ocr_output: str) -> str:
    base64_img = image_to_base64(image)

    system_prompt = """
    너는 수능 국어 문학 지문 정리 전문가야.

    다음은 OCR로 추출한 지문과 원본 이미지야. 너는 이 두 정보를 비교해서 다음과 같이 OCR 결과를 정확하게 수정해야 해.

    📌 반드시 지켜야 할 규칙:

    1. ✅ 이미지 안에서 특수기호(원형 문자/알파벳)를 정확히 식별하고, OCR 텍스트의 알맞은 위치에 삽입해야 해.
      - OCR 결과만 보면 안 돼. 반드시 이미지에서 특수기호 위치를 기준으로 판단해야 해.

    2. ✅ 삽입해야 할 특수기호는 다음과 같아:
      - 한글 원형 문자: ㉠, ㉡, ㉢, ㉣, ㉤
      - 원형 알파벳: ⓐ, ⓑ, ⓒ, ⓓ, ⓔ

    3. ✅ **모든 특수기호 뒤에는 밑줄로 강조된 문장이 반드시 있고**, 그 **강조된 문장 전체를 괄호 '( )'로 반드시 감싸야 해.**
      - 특수기호와 괄호는 붙여서 작성해: 예) ⓐ(강조된 문장)
      - 괄호는 해당 문장의 시작과 끝을 정확히 감싸야 하며, 줄바꿈이나 공백이 있어도 전체를 포함해야 해.

    4. ✅ 한 특수기호에 해당하는 강조 문장이 **여러 줄에 걸쳐 있더라도** 괄호로 정확히 감싸야 해.
      - 중간 줄바꿈이나 공백이 있더라도 강조 문장 전체가 괄호 안에 들어가야 해.

    5. ❌ 특수기호가 없는 문장은 수정하지 마.
      ❌ 철자 오류, 띄어쓰기 오류 등 OCR 자체 오류도 고치지 마.

    6. ✅ 출력은 반드시 지문 전체를 포함해야 하며, 특수기호 + 강조문장 부분만 수정해야 해.
      ❌ 설명, 해설, 추가 정보 없이 **지문 전체만 출력**해야 해.


     ⚠️ 반드시 특수기호 뒤에는 밑줄로 강조된 부분이 괄호'()'로 감싸져 있어야해.
     ⚠️ 특수기호 뒤에 괄호가 없다면 다시 밑줄로 강조된 부분을 다시 찾고 괄호를 추가해줘.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"OCR 추출 결과:\n{ocr_output.strip()}"},
                        {"type": "image_url", "image_url": {
                            "url": base64_img,
                            "detail": "auto"
                        }}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=16000  # 가능한 최대값 사용
        )
        output = response.choices[0].message.content.strip()
        if not output or "죄송하지만" in output or "도와드릴 수 없습니다" in output:
            raise ValueError("GPT 응답 오류 또는 내용 없음")
        return output
    except Exception as e:
        return f"[❌ GPT 정교화 실패]: {str(e)}"

# ✅ 특수기호 위치 점검 및 보정 요청
def verify_special_symbols(original_image: Image.Image, restored_text: str) -> str:
    base64_img = image_to_base64(original_image)

    system_prompt = """
      너는 수능 국어 문학의 지문 OCR 복원 전문가야.

      다음은 OCR로 추출한 지문과 원본 이미지야. 너는 이 두 정보를 비교해서 다음과 같이 OCR 결과를 정확하게 수정해야 해.
      추출한 지문의 특수기호(㉠, ㉡, ㉢, ㉣, ㉤, ⓐ, ⓑ, ⓒ, ⓓ, ⓔ 등)가 원본 이미지의 특수기호와 다를 수 있어서 수정해줘.

      원본 이미지를 보고
      - 문장 앞 기호가 이미지와 정확히 일치하는지 확인하고,
      - 잘못된 기호는 올바른 특수기호로 수정해.
      - 이미지에는 특수기호가 없는데 텍스트에는 있는 경우는 특수기호를 삭제해줘.
      - **추출한 지문의 특수기호 뒤에 괄호'('가 없다면 이미지에서 밑줄로 강조된 부분을 찾아서 괄호로 감싸서 지문을 수정해줘.**

      ⚠️ ①, ②, ③, ④, ⑤ 이런 원형 숫자 특수기호는 지문에 있을 수 없어 반드시 다시 올바른 특수기호로 수정해줘.
      ⚠️ 동일한 특수기호가 또 나올 수 없어 다시 올바른 특수기호로 수정해줘.
      ⚠️ 반드시 특수기호 뒤에는 밑줄로 강조된 부분이 괄호'()'로 반드시 감싸져 있어야해.
      ⚠️ 추출한 지문의 특수기호 뒤에 괄호가 없다면 원본 이미지에서 특수기호 뒤에 다시 밑줄로 강조된 부분을 다시 찾고 지문 텍스트에 괄호'()'를 추가해줘.
      ⚠️ 이미지 기호 위치를 꼭 확인해 판단하고,
      ⚠️ 출력은 반드시 지문 전체를 출력해.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"복원된 지문 일부:\n{restored_text.strip()}"},
                        {"type": "image_url", "image_url": {
                            "url": base64_img,
                            "detail": "high"
                        }}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=16000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[❌ GPT 기호 검토 실패]: {str(e)}"

# 이미지 우측 지문 범위 표시
def insert_passage_brackets_with_gpt(image: Image.Image, ocr_text: str) -> str:
    base64_img = image_to_base64(image)

    system_prompt = """
    너는 수능 국어 문학의 지문 OCR 복원 전문가야.

    다음은 OCR로 추출한 지문과 원본 이미지야.
    원본 이미지의 **좌측 혹은 우측에 [A], [B], [C] 등의 표시와 함께 특정 지문 범위를 나타내는 선들이 있는 경우가 있어.**
    이 시각적 정보는 어떤 지문 구간이 문제 풀이에서 중요한지를 알려주는 단서야.

    🔍 너의 임무는 다음과 같아:

    1. 지문 이미지의 맨 왼쪽과 맨 오른쪽을 잘 살펴보고,
      특정 지문 범위를 나타내는 선과 함께 [A], [B], [C] 등이 지문 내 **어디서부터 어디까지를 가리키는지** 판단해.
        - 각 범위는 반드시 **1줄 이상**이 되도록 하고, **문장이 끊기지 않게 자연스럽게 포함**해야 해.
        - 문장의 시작이나 끝이 잘리지 않도록, 지문 흐름에 맞게 해당 **시작 줄과 끝 줄 전체를 포함**해야 해.

    2. 만약 특정 지문 범위를 나타내는 선이 있다면, 판단된 범위를 지문 중 그대로 유지하되 아래 예시처럼 `[A]{}`로 감싸 강조해줘:
        ...
        [A] {
            (해당 범위 시작 줄)
            ...
            (해당 범위 끝 줄)
        }
        ...

    ⚠️ 유의사항:
    - 줄 단위로 판단하되, 의미 단위(문장 구조)를 최대한 유지해야 해.
    - 특정 지문 범위의 시작과 끝 위치는 이미지 오른쪽의 선과 시각적으로 정렬된 지문 줄을 찾아 판단해야 해.
    - 줄바꿈, 공백, 괄호, 특수기호 등은 OCR 텍스트 스타일을 그대로 유지해.
    """

    try:
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": system_prompt.strip()},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": f"OCR 결과:\n{ocr_text.strip()}"},
                        {"type": "image_url", "image_url": {
                            "url": base64_img,
                            "detail": "high"
                        }}
                    ]
                }
            ],
            temperature=0.1,
            max_tokens=16000
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[❌ GPT 지문 범위 삽입 실패]: {str(e)}"

def run_split_pipeline(image_path: str, parts: int = 4):
    print("📂 이미지 로딩 중...")
    image = Image.open(image_path)

    # 1. 이미지 수직 분할
    print(f"🔀 이미지 {parts}등분 중...")
    split_images = split_image_vertically(image, parts=parts)

    # 2. 각 분할 이미지 OCR
    ocr_parts = []
    for idx, part_img in enumerate(split_images):
        print(f"🔍 [{idx + 1}/{parts}] 분할 이미지 OCR 중...")
        part_text = gpt_ocr_text(part_img)
        ocr_parts.append(part_text)

    # 3. OCR 결과 병합
    combined_ocr = "\n".join(ocr_parts).strip()

    # 4. GPT로 정교화(특수기호&괄호 삽입)
    print("🔧 GPT 정교화 단계 진행 중...")
    refined_text = refine_text_with_gpt(image, combined_ocr)

    # 5. 특수기호 위치 검토 및 보정 => 성공
    print("🧠 특수기호 위치 검토 및 보정 중...")
    symbol_corrected_text = verify_special_symbols(image, refined_text)

    # 6. 이미지 우측 지문 범위 표시
    print("🗂️ GPT 지문 범위 표시([A], [B] 등) 삽입 중...")
    final_result = insert_passage_brackets_with_gpt(image, symbol_corrected_text)

    return final_result.strip()

"""### **문제 텍스트 추출**"""

def extract_question(question_path):
    

    # 파일 확장자에 맞게 MIME 타입 추정
    mime_type, _ = mimetypes.guess_type(question_path)
    if not mime_type:
        mime_type = "image/png"  # 기본값 fallback

    with open(question_path, "rb") as f:
        base64_img = base64.b64encode(f.read()).decode("utf-8")

    image_url = f"data:{mime_type};base64,{base64_img}"

    system_prompt = """
    너는 수능 국어 문학 문제 이미지를 텍스트로 정확하게 복원하는 OCR 모델이야.
    입력 이미지를 보고 **문제의 텍스트를 최대한 원문 그대로 추출**해야 해.

    입력 이미지에는 다음이 포함될 수 있어:
    - 문학 문제의 질문 문장
    - ①~⑤ 보기 선택지
    - 경우에 따라, 질문 문장 **뒤에** 제시되는 '<보기>' 문장

    🔹 반드시 지켜야 할 출력 규칙:

    1. 이미지에 '<보기>'가 존재하면, 질문 문장 **뒤에** <보기> 전체 내용을 정확히 포함해야 해.
    2. <보기>가 없다면, 질문 문장 다음에 바로 선택지를 출력해.
    3. 선택지는 항상 ①~⑤ 모두 빠짐없이 출력할 것.
    4. 출력 형식은 아래와 같아야 해:

    (질문 문장)

    <보기>
    (보기 내용)

    ① ...
    ② ...
    ③ ...
    ④ ...
    ⑤ ...

    ❗주의:
    - <보기>가 있다면 반드시 포함하고, **질문 뒤에 위치**시켜야 해.
    - 줄바꿈, 기호, 문장부호 등은 최대한 원문 그대로 복원해.
    - 절대 설명, 해설, 시스템 메시지 등 **추가 텍스트는 포함하지 마**.
    - 오직 이미지 속 문제 텍스트만 출력해.
    """

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_prompt.strip()},
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_url
                        }
                    }
                ]
            }
        ],
        max_tokens=1500,
        temperature=0.1
    )

    return response.choices[0].message.content.strip()

"""# **<답변,해설 모델>**"""

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# 개념 설명 체인
concept_prompt = ChatPromptTemplate.from_template("""
너는 한국의 대학수학능력시험의 국어 과목 중 문학의 개념에 대해 설명해주는 튜터야. 아래 질문에 대해 간단명료하게 설명해줘.

질문:
{question}
""")

concept_chain = concept_prompt | llm | StrOutputParser()

llm = ChatOpenAI(model="gpt-4o", temperature=0.3)

# <1. 개념 설명 chain>
concept_prompt = ChatPromptTemplate.from_template("""
너는 한국의 대학수학능력시험의 국어 과목 중 문학의 개념에 대해 설명해주는 튜터야. 아래 질문에 대해 간단명료하게 설명해줘.

질문:
{question}
""")
concept_chain = concept_prompt | llm | StrOutputParser()


# 문제 기반 QA 체인용 프롬프트
qa_prompt = ChatPromptTemplate.from_template("""
당신은 한국 수능 국어 문학 전문 튜터입니다.

다음은 수능 국어 문학 객관식 문제입니다. ①, ②, ③, ④, ⑤ 중 하나를 고르는 객관식 문제이며,
정답은 반드시 지문 및 <보기>의 정확한 분석에 근거해 판단해야 합니다.

🟨 반드시 지켜야 할 분석 기준:

1. 지문 분석을 중심으로 판단하며, **선택지의 내용이 지문과 일치하는지** 엄밀히 검토하세요.
2. 문제에 <보기>가 있다면, <보기> 속 설명(구조, 시점, 표현, 인물 해석 등)을 **지문에 어떻게 적용했는지** 구체적으로 분석하세요.
3. **서술 방식, 인물 심리 표현, 문체, 시점 변화, 지시 표현, 병렬 구조** 등은 서술상·표현상 특징 문제에서 핵심 근거입니다.
4. 참고 자료는 반드시 보조적 용도로만 활용합니다. 절대 지문을 덮어쓰거나 대체해서는 안 됩니다.

📌 답변 형식:

[정답]
- (①, ②, ③, ④, ⑤ 중 하나)

[해설]
- 문제에서 요구한 핵심 요소(예: 표현 방식, 구조, 시점 등)에 따라 왜 정답인지 논리적으로 설명합니다.
- 지문 및 <보기>의 문장을 **직접 인용**하여 명확한 판단 근거를 제시하세요.
- 나머지 오답 선택지들은 각각 왜 틀렸는지를 간단히 설명하세요.

📖 [지문]
{context}

📚 [참고 자료] — 필요 시만 사용 (retriever 제공):
{reference}

🙋‍♂️ [문제 및 <보기>]
{question}
""")

def format_with_retrieved_docs(inputs):
    question = inputs["question"]
    context = inputs["context"]

    # 🔍 관련 문서 검색 (context와 question 모두 기준으로 검색)
    retrieved_docs = retriever_literature.get_relevant_documents(f"{context}\n\n{question}")
    retrieved_context = "\n\n".join(doc.page_content for doc in retrieved_docs)

    return {
        "context": context,           # 지문 및 <보기>
        "reference": retrieved_context,  # 보조 자료
        "question": question
    }

# <2. 답변, 해설 QA 체인>: 문항 지문 중심 + retriever에서 검색한 내용 참고
rag_qa_chain = (
    RunnableLambda(format_with_retrieved_docs)
    | qa_prompt
    | llm
    | StrOutputParser()
)



# <문제 vs 개념 분류 함수>
def is_problem_question(question: str) -> bool:
    classification_prompt = ChatPromptTemplate.from_template("""
다음 질문이 문학 개념 질문인지, 지문 기반 문제인지 판별해줘. '개념' 또는 '문제' 중 하나로만 답해.

질문:
{question}
""")
    chain = classification_prompt | llm | StrOutputParser()
    result = chain.invoke({"question": question})
    return "문제" in result.strip()

# <최종 질문 처리 함수>
def tutor_response(question: str, passage: str = None) -> str:
    if is_problem_question(question):
        if not passage:
            print("❗ 오류: 문학 문제 풀이에는 지문(passage)이 필요합니다.")
            return "[❗ 오류: 지문이 없습니다]"

        print("📘 [문제에 대한 정답 및 해설]")
        response = rag_qa_chain.invoke({
            "context": passage,
            "question": question
        })
        return response
    else:
        print("📘 [문학 개념 설명]")
        response = concept_chain.invoke({"question": question})
        return response

"""# **<유사 기출문제 추천 모델>**"""

# <사용자 질문의 GPT 태깅>
def get_tags_from_gpt(query):
    prompt = f"""
            다음 문학 지문과 문제를 읽고 아래 항목을 JSON 형식으로 작성하세요.

            📚 입력 정보
            (지문&문제)
            {query}

            🧩 복합/단일 판단 기준:
            - 지문이 2개 이상이면 반드시 "복합"으로 분류하세요.
            - 문제에서 '공통점', '비교', '다음 글들', '(가)와 (나)'라는 표현이 등장하면 반드시 "복합"으로 분류하세요.
            - 위 조건 중 하나라도 충족하면 반드시 "복합"으로 분류하세요.
            - 지문이 1개이거나 지문 여러개 중에서 문제에서 한 특정 지문만 묻는 경우에만 "단일"로 분류하세요.
            ⚠️ 복합/단일 분류는 절대 틀리면 안 됩니다. 반드시 주의하세요.

            🧠 문제 유형 분류 기준 (지문 장르별로 아래 중 하나 선택):
            - 현대시:
              - 내용 이해: 시적 상황, 주제, 정서 및 태도 등 파악
              - 정서 및 태도 파악: 화자의 심리와 태도 흐름 이해
              - 표현 방식 분석: 비유, 상징, 반복, 설의 등
              - 시어 해석: 개별 시어 또는 표현의 상징/중의적 의미 해석
              - 상징/비유 해석: 중심 이미지나 상징 구조 해석

            - 고전 시가:
              - 내용 이해: 전체 의미, 작품 흐름, 정서 이해
              - 정서 파악: 임에 대한 마음, 자연/현실 인식
              - 표현 기법 분석: 고전적 수사 기법 분석 (영탄, 대조, 과장 등)
              - 임-화자 관계 이해: 교술/서정 시가에서의 관계 맥락
              - 변신/환상 표현 해석: 신화/환상적 요소 해석

            - 현대 소설:
              - 사건 흐름 파악: 줄거리 및 주요 사건 흐름 이해
              - 인물 심리 이해: 인물의 성격, 내적 심리, 관계 해석
              - 시점 및 서술 방식 분석: 서술자, 시점, 묘사 방식 분석
              - 주제/작가 의도 파악: 중심 주제, 주제의식 분석
              - 공간/배경 의미 분석: 배경이 가지는 상징적 의미 해석

            - 고전 소설:
              - 내용 이해: 줄거리, 사건 구조 파악
              - 인물 심리 및 운명 파악: 주요 인물의 성격과 운명
              - 상징 장치 해석: 꿈, 전기, 자연 요소 등 상징 구조 해석
              - 권선징악적 관점 분석: 인과적 세계관, 도덕적 교훈 해석
              - 서사 구조 분석: 도입-전개-위기-절정-결말의 구조

            - 극/수필:
              - 내용 이해: 상황, 대사, 사건의 흐름 이해
              - 표현 특성 분석: 대사, 해설, 장면 구성의 특징
              - 서술자의 개입 파악: 수필/극 중 서술자의 위치 및 역할
              - 주제 및 교훈 도출: 중심 주제 및 삶에 주는 교훈 해석

            ✒️ 지문 제목 및 작가 추출 기준:
            - 일반적으로 각 지문 끝에 다음 형식으로 표기됨:
              `- 김수영 , ｢그 방을 생각하며｣ -`
              → 작가: "김수영", 제목: "그 방을 생각하며"
            - 복합 지문일 경우:
              - "지문 제목": ["제목1", "제목2", ...]
              - "지문 작가": ["작가1", "작가2", ...]
              - 작자 미상일 경우 "작자 미상"으로 표기
              ⚠️ 제목 또는 작가를 찾을 수 없는 경우:
              - 반드시 다음과 같이 작성하세요.
              - 지문 제목: "지문 제목 없음"
              - 지문 작가: "작자 미상"
              - 복합 지문일 경우: ["지문 제목 없음", "지문 제목 없음"], ["작자 미상", "작자 미상"]

            📌 출력 형식 (모두 포함):
            - type: 반드시 "문학"
            - 지문 제목: 문자열 또는 문자열 리스트
            - 지문 장르: 문자열 또는 리스트 (갈래 복합이면 리스트)
            - 지문 작가: 문자열 또는 문자열 리스트
            - 복합/단일: "복합" 또는 "단일"
            - 문제 유형: 위 기준 중 적절한 것 하나

            📌 출력 예시:

            [지문 1개 예시]
            {{
              "type": "문학",
              "지문 제목": "숙향전",
              "지문 장르": "고전 소설",
              "지문 작가": "작자 미상",
              "복합/단일": "단일",
              "문제 유형": "인물 심리 및 운명 파악"
            }}

            [지문 여러개 예시]
            {{
              "type": "문학",
              "지문 제목": ["별사미인곡", "제목 없음", "백자증정부인박씨묘지명"],
              "지문 장르": ["고전 시가", "고전 시가", "고전 산문"],
              "지문 작가": ["김춘택", "이정보", "박지원"],
              "복합/단일": "복합",
              "문제 유형": "정서 파악"
            }}
        """

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "당신은 수능 국어 문학 전문 태깅 도우미입니다. ⚠️ 반드시 코드 블록 없이 순수 JSON만 출력하세요."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=1000
    )

    content = response.choices[0].message.content.strip()
    # GPT가 코드 블록으로 감쌀 경우 제거
    content = re.sub(r"```json\s*([\s\S]+?)\s*```", r"\1", content)
    content = re.sub(r"```[\s\S]+?```", "", content).strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError:
        print("❌ GPT 응답 JSON 파싱 실패:\n", content)
        return None


# 🔸 GPT 클라이언트 설정
client = openai.OpenAI()

# 🔸 문제 코드 추출 함수
def extract_question_code(source_str):
    if not source_str:
        return None
    source_str = source_str.replace("국어", "국어")
    match = re.search(r'(\d{4}-(?:\d{2}|수능)-국어_\d+)$', source_str)
    if match:
        return match.group(1)
    all_matches = re.findall(r'(\d{4}-(?:\d{2}|수능)-국어_\d+)', source_str)
    if all_matches:
        return all_matches[-1]
    return source_str


# 🔸 문서 메타데이터에 태그 병합
def merge_tags_to_docs(docs, tag_dict):
    for doc in docs:
        source = doc.metadata.get("source")
        source = unicodedata.normalize('NFC', source)
        code = extract_question_code(source)
        if code and code in tag_dict:
            doc.metadata["question_tags"] = tag_dict[code]
    print(f"✅ {len(docs)}개의 문서에 태그 병합 완료")
    return docs


# 🔹 태그 유사도 점수 계산 (부분 점수 포함)
def tag_similarity_score(user_tags, doc_tags):
    score = 0

    # 문제 유형 (4점, 완전 일치 시)
    if user_tags.get("문제 유형") == doc_tags.get("문제 유형"):
        score += 4

    # 복합/단일 (2점)
    if user_tags.get("복합/단일") == doc_tags.get("복합/단일"):
        score += 2

    # 지문 장르 (부분 점수 부여)
    user_genre = user_tags.get("지문 장르")
    doc_genre = doc_tags.get("지문 장르")

    if user_genre == doc_genre:
        score += 5
    else:
        # 비슷한 장르 부분 점수
        genre_similarities = [
            ("현대시", "고전 시가"),
            ("고전 시가", "현대시"),
            ("현대 소설", "고전 소설"),
            ("고전 소설", "현대 소설"),
        ]
        if (user_genre, doc_genre) in genre_similarities or (doc_genre, user_genre) in genre_similarities:
            score += 2

    # 지문 제목 (1점)
    user_title = user_tags.get("지문 제목")
    doc_title = doc_tags.get("지문 제목")

    if isinstance(user_title, list) and isinstance(doc_title, list):
        if set(user_title) & set(doc_title):
            score += 1
    elif isinstance(user_title, list):
        if doc_title in user_title:
            score += 1
    elif isinstance(doc_title, list):
        if user_title in doc_title:
            score += 1
    else:
        if user_title == doc_title:
            score += 1

    # 지문 작가 (1점)
    user_name = user_tags.get("지문 작가")
    doc_name = doc_tags.get("지문 작가")

    if isinstance(user_name, list) and isinstance(doc_name, list):
        if set(user_name) & set(doc_name):
            score += 1
    elif isinstance(user_name, list):
        if doc_name in user_name:
            score += 1
    elif isinstance(doc_name, list):
        if user_name in doc_name:
            score += 1
    else:
        if user_name == doc_name:
            score += 1

    return score



# 출처 코드 추출 함수
def extract_question_code(source_str):
    if not source_str:
        return None
    source_str = unicodedata.normalize('NFC', source_str)
    source_str = source_str.replace("국어", "국어").replace("수능", "수능")

    # 출처 코드 정규식 추출 (끝에 있는 코드 우선)
    match = re.search(r'(\d{4}-(?:\d{2}|수능)-국어_\d+)$', source_str)
    if match:
        return match.group(1)
    # 전체 중 마지막 코드 추출
    all_matches = re.findall(r'(\d{4}-(?:\d{2}|수능)-국어_\d+)', source_str)
    return all_matches[-1] if all_matches else None

# 이미지 저장 루트 폴더
IMAGE_ROOT = str(LITERATURE_IMAGE_DIR)

# 문제 이미지 경로 반환
def get_problem_image_path(question_code):
    if not question_code:
        return None
    return os.path.join(IMAGE_ROOT, f"{question_code}.png")

# 지문 이미지 경로 반환
def get_passage_image_path(passage_code):
    if not passage_code:
        return None
    passage_code = unicodedata.normalize("NFC", passage_code)
    match = re.match(r"(\d{4}-(?:\d{2}|수능)-국어)(_p\d+)", passage_code)
    if not match:
        print(f"❌ 지문 코드 파싱 실패: {passage_code}")
        return None
    base, p_part = match.groups()
    return os.path.join(IMAGE_ROOT, f"{base}{p_part}.png")

'''
# 문서 리스트에 태그 병합
def merge_tags_to_docs(docs, tag_dict):
    for doc in docs:
        source = doc.metadata.get("출처", "출처 미상") or doc.metadata.get("source", "출처 미상")
        code = extract_question_code(source)
        if not code:
            print(f"❌ [병합 실패] 출처 코드 추출 실패: {source}")
            continue
        if code not in tag_dict:
            print(f"❌ [병합 실패] 태그 없음: {code}")
            continue
        doc.metadata["question_tags"] = tag_dict[code]
        doc.metadata["출처"] = code
    return docs
'''
def tag_similarity_score(user_tags, doc_tags):
    """
    사용자 태그(user_tags)와 문서 태그(doc_tags)를 비교하여 유사도 점수를 계산하는 함수입니다.

    점수 배점:
    - 문제 유형: 2점 (완전 일치 시)
    - 복합/단일: 2점 (일치 시)
    - 지문 장르: 3점 (완전 일치), 2점 (유사 장르)
    - 지문 제목: 1점 (하나라도 일치 시)
    - 지문 작가: 1점 (하나라도 일치 시)
    """
    score = 0

    # 1) 문제 유형 (2점)
    if user_tags.get("문제 유형") == doc_tags.get("문제 유형"):
        score += 2

    # 2) 복합/단일 (2점)
    if user_tags.get("복합/단일") == doc_tags.get("복합/단일"):
        score += 2

    # 3) 지문 장르 (5점 완전 일치, 2점 유사 장르)
    user_genre = user_tags.get("지문 장르")
    doc_genre = doc_tags.get("지문 장르")

    if user_genre == doc_genre:
        score += 3
    else:
        # 유사 장르 쌍 (부분 점수)
        genre_similarities = [
            ("현대시", "고전 시가"),
            ("고전 시가", "현대시"),
            ("현대 소설", "고전 소설"),
            ("고전 소설", "현대 소설"),
        ]
        if (user_genre, doc_genre) in genre_similarities or (doc_genre, user_genre) in genre_similarities:
            score += 2

    # 4) 지문 제목 (1점, 하나라도 겹치면 점수 부여)
    user_title = user_tags.get("지문 제목")
    doc_title = doc_tags.get("지문 제목")

    if isinstance(user_title, list) and isinstance(doc_title, list):
        if set(user_title) & set(doc_title):  # 교집합이 있으면
            score += 1
    elif isinstance(user_title, list):
        if doc_title in user_title:
            score += 1
    elif isinstance(doc_title, list):
        if user_title in doc_title:
            score += 1
    else:
        if user_title == doc_title:
            score += 1

    # 5) 지문 작가 (1점, 하나라도 겹치면 점수 부여)
    user_name = user_tags.get("지문 작가")
    doc_name = doc_tags.get("지문 작가")

    if isinstance(user_name, list) and isinstance(doc_name, list):
        if set(user_name) & set(doc_name):
            score += 1
    elif isinstance(user_name, list):
        if doc_name in user_name:
            score += 1
    elif isinstance(doc_name, list):
        if user_name in doc_name:
            score += 1
    else:
        if user_name == doc_name:
            score += 1

    return score

def get_similar_problems_with_images(user_question, vectorstore, tag_dict, top_k):
    user_tags = get_tags_from_gpt(user_question)
    print("🟢 get_similar_problems_with_images 실행됨")
    if user_tags is None:
        print("❌ GPT 태그 추출 실패")
        return []

    results = vectorstore.similarity_search_with_score(user_question, k=10)

    docs = []
    for doc, score in results:
        doc.metadata["score"] = score
        docs.append(doc)

    docs = merge_tags_to_docs(docs, tag_dict)

    docs_with_score = []
    for doc in docs:
        doc_tags = doc.metadata.get("question_tags")
        if not doc_tags:
            continue
        tag_sim = tag_similarity_score(user_tags, doc_tags)
        embedding_sim = doc.metadata.get("score", 0)
        final_score = round(tag_sim * 0.7 + embedding_sim * 0.3, 4)
        docs_with_score.append((doc, doc_tags, tag_sim, embedding_sim, final_score))

    docs_sorted = sorted(docs_with_score, key=lambda x: x[4], reverse=True)

    similar_problems = []
    for i, (doc, doc_tags, tag_sim, embedding_sim, final_score) in enumerate(docs_sorted[:top_k]):
        question_code = extract_question_code(doc.metadata.get("source", "출처 미상"))
        passage_code = doc_tags.get("지문", "지문 없음")
        problem_img = get_problem_image_path(question_code)
        passage_img = get_passage_image_path(passage_code)

        similar_problems.append({
            "index": i + 1,
            "question_code": question_code,
            "final_score": final_score,
            "problem_img": problem_img,
            "passage_img": passage_img
        })

    return similar_problems


# 태그 로드 및 딕셔너리 생성
tag_json_path = LITERATURE_DATA / "literature_tagged.json"

with open(tag_json_path, 'r', encoding='utf-8') as f:
    tag_list = json.load(f)

tag_dict_literature = {}
for item in tag_list:
    code = extract_question_code(item.get("출처"))
    if code:
        tag_dict_literature[code] = {
            "출처": item.get("출처"),
            "문제 유형": item.get("문제 유형"),
            "복합/단일": item.get("복합/단일"),
            "지문 제목": item.get("지문 제목"),
            "지문 장르": item.get("지문 장르"),
            "지문 작가": item.get("지문 작가"),
            "지문": item.get("지문"),
        }

"""# **<통합 파이프라인>**
1. 문제 이미지 입력 → OCR로 문제 텍스트 추출

2. 지문 이미지 입력 → OCR 후 구조 보정된 지문 텍스트 생성

3. 정답 + 해설 생성 (RAG 기반)

4. 유사 기출 문제 추천 (임베딩 + 태그 기반)

"""

def pipeline_literature(
    question_image_path: str,
    passage_image_path: str,
    retriever,
    tag_dict,
    show_images: bool = False,
    recommend_top_k: int = 10
):
    """
    🔁 문학 문제 파이프라인: OCR → 답변/해설 → 유사문제 추천

    Parameters:
    - question_image_path: 문제 이미지 경로
    - passage_image_path: 지문 이미지 경로
    - retriever_answer: 답변 벡터스토어 retriever
    - tag_dict: 기출 태그 딕셔너리
    - show_images: Streamlit용 이미지 출력 여부
    - recommend_top_k: 추천할 유사문제 수
    """

    print("🔍 [1단계] 문제 이미지 OCR 추출 중...")
    question_text = extract_question(question_image_path)

    print("\n🔍 [2단계] 지문 이미지 OCR 및 정교화 중...")
    passage_text = run_split_pipeline(passage_image_path)

    print("\n🧠 [3단계] 문제 풀이 및 해설 생성 중...")
    answer_explanation_text = tutor_response(
        question=question_text,
        passage=passage_text
    )
    
    print("\n📂 [4단계] 유사 문제 추천 실행 중...")
    similar_problem_data = get_similar_problems_with_images(
        user_question=question_text,
        vectorstore=vectorstore_literature,
        tag_dict=tag_dict_literature,
        top_k=recommend_top_k
    )

    return {
        "question": question_text,
        "passage": passage_text,
        "response": answer_explanation_text,
        "similar_problems": similar_problem_data
    }