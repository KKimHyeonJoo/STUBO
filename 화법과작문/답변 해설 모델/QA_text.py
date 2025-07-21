# OCR 없이 텍스트 추출해서 모델에 넣어주는 코드(원본 이미지도 안 넣음)

import os
import json
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# API 키 설정
os.environ["OPENAI_API_KEY"] = "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"

# LLM 세팅
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)

# 프롬프트 템플릿
prompt = PromptTemplate.from_template("""
너는 고등학생 국어 선생님이야.
아래 문제에 대해 답과 해설을 자세히 알려줘.
답변 형식 정답 번호: ,해설: 구조로 맞추어서 해줘.

문제:
{question}

답변:
""")

# LLM 체인 생성
chain = LLMChain(llm=llm, prompt=prompt)

# 4. JSON 파일 경로 지정
BASE_PATH = "/Users/chaewon/Desktop/STUBOa/화법과작문/save_json_with_answers"
filename = "2025-06-화작_with_answers.json"

# 5. JSON 데이터 불러오기
with open(os.path.join(BASE_PATH, filename), "r", encoding="utf-8") as f:
    all_data = json.load(f)

# 6. 각 문제마다 "지문" + "해설" 합쳐서 chain.run() 반복 실행
for i, item in enumerate(all_data, 1):
    question_text = item.get("지문", "") + "\n\n" + item.get("문제", "")
    print(f"\n===== 문제 {i} 답변 및 해설 =====")
    response = chain.run(question=question_text)
    print(response)
    print(f"\n===== 문제 {i} 실제 정답 =====")
    print(item.get("답", ""))
