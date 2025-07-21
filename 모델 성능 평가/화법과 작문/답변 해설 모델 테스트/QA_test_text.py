import os
import json
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage

# API 키 설정
os.environ["OPENAI_API_KEY"] = "  "

# LLM 세팅
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# 4. JSON 파일 경로 지정
BASE_PATH = "/Users/chaewon/Desktop/STUBO/화법과 작문/테스트/답변 해설 모델"
filename = "09_clear_text.json"

# 5. JSON 데이터 불러오기
with open(os.path.join(BASE_PATH, filename), "r", encoding="utf-8") as f:
    all_data = json.load(f)

# 6. 각 문제마다 텍스트만 처리
for i, item in enumerate(all_data, 1):
    question_text = item.get("지문", "") + "\n\n" + item.get("문제", "")
    
    print(f"\n===== 문제 {i+34} 답변 및 해설 =====")
    print(f"문제 번호: {i+34}")
    
    # 텍스트만 처리
    response = llm.invoke(f"""
너는 고등학생 국어 선생님이야.
아래 문제에 대해 답과 해설을 자세히 알려줘.
답변 형식 정답 번호: ,해설: 구조로 맞추어서 해줘.

문제:
{question_text}

답변:
""")
    print(response.content)
    
    print(f"\n===== 문제 {i} 실제 정답 =====")
    print(item.get("답", ""))