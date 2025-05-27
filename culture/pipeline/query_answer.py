# python query_answer.py --query "개인이 속한 사회 집단의 변화가 그 개인에게 미치는 영향은?" 
import os
import faiss
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
from openai import OpenAI  # 변경

openai = OpenAI(api_key = "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")  # OpenAI 클라이언트 객체 생성 (api_key는 환경변수 등으로 설정 권장)

# 1. 로드
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
index = faiss.read_index(os.path.join(root_dir, "pipeline/index.faiss"))
with open(os.path.join(root_dir, "pipeline/questions.pkl"), "rb") as f:
    questions = pickle.load(f)

model = SentenceTransformer("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

# 2. 사용자 질문
query = input("질문을 입력하세요: ")
query_embedding = model.encode([query])[0].reshape(1, -1)

# 3. 유사한 질문 찾기
D, I = index.search(query_embedding, k=1)
matched = questions[I[0][0]]

# 4. 프롬프트 생성
prompt = f"""너는 사회문화 과목 수능특강 문제의 질문-답 챗봇이야.  
다음 문제의 질문을 읽고,  
- 해당 문제에서 요구하는 사회학적 개념이나 이론을 정확히 파악하고,  
- 그 개념을 정의하고 이론적 배경을 간단히 설명하고,  
- 개념이 현실 사회에서 어떻게 적용되는지 구체적인 사례를 들어 설명해.  
- 답변은 수험생이 이해하기 쉽도록 용어 풀이와 예시를 포함해 작성해.  
- 문제 의도와 맞지 않은 개념(예: 다른 연구 방법, 엉뚱한 이론 등)을 사용하지 마.  
질문: ‘의미 구성’과 ‘상황에 따른 새로운 정의’가 어떤 사회학 이론이나 개념과 연결되는지, 그리고 현실 사회에서 어떻게 적용될 수 있는지 구체적인 예를 들어 주세요.
"""

# 5. GPT 호출 (최신 API 방식)
response = openai.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[{"role": "user", "content": prompt}]
)

print("\n💡 GPT의 설명:\n")
print(response.choices[0].message.content)
