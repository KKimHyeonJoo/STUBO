import os
import json
import base64
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.schema import HumanMessage

# API 키 설정
os.environ["OPENAI_API_KEY"] = "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"

# LLM 세팅 (이미지 처리 가능한 모델 사용)
llm = ChatOpenAI(model="gpt-4o", temperature=0.1)

# 이미지를 base64로 인코딩하는 함수
def encode_image_to_base64(image_path):
    """이미지 파일을 base64로 인코딩"""
    try:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    except FileNotFoundError:
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return None

# 이미지가 있는지 확인하는 함수
def get_image_path(image_filename, base_path):
    """이미지 파일 경로를 찾아서 반환"""
    # 절대 경로로 output_images 폴더 지정
    image_dir = "/Users/chaewon/Desktop/STUBO/화법과 작문/output_images"
    image_path = os.path.join(image_dir, image_filename)
    
    if os.path.exists(image_path):
        return image_path
    else:
        print(f"이미지 파일을 찾을 수 없습니다: {image_path}")
        return None

# 4. JSON 파일 경로 지정
BASE_PATH = "/Users/chaewon/Desktop/STUBO/화법과 작문/테스트/답변 해설 모델"
filename = "06_clear_text.json"

# 5. JSON 데이터 불러오기
with open(os.path.join(BASE_PATH, filename), "r", encoding="utf-8") as f:
    all_data = json.load(f)

# 6. 각 문제마다 지문 이미지 + 문제 이미지 + 텍스트를 함께 처리
for i, (key, item) in enumerate(all_data.items(), 1):
    question_text = item.get("context", "") + "\n\n" + item.get("question", "")
    
    # 문제 이미지 파일명 추출 (키에서 추출)
    problem_image_filename = key
    
    # 지문 이미지 파일명 추출 (문제 번호에서 지문 번호 추출)
    # 예: 2025-06-화작_35.png -> 2025-06-화작_p9.png (35번 문제는 9번 지문)
    problem_number = key.split('_')[-1].replace('.png', '')
    
    # 문제 번호에 따른 지문 번호 매핑 (예시)
    passage_mapping = {
        '35': 'p9', '36': 'p9', '37': 'p9',  # 35-37번은 9번 지문
        '38': 'p10', '39': 'p10', '40': 'p10', '41': 'p10', '42': 'p10',  # 38-42번은 10번 지문
        '43': 'p11', '44': 'p11', '45': 'p11',  # 43-45번은 11번 지문
    }
    
    passage_number = passage_mapping.get(problem_number, 'p9')  # 기본값은 p9
    passage_image_filename = key.replace(f'_{problem_number}.png', f'_{passage_number}.png')
    
    # 이미지 경로 찾기
    problem_image_path = get_image_path(problem_image_filename, BASE_PATH)
    passage_image_path = get_image_path(passage_image_filename, BASE_PATH)
    
    print(f"\n===== 문제 {i} 답변 및 해설 =====")
    print(f"문제 이미지: {problem_image_filename}")
    print(f"지문 이미지: {passage_image_filename}")
    
    if problem_image_path and passage_image_path:
        print(f"문제 이미지 경로: {problem_image_path}")
        print(f"지문 이미지 경로: {passage_image_path}")
        
        # 두 이미지가 모두 있는 경우 이미지와 텍스트를 함께 처리
        try:
            # 두 이미지를 base64로 인코딩
            problem_base64_image = encode_image_to_base64(problem_image_path)
            passage_base64_image = encode_image_to_base64(passage_image_path)
            
            if problem_base64_image and passage_base64_image:
                print(f"문제 이미지 인코딩 성공: {len(problem_base64_image)} 문자")
                print(f"지문 이미지 인코딩 성공: {len(passage_base64_image)} 문자")
                
                # 지문 이미지 + 문제 이미지 + 텍스트를 함께 전송
                messages = [
                    HumanMessage(
                        content=[
                            {
                                "type": "text",
                                "text": f"""
너는 고등학생 국어 선생님이야.
아래 문제에 대해 답과 해설을 자세히 알려줘.
답변 형식 정답 번호: ,해설: 구조로 맞추어서 해줘.

문제:
{question_text}

답변:
"""
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{passage_base64_image}"
                                }
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{problem_base64_image}"
                                }
                            }
                        ]
                    )
                ]
                
                print("LLM에 지문 이미지 + 문제 이미지 + 텍스트 전송 중...")
                response = llm.invoke(messages)
                print(response.content)
            else:
                # 이미지 인코딩 실패 시 텍스트만 처리
                print("이미지 인코딩 실패, 텍스트만 처리합니다.")
                response = llm.invoke(f"""
너는 고등학생 국어 선생님이야.
아래 문제에 대해 답과 해설을 자세히 알려줘.
답변 형식 정답 번호: ,해설: 구조로 맞추어서 해줘.

문제:
{question_text}

답변:
""")
                print(response.content)
                
        except Exception as e:
            print(f"이미지 처리 중 오류 발생: {e}")
            # 오류 발생 시 텍스트만 처리
            response = llm.invoke(f"""
너는 고등학생 국어 선생님이야.
아래 문제에 대해 답과 해설을 자세히 알려줘.
답변 형식 정답 번호: ,해설: 구조로 맞추어서 해줘.

문제:
{question_text}

답변:
""")
            print(response.content)
    else:
        print(f"이미지 파일을 찾을 수 없습니다:")
        if not problem_image_path:
            print(f"  - 문제 이미지: {problem_image_filename}")
        if not passage_image_path:
            print(f"  - 지문 이미지: {passage_image_filename}")
        
        # 이미지가 없는 경우 텍스트만 처리
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
