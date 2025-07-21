import openai
import os
from glob import glob
import base64

client = openai.OpenAI(api_key="  ")  # 여기에 본인의 키 입력

image_dir = "/Users/chaewon/Desktop/STUBO/화법과 작문/output_images"
question_nums = list(range(35, 46))
question_files = []
context_files = []
for num in question_nums:
    q_path = os.path.join(image_dir, f"2024-09-화작_{num}.png")
    if 35 <= num <= 37:
        c_path = os.path.join(image_dir, "2024-09-화작_p9.png")
    elif 38 <= num <= 42:
        c_path = os.path.join(image_dir, "2024-09-화작_p10.png")
    elif 43 <= num <= 45:
        c_path = os.path.join(image_dir, "2024-09-화작_p11.png")
    else:
        continue
    if os.path.exists(q_path) and os.path.exists(c_path):
        question_files.append(q_path)
        context_files.append(c_path)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def extract_text_with_gpt4o(image_path):
    """GPT-4o를 사용하여 이미지에서 텍스트 추출"""
    b64_image = encode_image(image_path)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "이 이미지에서 모든 텍스트를 정확히 추출해줘. 줄바꿈과 공백을 원본 그대로 유지해서 알려줘."},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_image}"}},
                ]
            }
        ],
        max_tokens=16000,
    )
    
    return response.choices[0].message.content

print("문제 이미지 리스트:", question_files)
print("지문 이미지 리스트:", context_files)

for q_path, c_path in zip(question_files, context_files):
    print(f"\n\n===== {os.path.basename(q_path)} / {os.path.basename(c_path)} =====")

    # GPT-4o로 텍스트 추출
    print("지문 텍스트 추출 중...")
    context_text = extract_text_with_gpt4o(c_path)
    print("문제 텍스트 추출 중...")
    question_text = extract_text_with_gpt4o(q_path)

    print("\n[GPT-4o 추출 결과 - 지문 텍스트]\n", context_text)
    print("\n[GPT-4o 추출 결과 - 문제 텍스트]\n", question_text)

    # 추출된 텍스트와 이미지를 함께 GPT-4o에 전송하여 최종 답변 생성
    c_b64 = encode_image(c_path)
    q_b64 = encode_image(q_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"다음은 GPT-4o로 추출한 지문 텍스트입니다:\n{context_text}\n\n"
                        f"다음은 GPT-4o로 추출한 문제와 선택지 텍스트입니다:\n{question_text}\n\n"
                        "위 두 이미지와 추출된 텍스트를 참고하여 문제의 정답과 해설을 알려줘. "
                        "답변 형식 정답 번호: ,해설: 구조로 맞추어서 해줘."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{c_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{q_b64}"}},
                ]
            }
        ],
        max_tokens=16000,
    )

    print("\n[GPT-4o 최종 답변]\n", response.choices[0].message.content)
