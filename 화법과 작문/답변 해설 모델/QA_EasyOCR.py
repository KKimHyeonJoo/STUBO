# EasyOCR로 이미지 텍스트 추출 + 원본 이미지까지 해서 모델에 넣어주는 코드

import openai
import os
from glob import glob
import base64
import easyocr  # pip install easyocr

client = openai.OpenAI(api_key="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")  # 여기에 본인의 키 입력

image_dir = "/Users/chaewon/Desktop/STUBO/화법과 작문/output_images"
question_files = sorted(glob(os.path.join(image_dir, "2021-06-화작_42.png")))
context_files = sorted(glob(os.path.join(image_dir, "2021-06-화작_p10.png")))

reader = easyocr.Reader(['ko', 'en'])  # 한글 OCR

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def ocr_image(image_path):
    result = reader.readtext(image_path, detail=0, paragraph=True)
    return "\n".join(result)

print("문제 이미지 리스트:", question_files)
print("지문 이미지 리스트:", context_files)

for q_path, c_path in zip(question_files, context_files):
    print(f"\n\n===== {os.path.basename(q_path)} / {os.path.basename(c_path)} =====")

    # OCR 수행
    context_text = ocr_image(c_path)
    question_text = ocr_image(q_path)

    print("\n[OCR 결과 - 지문 텍스트]\n", context_text)
    print("\n[OCR 결과 - 문제 텍스트]\n", question_text)

    # 이미지 base64 인코딩
    c_b64 = encode_image(c_path)
    q_b64 = encode_image(q_path)

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": (
                        f"다음은 OCR로 추출한 지문 텍스트입니다:\n{context_text}\n\n"
                        f"다음은 OCR로 추출한 문제와 선택지 텍스트입니다:\n{question_text}\n\n"
                        "위 두 이미지와 OCR 텍스트를 참고하여 문제의 정답과 해설을 알려줘. "
                        "정답 번호와 해설을 분리해서 알려줘. (예: 정답: ③)"
                        "모든 선지에 대한 해설을 제공해줘."
                    )},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{c_b64}"}},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{q_b64}"}},
                ]
            }
        ],
        max_tokens=1000,
    )

    print("\n[GPT 응답]\n", response.choices[0].message.content)
