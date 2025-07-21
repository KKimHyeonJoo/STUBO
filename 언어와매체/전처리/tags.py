# 화작 문제 유형 태그 생성 코드
# 문제 저장한 json 파일에 유형 태그까지 추가함

import json
import openai
import os
import time
# OpenAI API 키 설정 (환경변수 사용 권장, 여기선 하드코딩)
client = openai.OpenAI(api_key="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")

## 태그 생성 함수
def generate_tags(question):
    prompt = f"""아래 언어와 매체 문제를 보고
1) 문제 유형
2) 풀이 접근법

두 가지 태그를 한글로 간단히 콤마로 구분해서 알려줘.

문제: {question}


태그:"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "당신은 수능국어 언어와 매체 교육 전문가입니다."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=50,
            temperature=0.3,
        )
        tags_text = response.choices[0].message.content.strip()
        tags = [tag.strip() for tag in tags_text.split(",") if tag.strip()]
        return tags
    except Exception as e:
        print("⚠️ 에러 발생:", e)
        return []

# 메인 처리 함수
def main():
    input_dir = "C:/Users/user/STUBO/언어와매체/save_json"          # 문제 원본 디렉토리
    output_dir = "C:/Users/user/STUBO/언어와매체/save_json_tagged"  # 태그 추가된 문제 저장 디렉토리
    os.makedirs(output_dir, exist_ok=True)

    for filename in os.listdir(input_dir):
        if not filename.endswith(".json"):
            continue

        input_path = os.path.join(input_dir, filename)
        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        print(f"\n📄 [{filename}] 태깅 시작")

        for idx, item in enumerate(data):
            question = item.get("question", "")
            tags = generate_tags(question)
            item["tag"] = tags   # 기존 `"tag": []` 항목에 덮어쓰기
            print(f"  #{idx+1} → {tags}")
            time.sleep(1)  # API 사용량 제한 대비

        output_path = os.path.join(output_dir, filename)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

        print(f"✅ 저장 완료: {output_path}")

if __name__ == "__main__":
    main()
