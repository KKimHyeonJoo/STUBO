import json
import openai
import os
from tqdm import tqdm
import re

# ✅ OpenAI API 키 설정 (환경변수 사용 권장, 여기선 하드코딩)
client = openai.OpenAI(api_key="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")

# 문제 유형 분류 기준 프롬프트
TYPE_GUIDE = """
수능 국어 화법과 작문 문제는 다음의 5가지 유형 중 하나로 분류됩니다:

1. 의사소통 상황 이해형
2. 담화/작문 방식 분석형
3. 자료 활용·적용형
4. 협력적 의사소통형
5. 수정·보완 판단형

다음 문제의 유형을 위 기준에 따라 판단하세요.
⚠️ 출력은 반드시 숫자 하나만! (예: 2)
"""

# GPT 기반 문제 유형 분류 함수
def classify_problem(problem_text):
    prompt = TYPE_GUIDE + "\n\n문제:\n" + problem_text + "\n\n유형 번호:"

    try:
        response = client.chat.completions.create(
            model="gpt-4",
            temperature=0,
            messages=[{"role": "user", "content": prompt}]
        )
        answer = response.choices[0].message.content.strip()
        print(f"[📥 GPT 응답] {answer}")
        print("[DEBUG] 문제:", problem_text[:30], "... GPT 응답:", answer)
    except Exception as e:
        print(f"❌ GPT 요청 오류: {e}")
        return "분류 실패"

    # 정규식으로 숫자 하나 추출
    match = re.search(r"\b([1-5])\b", answer)
    if not match:
        print(f"⚠️ 유형 번호 추출 실패! 응답 내용: {answer}")
        return "분류 실패"

    number = match.group(1)
    type_map = {
        "1": "의사소통 상황 이해형",
        "2": "담화/작문 방식 분석형",
        "3": "자료 활용·적용형",
        "4": "협력적 의사소통형",
        "5": "수정·보완 판단형"
    }
    return type_map.get(number, "분류 실패")

# 문제 텍스트 전처리 함수 (공백 정리 등)
def clean_problem_text(text):
    return re.sub(r"\s+", " ", text.strip())

# JSON 파일 하나 처리 (모든 항목 다시 분류)
def process_file(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    print(f"\n📄 파일 처리 중: {filepath}")
    for item in tqdm(data):
        cleaned = clean_problem_text(item.get("문제", ""))
        item["문제유형"] = classify_problem(cleaned)

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 폴더 안의 모든 화작 JSON 처리
def process_all_files(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith("화작.json"):
            full_path = os.path.join(folder_path, filename)
            process_file(full_path)

# 실행
if __name__ == "__main__":
    process_all_files("save_json")
    print("\n✅ 모든 화작 파일 처리 완료")
