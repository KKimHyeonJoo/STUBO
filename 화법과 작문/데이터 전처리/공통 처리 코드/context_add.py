# 지문과 문제 연결하는 코드

import json
import os
import re

# 문제번호 → 이미지 페이지 매핑
page_map = {
    "p9": [35, 36, 37],
    "p10": [38, 39, 40, 41, 42],
    "p11": [43, 44, 45],
}

# 각 문제번호 → p9/p10/p11
problem_to_page = {}
for page, problems in page_map.items():
    for num in problems:
        problem_to_page[num] = page

# 폴더 경로
BASE_PATH = "/Users/chaewon/Desktop/STUBO/화법과 작문/save_json_with_answers"
IMG_DIR = ""

# 대상 파일들
target_files = [
    f for f in os.listdir(BASE_PATH)
    if f.endswith("_with_answers.json") and re.match(r"\d{4}-\w{2}-화작_with_answers.json", f)
]

for filename in target_files:
    match = re.match(r"(\d{4})-(\w{2})-화작_with_answers.json", filename)
    if not match:
        continue

    year, month = match.group(1), match.group(2)
    prefix = f"{year}-{month}-화작"

    json_path = os.path.join(BASE_PATH, filename)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        question_text = item.get("문제", "")
        question_num = None
        if question_text:
            try:
                question_num = int(question_text.split(".")[0])
            except:
                continue
        
        page = problem_to_page.get(question_num)
        if page:
            image_filename = f"{prefix}_{page}.png"
            item["지문"] = os.path.join(IMG_DIR, image_filename)

    # 저장 (덮어쓰기)
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"지문 추가 완료: {filename}")
