# 나중에 문제 이미지 찾을 때 필요할 것 같아서 년도와 월을 추가하는 코드

import json
import os
import re

# 대상 폴더
BASE_PATH = "/Users/chaewon/Desktop/STUBO/화법과 작문/save_json_with_answers"

# 수정할 대상 파일 패턴: 2021-03-화작_with_answers.json 등
target_files = [
    f for f in os.listdir(BASE_PATH)
    if f.endswith("_with_answers.json") and re.match(r"\d{4}-\w{2}-화작_with_answers.json", f)
]

for filename in target_files:
    match = re.match(r"(\d{4})-(\w{2})-화작_with_answers.json", filename)
    if not match:
        continue
    year, month = int(match.group(1)), match.group(2)

    file_path = os.path.join(BASE_PATH, filename)

    # 파일 열기
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 각 항목에 연/월 추가
    for item in data:
        item["년도"] = year
        item["월"] = month

    # 덮어쓰기 저장
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"연/월 추가 완료: {filename}")
