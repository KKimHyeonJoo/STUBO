# 3월 모의고사 json 파일에 답과 해설 채우는 코드

import json
import os

YEARS = [2021, 2022, 2023, 2024]
MONTH = "03"
BASE_PATH = "/Users/chaewon/Desktop/STUBO/화법과 작문"
ANSWER_TEXT_DIR = os.path.join(BASE_PATH, "answer_text")

for year in YEARS:
    json_filename = f"{year}-{MONTH}-화작.json"
    answer_text_filename = f"{year}-{MONTH}-화작 해설.txt"
    json_path = os.path.join(BASE_PATH, "save_json", json_filename)
    answer_text_path = os.path.join(ANSWER_TEXT_DIR, answer_text_filename)

    if not os.path.exists(json_path):
        print(f"문제 JSON 파일 없음: {json_path}")
        continue
    if not os.path.exists(answer_text_path):
        print(f"해설 텍스트 파일 없음: {answer_text_path}")
        continue

    # 1. 문제 JSON 읽기
    with open(json_path, "r", encoding="utf-8") as f:
        save_data = json.load(f)

    # 2. 해설 텍스트(JSON 배열) 읽기
    with open(answer_text_path, "r", encoding="utf-8") as f:
        answer_data = json.load(f)

    # 3. 문제번호로 딕셔너리 생성
    answer_dict = {item["문제번호"]: item for item in answer_data}

    # 4. 문제별 정답과 해설 넣기
    for item in save_data:
        question_text = item.get("문제", "")
        question_num = None
        if question_text:
            try:
                question_num = int(question_text.split(".")[0])
            except:
                pass

        if question_num and question_num in answer_dict:
            item["답"] = answer_dict[question_num]["정답"]
            item["해설"] = answer_dict[question_num]["해설"]

    # 5. 저장 (원본명_with_answers.json)
    save_path = os.path.join(BASE_PATH,"save_json_with_answers", json_filename.replace(".json", "_with_answers.json"))
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"완료: {save_path}")
