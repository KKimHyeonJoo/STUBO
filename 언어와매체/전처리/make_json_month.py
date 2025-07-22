import os
import json

BASE_PATH = "C:/Users/user/STUBO/언어와매체/save_json_tagged"

for filename in os.listdir(BASE_PATH):
    if filename.endswith(".json"):
        file_path = os.path.join(BASE_PATH, filename)

        try:
            parts = filename.split("-")
            year = int(parts[0])
            raw_month = parts[1]

            # 회차 처리: 수능은 문자열, 나머지는 숫자
            if raw_month == "수능":
                month_name = "수능"
            else:
                month_name = int(raw_month)  # 03 → 3 등

        except Exception as e:
            print(f"⚠️ 파일명에서 '년', '회차' 추출 실패: {filename}")
            continue

        # 데이터 로드
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 각 문항에 '년', '회차' 추가
        for item in data:
            item["년"] = year
            item["월"] = month_name

        # 덮어쓰기 저장
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"✅ 저장 완료: {filename}")
