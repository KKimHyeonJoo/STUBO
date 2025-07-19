# 최종 json 파일 만들기 위해 문제 항목 덮어쓰기

import json

src_path = "화법과 작문/save_json/2025-06-화작.json"
dst_path = "화법과 작문/save_json_with_answers/2025-06-화작_with_answers.json"

with open(src_path, encoding="utf-8") as f:
    src_data = json.load(f)
with open(dst_path, encoding="utf-8") as f:
    dst_data = json.load(f)

if len(src_data) != len(dst_data):
    print("문항 개수가 다릅니다! 확인해주세요.")
    exit(1)

for s, d in zip(src_data, dst_data):
    d["문제"] = s["문제"]

with open(dst_path, "w", encoding="utf-8") as f:
    json.dump(dst_data, f, ensure_ascii=False, indent=2)

print("문제 항목 덮어쓰기 완료!") 