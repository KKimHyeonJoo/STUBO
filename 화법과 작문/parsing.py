import os
from pathlib import Path
import ksatparser

# 연도와 시험 종류 리스트 정의
years = [2021, 2022, 2023, 2024]
exams = ['06', '09', '수능']
subject = '화작'

file_paths = []
for year in years:
    for exam in exams:
        filename = f"{year}-{exam}-{subject}.pdf"
        path = f"/Users/chaewon/Desktop/화법과 작문/data/{filename}"
        file_paths.append(path)

output_dir = "/Users/chaewon/Desktop/화법과 작문/output_images"

for pdf_path in file_paths:
    if os.path.exists(pdf_path):
        ksatparser.parse_problem(pdf_path, output_dir)
    else:
        print(f"파일이 존재하지 않습니다: {pdf_path}")