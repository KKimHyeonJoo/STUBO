# 모의고사 원본 PDF에서 문제 이미지 추출하는 코드
# 화법과 작문은 35번~45번, 총 11문제
# 지문은 9번~11번, 총 3개

import os
from pathlib import Path
import ksatparser

# 연도와 시험 종류 리스트 정의
years = [2022]
exams = ['03']
subject = '화작'

file_paths = []
for year in years:
    for exam in exams:
        filename = f"{year}-{exam}-{subject}.pdf"
        path = f"/Users/chaewon/Desktop/STUBO/화법과 작문/data_a/{filename}"
        file_paths.append(path)

output_dir = "/Users/chaewon/Desktop/STUBO/화법과 작문/output_images"

for pdf_path in file_paths:
    if os.path.exists(pdf_path):
        ksatparser.parse_problem(pdf_path, output_dir)
    else:
        print(f"파일이 존재하지 않습니다: {pdf_path}")