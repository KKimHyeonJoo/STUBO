import os
from pathlib import Path
import ksatparser
import unicodedata

# 연도와 시험 종류 리스트 정의
years = [2022, 2023, 2024, 2025]
exams = ['03', '06', '09', '수능']
subject = '언어와매체'
subject_initial = '언어와매체'

folder = unicodedata.normalize('NFC', f"{subject}_기출문제")
file_paths = []

for year in years:
    for exam in exams:
        filename = f"{year}-{exam}-{subject_initial}.pdf"
        path = f"C:/Users/user/STUBO/{subject}/{folder}/{filename}"
        file_paths.append(path)

output_dir = f"C:/Users/user/STUBO/output_images_{subject_initial}"

for pdf_path in file_paths:
    print(f"확인 중: {pdf_path}")
    if os.path.exists(pdf_path):
        print(f"✅ 존재함: {pdf_path}")
        ksatparser.parse_problem(pdf_path, output_dir)
    else:
        print(f"❌ 파일이 존재하지 않습니다: {pdf_path}")
