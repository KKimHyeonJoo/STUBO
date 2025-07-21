import os
from PyPDF2 import PdfReader
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

# 텍스트 추출 함수
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    full_text = ""
    for page in reader.pages:
        text = page.extract_text()
        if text:
            full_text += text + "\n"
    return full_text

# 각 PDF 파일에 대해 텍스트 추출 및 저장
for file_path in file_paths:
    if os.path.exists(file_path):
        raw_text = extract_text_from_pdf(file_path)
        # txt 파일명 생성
        base_name = os.path.basename(file_path.replace('.pdf', '.txt'))
        save_path = os.path.join(f'/Users/user/STUBO/{subject}/save_text', base_name)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        print(f"텍스트 저장 완료: {save_path}")
    else:
        print(f"파일이 존재하지 않습니다: {file_path}")