# 모의고사 원본 PDF에서 문제 텍스트 추출하는 코드

import os
from PyPDF2 import PdfReader

# 연도와 시험 종류 리스트 정의
years = [2021, 2022, 2023, 2024]
exams = ['03']
subject = '화작'

# PDF 파일 경로 리스트 생성
pdf_paths = []
for year in years:
    for exam in exams:
        filename = f"{year}-{exam}-{subject}.pdf"
        path = f"/Users/chaewon/Desktop/STUBO/화법과 작문/data_a/{filename}"
        pdf_paths.append(path)

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
for pdf_path in pdf_paths:
    if os.path.exists(pdf_path):
        raw_text = extract_text_from_pdf(pdf_path)
        # txt 파일명 생성
        base_name = os.path.basename(pdf_path).replace('.pdf', '.txt')
        save_path = os.path.join('/Users/chaewon/Desktop/STUBO/화법과 작문/save_text', base_name)
        with open(save_path, 'w', encoding='utf-8') as f:
            f.write(raw_text)
        print(f"텍스트 저장 완료: {save_path}")
    else:
        print(f"파일이 존재하지 않습니다: {pdf_path}")
