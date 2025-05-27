# preprocessing/extract_qna.py

import fitz  # PyMuPDF
import os

pdf_path = "../data/text.pdf"

def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    full_text = ""
    for page in doc:
        full_text += page.get_text()
    return full_text

if __name__ == "__main__":
    text = extract_text(pdf_path)
    with open("../data/raw_text.txt", "w", encoding="utf-8") as f:
        f.write(text)
    print("✅ 텍스트 추출 완료!")
