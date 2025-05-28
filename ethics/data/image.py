import pdfplumber
import os
import re

# 경로 설정
pdf_path = "/Users/chaewon/Desktop/STUBO/ethics/pdf/ethics.pdf"
output_dir = "/Users/chaewon/Desktop/STUBO/ethics/image"
os.makedirs(output_dir, exist_ok=True)

# 문항 코드 → 시작 페이지 저장
question_pages = []  # [(문항번호, 시작페이지)]

with pdfplumber.open(pdf_path) as pdf:
    for i, page in enumerate(pdf.pages):
        text = page.extract_text()
        if not text:
            continue
        matches = re.findall(r"#문항코드\s*(\d{5}-\d{4})", text)
        for match in matches:
            question_pages.append((match, i))  # 예: ("25014-0003", 4)

    # 마지막 문항 → 끝페이지는 전체 페이지로 설정
    question_ranges = []
    for idx, (qid, start) in enumerate(question_pages):
        end = question_pages[idx + 1][1] if idx + 1 < len(question_pages) else len(pdf.pages)
        question_ranges.append((qid, start, end))

    # 각 문항별 범위 내 이미지 탐색
    for qid, start, end in question_ranges:
        found = False
        for page_num in range(start, end):
            page = pdf.pages[page_num]
            for img in page.images:
                x0, top, x1, bottom = img["x0"], img["top"], img["x1"], img["bottom"]
                cropped = page.within_bbox((x0, top, x1, bottom)).to_image(resolution=200)

                filename = f"q_{qid[-3:]}_0.jpeg"
                save_path = os.path.join(output_dir, filename)

                if not found:  # 한 문항에 하나만 저장
                    cropped.save(save_path)
                    print(f"✅ 저장 완료: {save_path}")
                    found = True
