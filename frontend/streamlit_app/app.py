from config import BASE_DIR
import streamlit as st
import tempfile
import os
import sys
from config import LANGMEDIA_IMAGE_DIR
sys.path.extend(str(BASE_DIR))

# 🔹 내부 파이프라인 함수 불러오기
from subject_literature.pipeline_literature import pipeline_literature, retriever_literature, tag_dict_literature # 문학
from subject_non_literature.pipeline_non_literature import pipeline_non_literature # 비문학
from subject_speechcomp.pipeline_speechcomp import analyze_problem # 화법과 작문
from subject_langmedia.pipeline_langmedia import pipeline_langmedia # 언어와 매체

# ✅ Streamlit 설정
st.set_page_config(page_title="STUBO", layout="centered")
st.title("📘 수능 국어 AI 튜터링 시스템")

st.markdown("""
📝 문학 지문 이미지와 문제 이미지를 업로드하면 자동으로
OCR → 정답/해설 생성 → 유사 문제 추천까지 수행합니다.
""")

# 🔹 과목 선택 (비문학, 화법과 작문, 언어와 매체) 
subject = st.radio("📚 분석할 과목을 선택하세요", ["문학", "비문학", "화법과 작문", "언어와 매체"])

# 🔹 이미지 업로드
passage_image = st.file_uploader("1️⃣ 지문 이미지 업로드", type=["png", "jpg", "jpeg"], key="passage")
question_image = st.file_uploader("2️⃣ 문제 이미지 업로드", type=["png", "jpg", "jpeg"], key="question")

# 🔹 유사 문제 추천 개수 선택
top_k = st.slider("🔁 유사 문제 추천 개수", 1, 10, 2)

if st.button("🚀 실행하기"):
    if not question_image:
        st.warning("⚠️ 문제 이미지는 반드시 업로드해주세요.")
    elif subject != "언어와 매체" and not passage_image:
        st.warning("⚠️ 지문 이미지도 함께 업로드해주세요.")
    else:
        # 지문 이미지가 있을 때만 임시 파일 생성
        if passage_image is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_passage:
                tmp_passage.write(passage_image.read())
                passage_img_path = tmp_passage.name
        else:
            passage_img_path = None  # 지문 이미지 없음

        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp_question:
            tmp_question.write(question_image.read())
            question_img_path = tmp_question.name

        # 업로드한 이미지 보여주기 (지문 이미지가 있을 때만)
        if passage_img_path is not None:
            st.image(passage_img_path, caption="📖 업로드한 지문 이미지", use_container_width=True)
        else:
            st.info("📭 지문 이미지가 업로드되지 않았습니다.")

        st.image(question_img_path, caption="❓ 업로드한 문제 이미지", use_container_width=True)

        # 🔸 과목 정보 출력
        st.markdown(f"🧾 선택한 과목: **{subject}**")

        # 🔸 파이프라인 실행
        with st.spinner("🧠 문제를 분석하고 있습니다..."):

        
            ### 문학
            if subject == "문학":
                result = pipeline_literature(
                    question_image_path=question_img_path,
                    passage_image_path=passage_img_path,
                    retriever=retriever_literature,
                    tag_dict=tag_dict_literature,
                    show_images=True,
                    recommend_top_k=top_k
                )
                st.markdown("### 🧠 정답 및 해설")
                st.markdown(result["response"])

                st.markdown("### 🔁 유사 기출문제 추천")
                for problem in result["similar_problems"]:
                    st.markdown(f"#### 📌 유사 문제 {problem['index']}: {problem['question_code']}")

                    if os.path.exists(problem["passage_img"]):
                        st.image(problem["passage_img"], caption="📖 지문 이미지", use_container_width=True)
                    else:
                        st.markdown(f"❌ 지문 이미지 없음: {problem['passage_img']}")

                    if os.path.exists(problem["problem_img"]):
                        st.image(problem["problem_img"], caption="❓ 문제 이미지", use_container_width=True)
                    else:
                        st.markdown(f"❌ 문제 이미지 없음: {problem['problem_img']}")
                        
            ### 비문학 (지문끼리 유사도 계산 -> 지문 1개, 문제 여러개)
            if subject == "비문학":
                response, similar_problems = pipeline_non_literature(
                    passage_img_path,
                    question_img_path,
                    top_k=top_k
                )

                st.markdown("### 🧠 정답 및 해설")
                st.markdown(response)

                st.markdown("### 🔁 유사 기출문제 추천")
                for i, problem in enumerate(similar_problems, 1):
                    st.markdown(f"#### 📌 유사 문제 {i}: {problem['year']}학년도 {problem['month']}월 p{problem['pNum']}")

                    if os.path.exists(problem["passage_img"]):
                        st.image(problem["passage_img"], caption="📖 지문 이미지", use_container_width=True)

                    for qnum, img_path in problem.get("problem_imgs", []):
                        if os.path.exists(img_path):
                            st.image(img_path, caption=f"❓ 문제 {qnum}번", use_container_width=True)


            ### 화법과 작문
            if subject == "화법과 작문":
                result = analyze_problem(
                    context_image_path=passage_img_path,
                    question_image_path=question_img_path,
                    top_k=top_k
                )

                st.markdown("### 🧠 정답 및 해설")
                st.markdown(result["gpt_response"])

                st.markdown("### 🔁 유사 기출문제 추천")

                for i, prob in enumerate(result["similar_problems"], 1):
                    st.markdown(f"#### 📌 유사 문제: {prob['year']}년 {prob['month']}월 (문제 {prob['problem_number']}번)")

                    # 지문 이미지
                    if os.path.exists(prob["passage_img_path"]):
                        st.image(prob["passage_img_path"], caption="📖 지문 이미지", use_container_width=True)
                    else:
                        st.warning(f"❌ 지문 이미지 없음: {prob['passage_img_path']}")

                    # 문제 이미지
                    if os.path.exists(prob["problem_img_path"]):
                        st.image(prob["problem_img_path"], caption="❓ 문제 이미지", use_container_width=True)
                    else:
                        st.warning(f"❌ 문제 이미지 없음: {prob['problem_img_path']}")

                    st.markdown(f"**정답:** {prob.get('answer', '없음')}")
                    st.markdown(f"**해설:** {prob.get('explanation', '없음')}")


            ### 언어와 매체
            if subject == "언어와 매체":
                result = pipeline_langmedia(
                  context_image_path=passage_img_path,
                  question_image_path=question_img_path,
                  top_k=top_k,
                  image_base_dir=str(LANGMEDIA_IMAGE_DIR)  # ← 경로를 반드시 지정
                )

                st.markdown("### 🧠 정답 및 해설")
                st.markdown(f"**정답:** {result['answer']}")
                st.markdown(f"**해설:** {result['explanation']}")

                st.markdown("### 🔁 유사 기출문제 추천")
                for i, prob in enumerate(result["similar_problems"], 1):
                    st.markdown(f"#### 📌 유사 문제 {i}: {prob.get('id_str', '출처 정보 없음')}")

                    for label, path_key in [("📖 지문 이미지", "passage_img_path"), ("❓ 문제 이미지", "problem_img_path")]:
                        path = prob.get(path_key)

                        if path_key == "passage_img_path" and not path:
                            st.info("📭 지문이 없는 문제입니다.")
                        elif isinstance(path, str) and os.path.exists(path):
                            st.image(path, caption=label, use_container_width=True)
                        else:
                            st.warning(f"❌ 이미지 없음: {path}")

        st.success("✅ 완료!")