import glob
import os
import json
import sys
import numpy as np
import faiss
import google.generativeai as genai
from sentence_transformers import SentenceTransformer

from dotenv import load_dotenv

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.stdout.reconfigure(encoding='utf-8')

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다. .env 파일을 확인하세요.")

genai.configure(api_key=api_key)
model = genai.GenerativeModel("models/gemini-2.5-flash")

embedder = SentenceTransformer('intfloat/multilingual-e5-large-instruct')

BASE_DIR = "company_FAQ"

json_files = {
    "테스트": os.path.join(BASE_DIR, "converted_faq_*.json"),
    "회사제도": os.path.join(BASE_DIR, "company_policy.json"),
    "인사/복지/출퇴근": os.path.join(BASE_DIR, "hr_welfare_attendance.json"),
    "IT": os.path.join(BASE_DIR, "it.json"),
    "업무툴/협업툴": os.path.join(BASE_DIR, "collaboration_tools.json"),
    "조직/부서 정보": os.path.join(BASE_DIR, "organization_department.json"),
    "업무 절차/규정": os.path.join(BASE_DIR, "workflow_policy.json")
}

documents = []
doc_texts = []

for category, path_pattern in json_files.items():
    for path in glob.glob(path_pattern):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                for entry in data:
                    documents.append((category, entry["question"], entry["answer"]))
                    doc_texts.append(entry["question"])

def embed_documents(texts):
    """문서(FAQ 질문) 임베딩"""
    prefixed = [
        f"Instruct: Retrieve semantically similar company FAQ questions\nQuery: {text}"
        for text in texts
    ]
    return embedder.encode(prefixed, convert_to_numpy=True, show_progress_bar=False)

def embed_query(query):
    """사용자 질문 임베딩"""
    instruction = "Instruct: Given a user question, retrieve the most relevant company FAQ question\nQuery: "
    return embedder.encode([instruction + query], convert_to_numpy=True)

print("임베딩 생성 중...")
doc_embeddings = embed_documents(doc_texts)
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatIP(dimension)
faiss.normalize_L2(doc_embeddings)
index.add(doc_embeddings)
print(f"{len(documents)}개 FAQ 로드 완료!\n")

synonyms = {
    "원격근무": "재택근무",
    "연차": "휴가",
    "VPN": "원격 접속",
    "대표님": "대표이사",
    "사장": "대표이사",
    "CEO": "대표이사",
}

def preprocess_question(q):
    """질문 전처리"""
    for key, val in synonyms.items():
        q = q.replace(key, val)
    return q.strip()

def keyword_match(question, documents):
    """키워드 기반 정확 매칭"""
    def normalize(text):
        return text.replace(" ", "").replace("?", "").replace("!", "").replace(".", "").lower()

    q_norm = normalize(question)

    for idx, (cat, q, a) in enumerate(documents):
        q_db_norm = normalize(q)

        if q_norm == q_db_norm or q_norm in q_db_norm or q_db_norm in q_norm:
            return idx, 1.0

        q_words = {w.strip("은는이가을를에서의") for w in q.split() if len(w) >= 2}
        question_words = {w.strip("은는이가을를에서의") for w in question.split() if len(w) >= 2}

        overlap = q_words & question_words

        if len(q_words) >= 2 and len(overlap) >= 2:
            if len(overlap) / len(q_words) >= 0.7:
                return idx, 0.95

    return None, 0.0

def ask_bot(question, debug=False):
    """FAQ 검색 및 답변 생성"""

    question = preprocess_question(question)

    match_idx, match_score = keyword_match(question, documents)

    if match_idx is not None:
        category, matched_q, answer = documents[match_idx]
        if debug:
            print(f"키워드 매칭 성공 ({match_score:.2f}): {matched_q}")
    else:
        q_vec = embed_query(question)
        faiss.normalize_L2(q_vec)

        similarities, indices = index.search(q_vec, 3)

        if debug:
            print("\n벡터 검색 결과:")
            for i in range(3):
                idx = indices[0][i]
                sim = similarities[0][i]
                cat, q, a = documents[idx]
                print(f"  {i+1}. [{sim:.4f}] {q}")

        best_sim = similarities[0][0]
        best_idx = indices[0][0]
        category, matched_q, answer = documents[best_idx]

        if debug:
            print(f"\n선택됨: {matched_q}")

        if best_sim < 0.4:
            if debug:
                print(f"유사도 낮음 ({best_sim:.4f}) - FAQ에 없는 질문일 수 있음")

    prompt = f"""당신은 회사 FAQ 챗봇입니다.

[검색된 FAQ]
카테고리: {category}
질문: {matched_q}
답변: {answer}

[사용자 질문]
{question}

**중요 지침:**
1. 위 FAQ가 사용자 질문과 **회사 업무/정책/제도/위키와 직접 관련**이 있다면 FAQ 내용으로 답변하세요.
2. 하지만 FAQ와 사용자 질문이 **단순히 주제만 비슷**하거나, **회사와 무관한 일반 상식 질문**이라면:
   - "💡 해당 질문은 회사 FAQ에 포함되지 않은 내용입니다." 라고 먼저 말하고
   - 그 다음 줄에 일반 상식으로 답변하세요.
"""

    response = model.generate_content(prompt)
    return response.text.strip()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QuestionRequest(BaseModel):
    question: str

class FaqAnswer(BaseModel):
    answer: str

@app.post("/api/faq/ask", response_model=FaqAnswer)
def api_ask(req: QuestionRequest):
    """프론트/백엔드에서 호출할 HTTP 엔드포인트"""
    answer_text = ask_bot(req.question, debug=False)
    return FaqAnswer(answer=answer_text)

# 4. 콘솔에서 돌리고 싶을 때 (옵션)
def chat_mode():
    print("\n" + "="*70)
    print("FAQ 챗봇 (종료: 'exit' 또는 '종료')")
    print("="*70 + "\n")

    while True:
        question = input("질문: ").strip()

        if question.lower() in ['exit', '종료', 'quit']:
            print("\n 챗봇을 종료합니다.")
            break

        if not question:
            continue

        answer = ask_bot(question, debug=True)
        print(f"\n답변:\n{answer}\n")
        print("-"*70 + "\n")

if __name__ == "__main__":
    chat_mode()