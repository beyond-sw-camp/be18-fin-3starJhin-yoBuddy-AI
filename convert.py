import os
import re
import json
import time
import pymysql
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path

# --------------------------------------------------
# 0) 기본 설정 (경로 / 환경변수 / Gemini 설정)
# --------------------------------------------------

# 이 파일(convert.py)이 있는 폴더 기준
BASE_DIR = Path(__file__).resolve().parent

# .env 로드
load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY 환경변수가 설정되어 있지 않습니다. .env 파일을 확인하세요.")

GENAI_MODEL_NAME = "models/gemini-2.5-flash"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(GENAI_MODEL_NAME)

# --------------------------------------------------
# 1) DB 설정 (.env에서 읽기)
# --------------------------------------------------

DB_HOST = os.environ["DB_HOST"]
DB_PORT = int(os.environ["DB_PORT"])
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_NAME = os.environ["DB_NAME"]

# 위키가 저장된 테이블/컬럼명 (프로젝트 DB 구조에 맞게 조정)
WIKI_TABLE = "wiki"
WIKI_TITLE_COL = "title"
WIKI_CONTENT_COL = "content"
WIKI_IS_DELETED_COL = "isdeleted"  # 실제 컬럼명에 맞게 사용

# JSON 출력 경로: convert.py가 있는 폴더 기준 company_FAQ 폴더
OUTPUT_DIR = BASE_DIR / "company_FAQ"
BASE_FILE_NAME = "converted_faq"  # converted_faq_1.json, converted_faq_2.json ...
MAX_ITEMS_PER_FILE = 500          # 한 파일당 최대 FAQ 개수


# --------------------------------------------------
# 2) DB에서 위키 내용 가져오기
# --------------------------------------------------

def fetch_wiki_from_db() -> str:
    """
    DB에서 위키들을 읽어와서,
    각 row를 "제목: 내용" 형식의 블록으로 만들어 \n\n 로 이어붙인 텍스트를 반환.
    """
    conn = pymysql.connect(
        host=DB_HOST,
        port=DB_PORT,
        user=DB_USER,
        password=DB_PASSWORD,
        db=DB_NAME,
        charset="utf8mb4",
        cursorclass=pymysql.cursors.DictCursor,
    )

    try:
        with conn.cursor() as cur:
            sql = f"""
            SELECT {WIKI_TITLE_COL}   AS title,
                   {WIKI_CONTENT_COL} AS content
            FROM {WIKI_TABLE}
            WHERE {WIKI_IS_DELETED_COL} = 0
            """
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        conn.close()

    blocks = []
    for row in rows:
        title = row["title"] or ""
        content = row["content"] or ""
        blocks.append(f"{title}: {content}")

    text = "\n\n".join(blocks)
    print(f"DB에서 {len(rows)}개 위키를 읽어왔습니다.")
    return text


# --------------------------------------------------
# 3) 텍스트를 적당한 길이의 chunk로 나누기
# --------------------------------------------------

def split_to_chunks(text: str):
    """
    긴 텍스트를 문단/문장 단위로 잘라서 chunk 리스트로 반환.
    """
    # 1) 문단 기준 분리
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []

    for para in paragraphs:
        # 너무 길면 문장 단위로 재분할
        if len(para) > 300:
            sentences = re.split(r'(?<=\.)\s+', para)
            buffer = ""

            for sent in sentences:
                if len(buffer) + len(sent) < 250:
                    buffer += sent + " "
                else:
                    chunks.append(buffer.strip())
                    buffer = sent + " "

            if buffer.strip():
                chunks.append(buffer.strip())
        else:
            chunks.append(para)

    print(f"총 {len(chunks)}개의 chunk로 분리되었습니다.")
    return chunks


# --------------------------------------------------
# 4) Gemini로 자연스러운 질문 생성
# --------------------------------------------------

def generate_question_with_gemini(chunk: str) -> str:
    """
    chunk 내용을 기반으로, 실제 사람이 물어볼 만한 FAQ 질문 한 문장을 Gemini에게 생성 요청.
    """
    prompt = f"""
아래 내용을 읽고, 실제 사용자가 이 내용을 질문하려고 할 때 자연스럽게 물어볼 'FAQ 스타일 질문'을 한 문장으로 만들어줘.

내용:
\"\"\"{chunk}\"\"\"


질문 생성 규칙:
- 자연스러운 질문일 것
- "이 문단", "내용", "핵심" 같은 단어 절대 사용 금지
- 가게, 제도, 설명 등의 대상에 맞춰 실제 사람이 묻는 방식으로 작성
- 예시와 유사한 톤을 사용할 것:
  - 회사 근처에 어떤 맛집이 있나요?
  - 김밥천국은 어떤 곳인가요?
  - 이 가게의 특징은 무엇인가요?
  - 어떤 서비스를 제공하나요?
- 답변에 직접 등장하는 대상(가게명, 개념명)을 사용해 질문을 구성할 것

출력 형식:
- 질문 문장만 출력
"""

    response = model.generate_content(prompt)
    question = (response.text or "").strip()
    question = question.replace("질문:", "").strip()

    # 너무 길면 조금 잘라주기
    if len(question) > 120:
        question = question[:120] + "..."

    return question


# --------------------------------------------------
# 5) chunks → (Q, A) entry 리스트 만들기
# --------------------------------------------------

def build_entries_from_text(text: str):
    """
    전체 텍스트에서 chunk를 만들고,
    각 chunk에 대해 (question, answer) entry를 생성한 리스트를 반환.
    """
    chunks = split_to_chunks(text)
    entries = []

    for idx, chunk in enumerate(chunks, start=1):
        print(f"\n▶ [{idx}/{len(chunks)}] Gemini 질문 생성 중…")

        # 무료 티어: 분당 10회 제한 → 호출 간격을 넉넉하게 벌려줌
        if idx > 1:
            time.sleep(7)  # 두 번째 호출부터 7초 쉬고 호출

        q = generate_question_with_gemini(chunk)

        entries.append({
            "question": q,
            "answer": chunk,  # 일단은 chunk 전체를 answer로 사용
        })

    print(f"\n총 {len(entries)}개의 FAQ 엔트리가 생성되었습니다.")
    return entries


# --------------------------------------------------
# 6) JSON 저장 (여러 파일로 나누기)
# --------------------------------------------------

def save_as_multi_json(entries):
    """
    entries 리스트를 MAX_ITEMS_PER_FILE 기준으로 잘라
    converted_faq_1.json, converted_faq_2.json ... 형태로 저장.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    total = len(entries)
    if total == 0:
        print("저장할 entry가 없습니다.")
        return

    file_index = 1
    for i in range(0, total, MAX_ITEMS_PER_FILE):
        slice_entries = entries[i: i + MAX_ITEMS_PER_FILE]
        file_path = OUTPUT_DIR / f"{BASE_FILE_NAME}_{file_index}.json"

        with file_path.open("w", encoding="utf-8") as f:
            json.dump(slice_entries, f, indent=2, ensure_ascii=False)

        print(f" {len(slice_entries)}개를 {file_path} 에 저장했습니다.")
        file_index += 1

    print(f"\nJSON 파일 저장 완료! (총 {file_index - 1}개 파일)")

    # 🔍 디버깅용: 실제로 어떤 경로/파일을 보고 있는지 출력
    print("\n[DEBUG] 현재 작업 디렉토리:", os.getcwd())
    print("[DEBUG] OUTPUT_DIR:", OUTPUT_DIR)
    try:
        print("[DEBUG] OUTPUT_DIR 안의 파일들:", [p.name for p in OUTPUT_DIR.iterdir()])
    except FileNotFoundError:
        print("[DEBUG] OUTPUT_DIR 가 존재하지 않습니다.")


# --------------------------------------------------
# 7) 메인 실행
# --------------------------------------------------

def export_from_db_to_multi_json():
    # 1) DB에서 위키 읽어오기
    text = fetch_wiki_from_db()

    if not text.strip():
        print("DB에서 가져온 내용이 비어 있습니다. 종료합니다.")
        return

    # 2) (Q, A) 엔트리 생성
    entries = build_entries_from_text(text)

    # 3) 여러 JSON 파일로 나누어 저장
    save_as_multi_json(entries)


if __name__ == "__main__":
    export_from_db_to_multi_json()
