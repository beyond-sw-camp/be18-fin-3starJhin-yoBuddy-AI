import os
import re
import json
import time
import pymysql
import google.generativeai as genai
from dotenv import load_dotenv
from pathlib import Path
from datetime import datetime

BASE_DIR = Path(__file__).resolve().parent

load_dotenv(BASE_DIR / ".env")

API_KEY = os.getenv("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("GEMINI_API_KEY가 설정되어 있지 않습니다. .env 파일을 확인하세요.")

GENAI_MODEL_NAME = "models/gemini-2.5-flash"
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(GENAI_MODEL_NAME)

DB_HOST = os.environ["DB_HOST"]
DB_PORT = int(os.environ["DB_PORT"])
DB_USER = os.environ["DB_USER"]
DB_PASSWORD = os.environ["DB_PASSWORD"]
DB_NAME = os.environ["DB_NAME"]

WIKI_TABLE = "wiki"
WIKI_ID_COL = "wiki_id"         
WIKI_TITLE_COL = "title"
WIKI_CONTENT_COL = "content"
WIKI_UPDATED_AT_COL = "update_at"
WIKI_IS_DELETED_COL = "isdeleted"

OUTPUT_DIR = BASE_DIR / "company_FAQ"
BASE_FILE_NAME = "converted_faq"
MAX_ITEMS_PER_FILE = 500

CACHE_PATH = OUTPUT_DIR / "faq_cache.json"


def fetch_wiki_rows():
    """
    DB에서 (id, title, content, updated_at)을 읽어와 리스트로 반환.
    isdeleted = 0 인 것만.
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
            SELECT
                {WIKI_ID_COL}          AS id,
                {WIKI_TITLE_COL}       AS title,
                {WIKI_CONTENT_COL}     AS content,
                {WIKI_UPDATED_AT_COL}  AS updated_at
            FROM {WIKI_TABLE}
            WHERE {WIKI_IS_DELETED_COL} = 0
            """
            cur.execute(sql)
            rows = cur.fetchall()
    finally:
        conn.close()

    print(f"DB에서 {len(rows)}개 위키를 읽어왔습니다.")
    return rows


def split_to_chunks(text: str):
    """
    긴 텍스트를 문단/문장 단위로 잘라서 chunk 리스트로 반환.
    한 wiki row 안에서 여러 FAQ를 뽑고 싶을 때 사용.
    """
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    chunks = []

    for para in paragraphs:
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

    print(f"    - 이 위키에서 {len(chunks)}개 chunk로 분리됨")
    return chunks


def generate_question_with_gemini(chunk: str) -> str:
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

    if len(question) > 120:
        question = question[:120] + "..."

    return question

def generate_overall_faq(text: str):
    """
    text 전체를 대표하는 '요약형 FAQ' (질문+답변) 1개 생성
    """
    prompt = f"""
아래 내용을 하나의 위키 문서라고 생각하고,
이 전체 내용을 자연스럽게 묻는 FAQ 스타일 질문 1개와,
그에 대한 요약 답변 1개를 만들어줘.

내용:
\"\"\"{text}\"\"\"

작성 규칙:
- 질문은 이 문서 전체를 대표하는 질문 1개
  예) "회사 근처에는 어떤 맛집이 있나요?"
- 답변은 핵심 내용들을 한 번에 요약해서 정리
- 문체는 회사 FAQ 느낌의 존댓말로 작성

출력 형식 (반드시 이 형식 유지):
질문: ...
답변: ...
"""
    response = model.generate_content(prompt)
    raw = (response.text or "").strip()

    q = ""
    a = ""

    for line in raw.splitlines():
        line = line.strip()
        if line.startswith("질문:"):
            q = line.replace("질문:", "").strip()
        elif line.startswith("답변:"):
            a = line.replace("답변:", "").strip()

    if not q:
        q = "이 위키의 전체 내용을 한 번에 요약하면 어떻게 되나요?"
    if not a:
        a = text[:200] + "..."

    return q, a

def load_cache():
    if CACHE_PATH.exists():
        with CACHE_PATH.open("r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(cache: dict):
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CACHE_PATH.open("w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, ensure_ascii=False)

def build_entries_with_cache(rows):
    """
    rows: fetch_wiki_rows() 결과
    cache 구조:
    {
      "123": {
        "updated_at": "...",
        "entries": [
          {"question": "...", "answer": "..."},
          ...
        ]
      },
      ...
    }
    """
    cache = load_cache()
    entries = []
    current_ids = set()

    for idx, row in enumerate(rows, start=1):
        wiki_id = str(row["id"])
        title = row["title"] or ""
        content = row["content"] or ""
        updated_at = row["updated_at"]

        if isinstance(updated_at, datetime):
            updated_at_str = updated_at.isoformat()
        else:
            updated_at_str = str(updated_at)

        current_ids.add(wiki_id)

        full_text = f"{title}: {content}"

        if wiki_id in cache and cache[wiki_id].get("updated_at") == updated_at_str:
            print(f"[{idx}] 캐시 재사용: {title}")
            wiki_entries = cache[wiki_id]["entries"]
        else:
            print(f"[{idx}] 새로 생성: {title}")

            chunks = split_to_chunks(full_text)

            wiki_entries = []

            for c_idx, chunk in enumerate(chunks, start=1):
                print(f"    - chunk {c_idx}/{len(chunks)} 질문 생성 중...")
                if c_idx > 1:
                    time.sleep(7)

                q = generate_question_with_gemini(chunk)
                wiki_entries.append({
                    "question": q,
                    "answer": chunk
                })

            print("    - 전체 요약 FAQ 생성 중...")
            overall_q, overall_a = generate_overall_faq(full_text)
            wiki_entries.append({
                "question": overall_q,
                "answer": overall_a
            })

            cache[wiki_id] = {
                "updated_at": updated_at_str,
                "entries": wiki_entries
            }

        entries.extend(wiki_entries)

    removed_ids = [wid for wid in cache.keys() if wid not in current_ids]
    for wid in removed_ids:
        del cache[wid]
    if removed_ids:
        print(f"캐시에서 삭제된 wiki {len(removed_ids)}개 정리")

    save_cache(cache)

    print(f"\n총 {len(entries)}개의 FAQ 엔트리가 생성되었습니다.")
    return entries


def save_as_multi_json(entries):
    """
    entries 리스트를 MAX_ITEMS_PER_FILE 기준으로 잘라
    converted_faq_1.json, converted_faq_2.json ... 형태로 저장.
    기존 converted_faq_* 파일은 먼저 삭제.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    old_files = list(OUTPUT_DIR.glob(f"{BASE_FILE_NAME}_*.json"))
    for f in old_files:
        f.unlink()
    if old_files:
        print(f"기존 FAQ JSON {len(old_files)}개 삭제")

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

        print(f"{len(slice_entries)}개를 {file_path} 에 저장했습니다.")
        file_index += 1

    print(f"\nJSON 파일 저장 완료! (총 {file_index - 1}개 파일)")

def export_from_db_to_multi_json():
    rows = fetch_wiki_rows()
    if not rows:
        print("DB에서 가져온 내용이 비어 있습니다. 종료합니다.")
        return

    entries = build_entries_with_cache(rows)
    save_as_multi_json(entries)


if __name__ == "__main__":
    export_from_db_to_multi_json()
