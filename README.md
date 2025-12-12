<p align="center">
  <img src="./docs/YoBuddy.png" alt="YoBuddy Architecture" />
</p>

<h1 align="center">🧠 Company FAQ Chatbot</h1>


## 📚 데이터 소스

- **사내 Wiki (DB `wiki` 테이블)** → 자동 FAQ 변환
- **정제된 JSON FAQ들 (`company_FAQ/*.json`)**
  - 회사제도  
  - 인사/복지/출퇴근  
  - IT  
  - 업무툴/협업툴  
  - 조직/부서 정보  
  - 업무 절차/규정  
  - 테스트/기타 등  

---

## 🏗 전체 구성

1. **오프라인 변환 파이프라인 (`convert.py`)**
   - DB `wiki` → FAQ 형태 `(question, answer)` JSON 자동 생성  
   - Gemini 기반 질문 생성 + 요약 FAQ 생성  
   - 변경분만 재생성하는 캐시 구조 (`faq_cache.json`)

2. **FAQ 로딩 & 벡터 인덱스 구축 (`chatbot.py`)**
   - `company_FAQ/*.json` + 카테고리별 JSON 로딩  
   - FAQ 질문 텍스트를 `multilingual-e5` 로 임베딩  
   - `FAISS(IndexFlatIP)` 로 벡터 검색 인덱스 생성  

3. **실시간 질의응답 API (FastAPI)**
   - `POST /api/faq/ask` 엔드포인트  
   - 키워드 매칭 + 벡터 검색 + LLM 후처리  
   - 회사 FAQ에 없는 **일반 상식 질문**도 대응 가능하도록 프롬프트 설계  

---

## 🔄 파이프라인 단계별 설명

#### 1️⃣ 데이터 수집 & 전처리 (`convert.py`)

- `.env` 의 DB 설정 사용  
  (`DB_HOST`, `DB_PORT`, `DB_USER`, `DB_PASSWORD`, `DB_NAME`)
- MySQL / MariaDB 접속 후 `wiki` 테이블에서 `isdeleted = 0` 인 행만 조회
- 사용 컬럼: `id`, `title`, `content`, `updated_at`
- 각 문서 처리
  - 개행(빈 줄) 기준 문단 분리
  - 너무 긴 문단은 문장 단위로 추가 분할
  - → 하나의 wiki가 여러 개의 **“질문 후보 chunk”** 로 나뉘도록 설계

#### 2️⃣ FAQ 질문 자동 생성 (Gemini)

- 모델: `models/gemini-2.5-flash`
- **각 chunk 단위**
  - 실제 사용자가 질문할 법한 **FAQ 스타일 질문 1개** 생성
  - “이 문단/내용/핵심” 같은 어색한 표현이 안 나오도록 프롬프트 제어
- **문서 전체 단위**
  - 문서 전체를 대표하는 질문 1개
  - 해당 내용을 요약한 답변 1개
  - → “이 문서를 한 번에 설명하는 대표 FAQ” 역할

#### 3️⃣ FAQ JSON & 캐시 구조

- 생성된 `(question, answer)` 목록을
  - `company_FAQ/converted_faq_1.json`  
  - `company_FAQ/converted_faq_2.json`  
  - … 형태로 저장
- `faq_cache.json` 에 저장되는 정보
  - 각 `wiki_id` 의 `updated_at`
  - 해당 wiki에서 생성된 FAQ entries
- 동작 방식
  - `updated_at` 이 바뀌지 않은 wiki → 기존 FAQ 그대로 재사용  
  - DB에서 삭제된 wiki → 캐시에서도 제거  
  - → DB 상태와 FAQ 파일 상태를 최대한 동기화

#### 4️⃣ 임베딩 & 벡터 인덱스 구축 (`chatbot.py`)

- 임베딩 모델: `intfloat/multilingual-e5-large-instruct`
- 입력 포맷:
  - `"Instruct: ... Query: {질문}"` 형식으로 prefix 부여  
- 모든 FAQ 질문에 대해:
  - 벡터 생성 후 L2 정규화
  - `FAISS IndexFlatIP` 에 저장
- 결과:
  - 한국어/영어/혼용 질문도 의미 기반 유사도 검색 가능

#### 5️⃣ 실시간 질문 처리 로직

##### (1) 질문 전처리

- 동의어 치환 (synonym dictionary) 예:
  - `"원격근무"` → `"재택근무"`
  - `"CEO"`, `"대표님"`, `"사장"` → `"대표이사"`
- 공백/문장부호 제거, 조사 제거 등으로 검색에 유리한 형태로 정규화

##### (2) 키워드 매칭 (우선 전략)

- 정규화된 사용자 질문 vs FAQ 질문 비교:
  - 완전 일치
  - 포함 관계
  - 단어 교집합 비율 ≥ 70%
- 잘 맞는 항목이 있으면
  - 벡터 검색 없이 바로 해당 FAQ 선택  
  - (표현이 고정된 사내 규정 등에 유리)

##### (3) 벡터 검색 (보조 전략)

- 키워드 매칭 실패 시:
  - 사용자 질문을 임베딩 후 FAISS Top-3 검색
  - 가장 유사도가 높은 질문 선택
  - 유사도가 너무 낮으면  
    - “FAQ에 없는 질문일 수 있음”으로 판단 (디버깅용 로그)

##### (4) LLM 후처리 (최종 답변 생성)

- 선택된 FAQ의 카테고리, 질문, 원문 답변을 컨텍스트로 전달
- Gemini 역할:
  - **회사 FAQ/제도/정책 관련 질문**
    - FAQ를 기반으로 정리해서 답변
  - **회사와 무관한 일반 상식 질문**
    - 1줄 안내:  
      `💡 해당 질문은 회사 FAQ에 포함되지 않은 내용입니다.`
    - 이후 일반 상식 기준으로 설명

#### 6️⃣ API 제공 방식 (FastAPI)

- 엔드포인트: `POST /api/faq/ask`

#### Request 예시

```json
{
  "question": "연차는 어떻게 신청해?"
}
```

#### Response 예시

```json
{
  "answer": "HR 포털(hr.beyondcorp.com)에 로그인 후 '휴가신청' 메뉴에서 신청합니다. 최소 3일 전 신청이 원칙이고, 팀장 승인 후 확정됩니다. 당일 연차는 오전 9시 전까지 Slack으로 팀장님께 먼저 알리고, 사후에 포털에 등록하세요. 연차 잔여일수는 포털 대시보드에서 실시간 확인 가능합니다. 문의: 인사팀 김하늘 매니저 (내선: 2001, hanuel.kim@beyondcorp.com)"
}
```

## 🛠 기술 스택
🏗 Backend & AI
- Python
- FastAPI
- Google Generative AI (Gemini 2.5 Flash)
- SentenceTransformers – intfloat/multilingual-e5-large-instruct
- Pydantic
- Uvicorn

📦 Data & Storage
- MySQL / MariaDB (wiki 테이블)
- JSON FAQ 파일 (company_FAQ/*.json)
- 캐시 파일 (faq_cache.json)
  
🔍 검색 & NLP
- 임베딩 기반 벡터 검색
FAISS (IndexFlatIP + L2 Normalize)
- 키워드 매칭 + 동의어 사전 조합
  
🔧 기타
- .env 기반 환경 변수 관리 (DB_*, GEMINI_API_KEY 등)
- pymysql 로 DB 연동
- 콘솔 모드 챗봇 지원 (로컬 테스트용)
