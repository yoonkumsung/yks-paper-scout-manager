# Paper Scout

arXiv에서 매일 쏟아지는 논문 중 내 연구와 관련된 핵심 논문만 골라 한국어 리포트로 만들어주는 **AI 논문 큐레이션 시스템**.

- AI가 논문을 읽고 관련성을 0~100점으로 평가
- 한국어로 핵심 기여, 방법론, 실험 결과를 요약
- GitHub Actions로 매일 자동 실행
- 리마인드 시스템으로 고점수 논문 재노출
- 무료 LLM 모델 사용 (OpenRouter)

---

## 설치 및 실행

```bash
git clone https://github.com/your-username/yks-paper-collector.git
cd yks-paper-collector
pip install -r requirements.txt
```

[OpenRouter](https://openrouter.ai/)에서 API 키를 발급받고 `.env`에 설정:

```bash
OPENROUTER_API_KEY=your-key

# 알림 (선택)
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TELEGRAM_BOT_TOKEN_YKS=your-bot-token
TELEGRAM_CHAT_ID_YKS=your-chat-id
```

실행:

```bash
python main.py          # 웹 UI 실행 (http://127.0.0.1:8585)
python main.py run      # CLI로 전체 파이프라인 실행
```

---

## CLI 사용법

```bash
# 파이프라인
python main.py run                          # 전체 파이프라인
python main.py run --topic my-topic         # 특정 토픽만
python main.py run --date-from 2025-01-01   # 날짜 지정
python main.py dry-run                      # 키워드 생성만 (논문 수집 안 함)

# 토픽 관리
python main.py topic list                   # 토픽 목록
python main.py topic add                    # 토픽 추가 (대화형)
python main.py topic edit <slug>            # 토픽 편집 ($EDITOR)
python main.py topic remove <slug>          # 토픽 삭제

# 웹 UI
python main.py ui                           # 웹 UI 실행
python main.py ui --port 9090               # 포트 지정
python main.py ui --no-browser              # 브라우저 자동 열기 안 함
```

공통 옵션:

| 옵션 | 설명 | 기본값 |
|------|------|--------|
| `--date-from` | 검색 시작일 (YYYY-MM-DD) | 마지막 성공 시점 |
| `--date-to` | 검색 종료일 (YYYY-MM-DD) | 오늘 |
| `--dedup` | `skip_recent` / `none` | `skip_recent` |
| `--topic` | 특정 토픽만 실행 | 전체 |
| `--mode` | `full` / `dry-run` | `full` |

---

## 웹 UI

`python main.py` (인자 없이) 또는 `python main.py ui`로 실행.

- 초기 설정 마법사 (API 키, 알림 설정)
- 토픽 추가 및 AI 기반 arXiv 카테고리 추천
- 키워드 자동 생성 (드라이런) 및 수동 편집
- 파이프라인 실행 (SSE 실시간 로그)
- GitHub Actions 설정 동기화

---

## GitHub Actions 자동 실행

매일 **UTC 02:00 (KST 11:00)**에 자동 실행됩니다.

### 설정

1. GitHub에 리포지토리 포크
2. Repository Settings > Actions > General > **Read and write permissions** 활성화
3. GitHub Secrets 설정:
   - `OPENROUTER_API_KEY`
   - `DISCORD_WEBHOOK_URL` / `TELEGRAM_BOT_TOKEN` / `TELEGRAM_CHAT_ID` (선택)
   - `SUPABASE_DB_URL` (Supabase DB 사용 시)
4. 웹 UI의 **"GitHub Actions 동기화"** 버튼으로 로컬 설정 반영

동기화 시 업로드되는 Secrets:

| 로컬 파일 | GitHub Secret |
|----------|---------------|
| `config.yaml` | `PAPER_SCOUT_CONFIG` |
| `data/keyword_cache.json` | `PAPER_SCOUT_KEYWORD_CACHE` |
| `data/last_success.json` | `PAPER_SCOUT_LAST_SUCCESS` |
| `data/model_caps.json` | `PAPER_SCOUT_MODEL_CAPS` |

### 수동 실행

GitHub Actions 탭 > "Run workflow" 클릭. `date_from`, `date_to`, `mode`, `dedup` 파라미터 지정 가능.

### 날짜 동작

| 상황 | 검색 범위 |
|------|----------|
| 자동/수동 (날짜 미지정) | 마지막 성공 ~ 오늘 |
| 수동 (날짜 지정) | 지정한 범위 |
| 첫 실행 (이력 없음) | 최근 3일 |

---

## 아키텍처

### 파이프라인

```
Preflight → Topic Loop (토픽별) → Post Loop
```

**Preflight**: 설정 검증, API 키 확인, 검색 윈도우 계산

**Topic Loop** (각 토픽):
1. Agent 1 (Keyword Expander): 한국어 설명 → 영어 검색 키워드 (캐시 지원)
2. arXiv API 논문 수집
3. 3단계 하이브리드 필터 (규칙 → 방어캡 → embedding/최신순)
4. Agent 2 (Scorer): 배치 단위 점수 매기기 (모델 폴백 지원)
5. Ranker: 보너스 적용 + 코드 감지
6. Agent 3 (Summarizer): 상위 논문 한국어 요약 (Tier 1/2)
7. 렌더링: JSON, Markdown, HTML 리포트
8. 알림: Discord / Telegram

**Post Loop**: 교차 토픽 중복 태깅, 메타데이터 커밋

### 점수 시스템

**기본 점수** (Agent 2): 0-100 관련성 점수 + 플래그 (edge, realtime, code)

**보너스**: edge +5, realtime +5, code +3 (최대 +13, 100 상한)

**최종 점수** (embedding 비활성화 시, 기본):
```
final = 0.80 * (base + bonus) + 0.20 * recency
```

**최종 점수** (embedding 활성화 시):
```
final = 0.55 * (base + bonus) + 0.35 * embed + 0.10 * recency
```

**임계값 완화**: 기준 충족 논문 < 5개이면 50 → 40으로 단계 완화

### 주간 작업

매주 일요일 자동 실행:
- DB 정리 (보존 기간 초과 데이터 삭제, VACUUM)
- 주간 트렌드 요약
- 논문 업데이트 스캔
- 시각화 생성 (선택, `requirements-viz.txt` 필요)

### LLM 모델 폴백

요청 실패 시 `fallback_models`에 설정된 순서대로 모델을 자동 전환합니다. 타임아웃도 단계별로 증가 (180s → 240s → 300s).

---

## 기술 스택

| 구성 요소 | 기술 |
|----------|------|
| LLM | OpenRouter API (DeepSeek, Qwen, GLM 등 무료 모델) |
| 논문 소스 | arXiv API |
| 데이터베이스 | SQLite (WAL 모드) 또는 Supabase (PostgreSQL) |
| 중복 제거 | DB + JSONL 2티어 (30일 롤링) |
| 웹 UI | Flask + SSE 실시간 로그 |
| CI/CD | GitHub Actions |
| 리포트 배포 | GitHub Pages (gh-pages 브랜치) |
| 읽음 추적 | Supabase (선택, 크로스 디바이스 동기화) |
| 알림 | Discord 웹훅, Telegram Bot API |

---

## 설정 (config.yaml)

`config.yaml`에서 전체 동작을 설정합니다. 웹 UI 설정 탭에서도 편집 가능. 첫 실행 시 `config.example.yaml`을 복사하여 사용합니다.

### 토픽 설정

```yaml
topics:
  - slug: my-research
    name: My Research Topic
    description: |
      한국어로 연구 관심사를 상세히 설명합니다.
    arxiv_categories:
      - cs.AI
      - cs.LG
    notify:
      - provider: telegram
        secret_key: YKS
        events: [start, complete]
        send: [link]  # "link" | "readonly_link" | "md"
```

### 데이터베이스 설정

```yaml
database:
  provider: sqlite          # "sqlite" (기본) 또는 "supabase"
  path: data/paper_scout.db # sqlite 전용
  supabase:
    connection_string_env: SUPABASE_DB_URL  # PostgreSQL 연결 문자열 환경변수
```

SQLite는 로컬 실행, Supabase(PostgreSQL)는 클라우드 환경에서 사용합니다.

### 알림 환경변수 매핑

토픽의 `secret_key`에 따라 환경변수 이름이 결정됩니다:

| Provider | 환경변수 | 예시 (secret_key: YKS) |
|----------|---------|----------------------|
| Discord | `DISCORD_WEBHOOK_{KEY}` | `DISCORD_WEBHOOK_YKS` |
| Telegram | `TELEGRAM_BOT_TOKEN_{KEY}` | `TELEGRAM_BOT_TOKEN_YKS` |
| Telegram | `TELEGRAM_CHAT_ID_{KEY}` | `TELEGRAM_CHAT_ID_YKS` |

Discord는 `DISCORD_WEBHOOK_URL`을 폴백으로도 지원합니다.

---

## GitHub Pages & 읽음 추적

### GitHub Pages

리포트를 GitHub Pages에 배포하면 Telegram 알림에서 링크를 통해 바로 접근할 수 있습니다.

```yaml
output:
  gh_pages:
    enabled: true
    base_url: https://USERNAME.github.io/REPO_NAME
```

### 크로스 디바이스 읽음 추적 (Supabase)

논문 클릭 시 자동 "읽음" 처리, 여러 기기에서 동기화됩니다.

1. [supabase.com](https://supabase.com)에서 무료 프로젝트 생성
2. SQL Editor에서 테이블 생성:

```sql
CREATE TABLE read_papers (
  id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  paper_url text NOT NULL UNIQUE,
  read_at timestamptz DEFAULT now()
);

ALTER TABLE read_papers ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Allow anon read" ON read_papers FOR SELECT USING (true);
CREATE POLICY "Allow anon insert" ON read_papers FOR INSERT WITH CHECK (true);
CREATE INDEX idx_read_papers_url ON read_papers (paper_url);
```

3. `config.yaml`에서 설정:

```yaml
read_sync:
  enabled: true
  provider: supabase
  supabase_url: 'https://xxxx.supabase.co'
  supabase_anon_key: 'eyJxxxx...'
```

Supabase 미설정 시 브라우저 localStorage로 동작합니다 (기기 간 동기화 없음).

---

## 확장

### 새 논문 소스 추가

`core/sources/base.py`의 `SourceAdapter`를 상속하여 `core/sources/`에 구현하고, `registry.py`에 등록합니다.

### 커스텀 알림기 추가

`output/notifiers/base.py`의 `NotifierBase`를 상속하여 구현하고, `registry.py`에 등록합니다.

---

## arXiv API 준수

[arXiv API](https://info.arxiv.org/help/api/) 이용 약관을 준수합니다. 요청 간 최소 3초 지연 적용 (`sources.arxiv.delay_seconds`).

## License

MIT License - Copyright (c) 2026 Paper Scout Contributors
