# Paper Scout

연구자를 위한 **AI 논문 큐레이션 시스템**. arXiv에서 매일 쏟아지는 수천 편의 논문 중 내 연구와 관련된 핵심 논문만 골라 한국어 리포트로 만들어줍니다.

### 왜 만들었나?

arXiv에는 매일 500편 이상의 논문이 올라옵니다. 이 중 내 연구에 진짜 필요한 논문을 찾으려면 매일 시간을 들여 훑어봐야 합니다. Paper Scout는 이 과정을 자동화합니다:

- **AI가 논문을 읽고 점수를 매깁니다** — 내 연구 주제와의 관련성을 0~100점으로 평가
- **한국어로 요약해줍니다** — 핵심 기여, 방법론, 실험 결과를 한눈에 파악
- **매일 자동으로 실행됩니다** — GitHub Actions로 매일 아침 리포트가 도착
- **놓친 논문을 다시 추천합니다** — 리마인드 시스템으로 중요 논문 재노출

### 기대 효과

- 논문 탐색 시간 **90% 이상 절감** (매일 1시간 → 5분)
- 관련 논문을 **놓치지 않는** 체계적 모니터링
- 한국어 요약으로 **빠른 의사결정** (읽을지 말지 즉시 판단)
- 무료 LLM 모델 사용으로 **비용 부담 제로**

---

## 사용 가이드

### 1단계: 설치

```bash
git clone https://github.com/your-username/yks-paper-collector.git
cd yks-paper-collector
pip install -r requirements.txt
```

### 2단계: API 키 준비

[OpenRouter](https://openrouter.ai/)에서 API 키를 발급받습니다 (무료 모델 지원).

```bash
echo "OPENROUTER_API_KEY=your-key" > .env
```

알림을 받으려면 Discord 웹훅 URL이나 Telegram 봇 토큰도 `.env`에 추가합니다:

```bash
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/...
TELEGRAM_BOT_TOKEN_YKS=your-bot-token
TELEGRAM_CHAT_ID_YKS=your-chat-id
```

### 3단계: 실행

```bash
python main.py
```

브라우저가 자동으로 열리며 (`http://127.0.0.1:8585`), 웹 UI에서 모든 설정과 실행을 할 수 있습니다.

### 4단계: 웹 UI에서 설정

1. **초기 설정 마법사**: 첫 실행 시 API 키 입력, 알림 설정을 안내합니다
2. **토픽 추가**: 연구 관심사를 한국어로 입력하면, AI가 arXiv 카테고리를 추천합니다
3. **키워드 생성**: "드라이런" 버튼으로 AI가 검색 키워드를 자동 생성합니다
4. **키워드 편집**: 생성된 키워드를 직접 수정하고 쿼리를 미리볼 수 있습니다
5. **파이프라인 실행**: "실행" 버튼으로 논문 수집 ~ 리포트 생성까지 전체 파이프라인을 실행합니다

### 5단계: GitHub Actions 자동 실행

로컬에서 설정을 완료한 후, GitHub Actions로 매일 자동 실행할 수 있습니다.

1. 이 리포지토리를 GitHub에 포크합니다
2. GitHub Secrets를 설정합니다:
   - `OPENROUTER_API_KEY`: OpenRouter API 키
   - 알림용: `DISCORD_WEBHOOK_URL`, `TELEGRAM_BOT_TOKEN`, `TELEGRAM_CHAT_ID` 등
3. Repository Settings > Actions > General에서 **Read and write permissions** 활성화
4. 웹 UI의 **"GitHub Actions 동기화"** 버튼을 클릭하여 로컬 설정을 GitHub에 반영합니다

동기화 버튼은 다음 파일을 GitHub Secrets로 업로드합니다:
- `config.yaml` → `PAPER_SCOUT_CONFIG`
- `data/keyword_cache.json` → `PAPER_SCOUT_KEYWORD_CACHE`
- `data/last_success.json` → `PAPER_SCOUT_LAST_SUCCESS`
- `data/model_caps.json` → `PAPER_SCOUT_MODEL_CAPS`

워크플로우는 매일 **UTC 02:00 (KST 11:00)**에 자동 실행됩니다.

### 수동 실행

GitHub Actions 탭에서 "Run workflow"를 클릭하여 수동 실행할 수도 있습니다:

| 파라미터 | 설명 | 기본값 |
|---------|------|--------|
| `date_from` | 검색 시작일 (YYYY-MM-DD, KST) | 자동 계산 |
| `date_to` | 검색 종료일 (YYYY-MM-DD, KST) | 오늘 |
| `mode` | `full` (전체) / `dry-run` (키워드만) | `full` |
| `dedup` | `skip_recent` (중복 제거) / `none` | `skip_recent` |

### 날짜 동작

| 상황 | 검색 범위 |
|------|----------|
| 자동 실행 (매일) | 마지막 성공 ~ 오늘 |
| 수동 실행 (날짜 미지정) | 마지막 성공 ~ 오늘 |
| 수동 실행 (날짜 지정) | 지정한 범위 |
| 첫 실행 (이력 없음) | 최근 3일 |

### CLI 사용법

웹 UI 외에 CLI로도 사용할 수 있습니다:

```bash
python main.py run                          # 전체 파이프라인
python main.py run --topic my-topic         # 특정 토픽만
python main.py dry-run                      # 키워드 생성만
python main.py run --date-from 2025-01-01   # 날짜 지정
python main.py topic list                   # 토픽 목록
python main.py topic add                    # 토픽 추가 (대화형)
```

---

## 리포트

파이프라인 실행 후 `tmp/reports/`에 HTML 리포트가 생성됩니다.

- **Tier 1 카드**: 상위 논문에 대한 상세 분석 (한국어 요약, 핵심 기여, 방법론 등)
- **Tier 2 카드**: 나머지 논문의 간략 요약
- **3개 탭**: 오늘의 논문 / 리마인드 (고점수 재추천) / 제외됨
- **PDF 다운로드**: 각 논문의 arXiv PDF 직접 다운로드
- **다크 모드**: 자동 지원

---

## 아키텍처

### 파이프라인 흐름

```
Preflight → Topic Loop (토픽별) → Post Loop
```

**Preflight**: 설정 검증, API 키 확인, 검색 윈도우 계산

**Topic Loop** (각 토픽에 대해):
1. Agent 1: 한국어 설명 → 영어 키워드 확장 (캐시 지원)
2. arXiv API로 논문 수집
3. 3단계 하이브리드 필터 (규칙 → 방어캡 → embedding/최신순)
4. Agent 2: 배치 단위 논문 점수 매기기 (모델 폴백 지원)
5. Ranker: 보너스 적용 + 코드 감지
6. Agent 3: 상위 논문 한국어 요약 (Tier 1/2)
7. 렌더링: JSON, Markdown, HTML 리포트 생성
8. 알림: Discord / Telegram

**Post Loop**: 교차 토픽 중복 태깅, 메타데이터 커밋, 정리

### 점수 시스템

**기본 점수** (Agent 2): 0-100 관련성 점수 + 플래그 (엣지, 실시간, 코드)

**보너스**: 엣지 연구 +5, 실시간 +5, 코드 공개 +3 (최대 +13)

**최종 점수** (embedding 비활성화 시, 기본):
```
final = 0.80 × (base_score + bonus) + 0.20 × recency_score
```

**임계값 완화**: 기준 충족 논문 < 5개이면 60 → 50 → 40으로 단계 완화

### 주요 기술 스택

| 구성 요소 | 기술 |
|----------|------|
| LLM | OpenRouter API (DeepSeek, Qwen, GLM 등 무료 모델) |
| 논문 소스 | arXiv API |
| 데이터베이스 | SQLite (WAL 모드) |
| 중복 제거 | SQLite + JSONL 2티어 |
| 웹 UI | Flask + SSE 실시간 로그 |
| CI/CD | GitHub Actions (매일 UTC 02:00) |
| 리포트 배포 | GitHub Pages (gh-pages 브랜치) |
| 읽음 추적 | Supabase (선택, 크로스 디바이스 동기화) |
| 알림 | Discord 웹훅, Telegram Bot API |

### LLM 에이전트

| 에이전트 | 역할 | 입력 | 출력 |
|---------|------|------|------|
| Agent 1 (Keyword Expander) | 키워드 확장 | 한국어 토픽 설명 | 영어 검색 키워드 |
| Agent 2 (Scorer) | 논문 점수 | 논문 제목+초록 | base_score + 플래그 |
| Agent 3 (Summarizer) | 한국어 요약 | 상위 논문 | Tier 1/2 요약 |

### 주간 작업

매주 일요일 UTC 02:00에 자동 실행:
- DB 정리 (90일 이상 데이터 삭제, VACUUM)
- 주간 트렌드 요약
- 논문 업데이트 스캔
- 시각화 생성 (선택, `requirements-viz.txt` 필요)

---

## 구성

`config.yaml`에서 전체 동작을 설정합니다. 웹 UI의 설정 탭에서도 편집 가능합니다.

### 토픽 설정 예시

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
```

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

### GitHub Pages 배포

리포트를 GitHub Pages에 배포하면 고정 URL로 접근할 수 있습니다. Telegram 알림에서 링크를 클릭하면 바로 리포트를 볼 수 있습니다.

`config.yaml`에서 설정:

```yaml
output:
  gh_pages:
    enabled: true
    base_url: https://USERNAME.github.io/REPO_NAME
    notify_mode: link  # "link" = URL만 전송, "file" = HTML 파일 첨부
```

### 크로스 디바이스 읽음 추적 (Supabase)

논문을 클릭하면 자동으로 "읽음" 처리되며, 여러 기기에서 읽음 상태가 동기화됩니다. 무료 [Supabase](https://supabase.com) 프로젝트를 생성하여 연동합니다.

**설정 방법:**

1. [supabase.com](https://supabase.com)에서 무료 프로젝트 생성
2. **SQL Editor**에서 테이블 생성:

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

3. **Settings > API** 페이지에서 두 값을 복사:
   - **Data API URL** (`https://xxxx.supabase.co`)
   - **anon public key** (`eyJxxxx...`)

4. `config.yaml` 또는 로컬 UI 설정 탭에서 입력:

```yaml
read_sync:
  enabled: true
  provider: supabase
  supabase_url: 'https://xxxx.supabase.co'
  supabase_anon_key: 'eyJxxxx...'
```

Supabase를 설정하지 않아도 논문 읽음 추적은 브라우저 localStorage로 동작합니다 (단, 기기 간 동기화 없음).

---

## 확장

### 새 논문 소스 추가

`core/sources/base.py`의 `SourceAdapter`를 상속하여 `core/sources/`에 어댑터를 구현하고, `registry.py`에 등록합니다.

### 커스텀 알림기 추가

`output/notifiers/base.py`의 `NotifierBase`를 상속하여 구현하고, `registry.py`에 등록합니다.

---

## arXiv API 준수

이 프로젝트는 [arXiv API](https://info.arxiv.org/help/api/) 이용 약관을 준수합니다.
요청 간 최소 3초 지연을 적용합니다 (`sources.arxiv.delay_seconds`).

## License

MIT License - Copyright (c) 2026 Paper Scout Contributors
