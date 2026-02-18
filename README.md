# Paper Scout

LLM 기반 평가와 한국어 리포트 생성을 지원하는 자동화된 arXiv 논문 수집 및 분석 시스템.

## 개요

Paper Scout는 GitHub Actions에서 완전히 실행되는 서버리스 시스템으로, 사용자 정의 주제에 따라 arXiv에서 연구 논문을 수집하고, LLM 에이전트로 평가하며, 우선순위가 지정된 한국어 리포트를 생성하여 GitHub Pages를 통해 정적 HTML로 배포합니다. 이 시스템은 멀티 토픽 추적, 선택적 embedding 기반 순위와 결합된 하이브리드 필터링, Discord 및 Telegram 채널을 통한 알림을 지원합니다.

사용자는 연구 관심사를 한국어로 설명하고, 시스템은 키워드 확장, 쿼리 생성, 논문 수집, 관련성 점수 매기기, 클러스터링, 요약, 리포트 전달을 매일 자동으로 처리합니다.

## 주요 기능

### 핵심 역량

- **서버리스 아키텍처**: GitHub Actions에서 완전히 실행되며 관리할 인프라가 없음
- **멀티 토픽 추적**: 독립적인 구성으로 여러 연구 영역을 동시에 모니터링
- **LLM 에이전트 파이프라인**: 키워드 확장, 점수 매기기, 요약을 처리하는 3개의 전문 에이전트
- **하이브리드 필터링**: 규칙 기반 필터링 + 방어 캡 + 선택적 embedding 유사도 순위의 3단계 필터
- **한국어 리포트**: 영어 논문에서 포괄적인 한국어 리포트 생성
- **다중 알림 채널**: 토픽별 독립 설정이 가능한 Discord 및 Telegram 통합
- **GitHub Pages 배포**: 반응형 디자인의 정적 HTML 리포트
- **로컬 관리**: 토픽 관리, 키워드 편집, 파이프라인 실행을 위한 웹 UI 및 CLI

### 기술적 특징

- **무료 모델 지원**: OpenRouter를 통한 무료 모델 호환 (DeepSeek, Qwen, GLM 등)
- **모델 폴백**: 타임아웃 시 자동으로 대체 모델로 전환하는 sticky fallback 메커니즘
- **배치 타임아웃**: 배치별 시간 제한으로 무한 대기 방지
- **코드 감지**: regex + LLM 이중 감지로 코드 공개 논문 식별
- **결정론적 점수 매기기**: 엣지 연구, 실시간 애플리케이션, 코드 가용성에 대한 보너스 시스템
- **교차 실행 중복 제거**: 원자적 파일 쓰기를 사용하는 SQLite 데이터베이스 + JSONL 기반 dedup
- **리마인드 시스템**: 고품질 논문 재추천 (교차 실행 추적)
- **확장 가능한 아키텍처**: 소스 어댑터 패턴으로 arXiv 이외의 추가 논문 소스 지원 가능
- **주간 자동화**: 데이터베이스 유지보수, 트렌드 분석, 업데이트 스캔, 선택적 시각화

## 아키텍처

### 파이프라인 흐름

```
사전 점검 (Preflight)
    ├─ 설정 검증, API 키 확인, RPM/일일 제한 감지
    ├─ response_format 지원 여부 판별
    └─ 검색 윈도우 계산 (3단계 폴백)
    ↓
토픽 루프 (활성화된 각 토픽에 대해)
    ├─ 에이전트 1: 키워드 확장 (캐시 지원)
    ├─ 쿼리 빌더: arXiv 검색 쿼리 생성
    ├─ 수집: arXiv에서 논문 가져오기
    ├─ 필터: 3단계 하이브리드 필터 (규칙 → 방어캡 → embedding/최신순)
    ├─ 에이전트 2: 논문 점수 매기기 (배치 처리 + 모델 폴백)
    ├─ 순위: 보너스 시스템 + 코드 감지 결합
    ├─ 에이전트 3: 상위 논문 한국어 요약 (Tier 1/2)
    ├─ 클러스터: 관련 논문 그룹화
    ├─ 렌더링: JSON, Markdown, HTML 생성
    └─ 알림: Discord, Telegram (토픽별 채널)
    ↓
루프 후 처리
    ├─ 교차 토픽 중복 태깅
    ├─ HTML 인덱스 빌드
    ├─ 알림 발송
    ├─ 메타데이터 Git 커밋
    └─ 임시 파일 정리
    ↓
주간 작업 (일요일 UTC 02:00)
    ├─ 데이터베이스 정리 (90일)
    ├─ 트렌드 요약
    ├─ 업데이트 스캔
    └─ 선택적 시각화
```

### 시스템 구성요소

**핵심 파이프라인**:
- `core/pipeline/preflight.py` - 설정 검증, API 키 확인, 컴포넌트 초기화
- `core/pipeline/topic_loop.py` - 토픽별 파이프라인 오케스트레이션
- `core/pipeline/search_window.py` - 3단계 폴백 검색 윈도우 계산
- `core/pipeline/post_loop.py` - 교차 토픽 처리, 알림, Git 커밋

**LLM 에이전트**:
- `agents/base_agent.py` - 에이전트 공통 기반 클래스 (JSON 파싱, think block 제거)
- `agents/keyword_expander.py` - 에이전트 1: 한국어 설명에서 키워드 확장
- `agents/scorer.py` - 에이전트 2: base_score 및 플래그로 논문 점수 매기기
- `agents/summarizer.py` - 에이전트 3: Tier 1/2 한국어 요약 생성

**LLM 인프라**:
- `core/llm/openrouter_client.py` - OpenRouter API 클라이언트 (재시도, 모델 폴백)
- `core/llm/rate_limiter.py` - RPM 슬라이딩 윈도우 + 일일 사용량 제한

**데이터 수집**:
- `core/sources/arxiv.py` - arXiv API 통합 (부분 결과 보존)
- `core/sources/arxiv_query_builder.py` - 다단계 검색 쿼리 구성
- `core/sources/base.py` - 소스 어댑터 추상 클래스
- `core/sources/registry.py` - 소스 어댑터 레지스트리

**처리**:
- `core/scoring/hybrid_filter.py` - 3단계 하이브리드 필터 (규칙 → 캡 → 정렬)
- `core/scoring/ranker.py` - 보너스 시스템, 최종 점수, 임계값 완화
- `core/scoring/code_detector.py` - regex + LLM 이중 코드 감지
- `core/embeddings/embedding_ranker.py` - 선택적 embedding 유사도
- `core/clustering/clusterer.py` - 논문 그룹화
- `core/parsing/json_parser.py` - 괄호 균형을 맞춘 강력한 JSON 파싱

**스토리지**:
- `core/storage/db_manager.py` - SQLite 데이터베이스 (WAL 모드)
- `core/storage/dedup.py` - 2티어 중복 제거 (인런 + 교차 실행, 원자적 파일 쓰기)
- `core/storage/usage_tracker.py` - 일일 API 호출 및 토픽 완료 추적

**출력**:
- `output/render/json_exporter.py` - 구조화된 JSON 출력
- `output/render/md_generator.py` - Markdown 리포트
- `output/render/html_generator.py` - 정적 HTML 사이트
- `output/notifiers/discord.py` - Discord 웹훅
- `output/notifiers/telegram.py` - Telegram 봇 API
- `output/notifiers/error_alert.py` - 파이프라인 오류 알림 (한국어 포맷)
- `output/notifiers/base.py` - 알림기 추상 클래스
- `output/notifiers/registry.py` - 알림기 레지스트리 (환경변수 매핑)
- `output/github_issue.py` - GitHub Issue 통합

**주간 작업**:
- `core/pipeline/weekly_guard.py` - 주간 실행 조건 판별
- `core/pipeline/weekly_db_maintenance.py` - DB 정리 및 압축
- `core/pipeline/weekly_summary.py` - 주간 트렌드 요약
- `core/pipeline/weekly_updates.py` - 논문 업데이트 스캔
- `core/pipeline/weekly_viz.py` - 시각화 생성

**관리**:
- `cli/commands.py` - CLI 명령 라우팅
- `cli/topic_commands.py` - 토픽 관리 명령
- `local_ui/app.py` - Flask 웹 인터페이스
- `local_ui/pipeline_runner.py` - 백그라운드 파이프라인 실행 (SSE 스트리밍)
- `local_ui/api/pipeline.py` - 파이프라인 API (실행, 취소, 상태 초기화)
- `local_ui/api/topics.py` - 토픽 CRUD API
- `local_ui/api/settings.py` - 설정 관리 API
- `local_ui/api/setup.py` - 초기 구성 마법사
- `local_ui/api/recommend.py` - arXiv 카테고리 추천 (LLM + 규칙 기반)

## 빠른 시작

### 사전 요구사항

- Python 3.9 이상
- Actions가 활성화된 GitHub 계정
- OpenRouter API 키 (무료 모델 지원)
- 선택사항: Discord 웹훅 URL 및/또는 Telegram 봇 인증 정보

### 로컬 설정

1. 리포지토리 클론:
```bash
git clone https://github.com/your-username/yks-paper-collector.git
cd yks-paper-collector
```

2. 의존성 설치:
```bash
pip install -r requirements.txt
```

3. 선택사항: 추가 기능 설치:
```bash
pip install -r requirements-embed.txt  # Embedding 지원
pip install -r requirements-viz.txt    # 주간 시각화
```

4. 환경변수 설정:
```bash
# .env 파일 생성
echo "OPENROUTER_API_KEY=your-api-key-here" > .env
```

5. 웹 UI로 설정:
```bash
python main.py ui
```
웹 UI에서 토픽 추가, 카테고리 선택, 알림 설정을 GUI로 진행할 수 있습니다.

6. 또는 CLI로 토픽 추가:
```bash
python main.py topic add
```

7. 파이프라인 테스트:
```bash
python main.py dry-run
```
에이전트 1 키워드 확장 및 쿼리 빌드만 실행합니다 (논문 수집 없음).

### GitHub Actions 설정

1. 이 리포지토리를 GitHub 계정으로 포크

2. GitHub Secrets 구성:
   - `OPENROUTER_API_KEY`: OpenRouter API 키
   - `DISCORD_WEBHOOK_{SECRET_KEY}`: Discord 웹훅 URL (토픽별, 선택사항)
   - `TELEGRAM_BOT_TOKEN_{SECRET_KEY}`: Telegram 봇 토큰 (선택사항)
   - `TELEGRAM_CHAT_ID_{SECRET_KEY}`: Telegram 채팅 ID (선택사항)

   `SECRET_KEY`는 `config.yaml`의 토픽별 `notify.secret_key` 값과 일치해야 합니다.

3. GitHub Actions 활성화:
   - 리포지토리 Settings > Actions > General로 이동
   - 워크플로우에 대해 "Read and write permissions" 활성화

4. GitHub Pages 활성화:
   - Settings > Pages로 이동
   - Source: Deploy from branch
   - Branch: gh-pages
   - Folder: / (root)

5. 토픽이 정의된 `config.yaml`을 커밋하고 푸시

6. 워크플로우는 매일 UTC 02:00 (KST 11:00)에 자동으로 실행됩니다

7. 수동 실행:
   - Actions 탭으로 이동
   - "Paper Scout Daily Run" 워크플로우 선택
   - "Run workflow" 클릭
   - 선택사항으로 날짜 범위(KST), 모드 또는 중복 제거 설정 지정

## 구성

### 기본 구성 구조

```yaml
llm:
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
  model: deepseek/deepseek-r1-0528:free
  fallback_models:
    - qwen/qwen3-235b-a22b-thinking-2507
    - z-ai/glm-4.5-air:free
  timeout:
    attempt_1: 180
    attempt_2: 240
    fallback: 300
  retry:
    max_retries: 3
    backoff_base: 2
    jitter: true

agents:
  keyword_expander:
    max_tokens: 40000
    temperature: 0.3
    cache_ttl_days: 30
  scorer:
    batch_size: 5
    batch_timeout_seconds: 500
    temperature: 0.2
  summarizer:
    tier1_batch_size: 5
    tier2_batch_size: 10
    temperature: 0.4

sources:
  arxiv:
    max_results_per_query: 200
    delay_seconds: 3

filter:
  pre_embed_cap: 2000

scoring:
  weights:
    embedding_off:
      llm: 0.8
      recency: 0.2
    embedding_on:
      llm: 0.55
      embed: 0.35
      recency: 0.1
  bonus:
    is_edge: 5
    is_realtime: 5
    has_code: 3
  thresholds:
    default: 60
    relaxation_steps: [50, 40]
    min_papers: 5
  discard_cutoff: 20

remind:
  enabled: true
  min_score: 80
  max_expose_count: 2

clustering:
  enabled: true
  similarity_threshold: 0.85

topics:
  - slug: my-research
    name: My Research Topic
    description: |
      한국어로 연구 관심사를 상세히 설명합니다.
      여러 줄로 작성할 수 있습니다.
    arxiv_categories:
      - cs.AI
      - cs.LG
      - cs.CV
    notify:
      - provider: telegram
        secret_key: MY_KEY
        events: [start, complete]

output:
  gh_pages:
    enabled: true
    retention_days: 90
  github_issue:
    enabled: false

notifications:
  weekly_summary:
    provider: ""
```

### 토픽 구성

각 토픽은 다음을 필요로 합니다:
- `slug`: URL에 안전한 식별자 (알림 secret 매핑에 사용)
- `name`: 표시 이름
- `description`: 한국어 연구 관심 설명 (에이전트 1 입력)
- `arxiv_categories`: 검색 대상 arXiv 카테고리 목록

선택 필드:
- `notify`: 토픽별 알림 채널 목록 (`provider`, `secret_key`, `channel_id`, `events`)
- `must_concepts_en`: 필수 포함 개념 (영어)
- `should_concepts_en`: 선호 개념 (영어)
- `must_not_en`: 제외 키워드 (영어)

## CLI 사용법

### 토픽 관리

모든 토픽 나열:
```bash
python main.py topic list
```

새 토픽 추가 (대화형):
```bash
python main.py topic add
```

기존 토픽 편집:
```bash
python main.py topic edit edge-ai
```

토픽 제거:
```bash
python main.py topic remove edge-ai
```

### 파이프라인 실행

전체 파이프라인 실행 (모든 활성화된 토픽):
```bash
python main.py run
```

특정 토픽만 실행:
```bash
python main.py run --topic edge-ai
```

드라이 런 (에이전트 1 + QueryBuilder만):
```bash
python main.py dry-run
```

날짜 범위 지정 (UTC):
```bash
python main.py run --date-from 2024-01-01 --date-to 2024-01-31
```

교차 실행 중복 제거 비활성화:
```bash
python main.py run --dedup none
```

### 로컬 웹 UI

Flask 웹 인터페이스 시작:
```bash
python main.py ui
```

`http://127.0.0.1:8585`에서 접근 (브라우저 자동 실행)

```bash
# 포트 변경
python main.py ui --port 9090

# 브라우저 자동 열기 비활성화
python main.py ui --no-browser
```

기능:
- 토픽 관리 (추가, 편집, 삭제, arXiv 카테고리 AI 추천)
- 키워드 생성 (드라이런) 및 실시간 스트리밍 로그
- 생성된 키워드 편집 및 쿼리 미리보기
- 파이프라인 실행 (진행률, 단계별 상태, 로그 확인)
- 파이프라인 취소 및 상태 초기화
- 설정 뷰어 및 편집기
- 초기 구성 마법사
- DB 상태 대시보드

## 점수 시스템

### 기본 점수 매기기 (에이전트 2)

에이전트 2는 각 논문을 배치 단위로 평가하고 반환합니다:
- `base_score`: 0-100 관련성 점수
- `is_edge`: 엣지 컴퓨팅/배포 집중을 위한 불리언 플래그
- `is_realtime`: 실시간 시스템 애플리케이션을 위한 불리언 플래그
- `mentions_code`: LLM이 감지한 코드 가용성 플래그
- `is_metaphorical`: 비유적 표현 감지 플래그

배치 처리 시 `batch_timeout_seconds` (기본 500초) 초과 시 자동으로 fallback 모델로 전환됩니다.

### 코드 감지

코드 가용성은 이중 감지로 판별됩니다:
1. **Regex 기반**: GitHub URL, "code available", "open source" 등 패턴 매칭
2. **LLM 기반**: 에이전트 2의 `mentions_code` 플래그

두 결과를 병합하여 `has_code_source` 값을 결정합니다: `regex`, `llm`, `both`, `none`

### 보너스 시스템 (Ranker)

ranker는 구성 가능한 보너스를 적용합니다:
- 엣지 연구: +5점 (기본값)
- 실시간 애플리케이션: +5점 (기본값)
- 코드 가용성: +3점 (기본값)

총 보너스: 최대 +13점

### 최종 점수 계산

**embedding 활성화 시**:
```
final_score = (0.55 × llm_score) + (0.35 × embed_score × 100) + (0.10 × recency_score)
```

**embedding 비활성화 시** (기본값):
```
final_score = (0.80 × llm_score) + (0.20 × recency_score)
```

여기서:
- `llm_score = base_score + bonus`
- `embed_score` = 토픽 설명과의 코사인 유사도 (0.0-1.0)
- `recency_score` = 오늘은 100, 시간이 지남에 따라 감소

### 임계값 완화

최소 임계값을 충족하는 논문이 5개 미만인 경우, 시스템은 단계적으로 완화합니다:
1. 초기 임계값: 60
2. 첫 번째 완화: 50
3. 두 번째 완화: 40

`discard_cutoff` (기본 20) 미만의 논문은 완전히 제외됩니다.

### 리마인드 시스템

고품질 논문을 여러 실행에 걸쳐 재추천합니다:
- `min_score` (기본 80) 이상의 논문만 대상
- 최대 `max_expose_count` (기본 2)회까지 재추천
- 교차 실행 추적으로 중복 노출 방지

## 주간 작업

매주 일요일 UTC 02:00에 시스템은 유지보수 작업을 실행합니다:

### 데이터베이스 정리
- 90일 이상 된 논문 및 관련 데이터 제거
- 고아 레코드 정리
- SQLite 데이터베이스 압축 (VACUUM)

### 트렌드 요약
- 주간 상위 키워드 분석
- 신흥 연구 트렌드 식별
- 주간 트렌드 리포트 생성

### 업데이트 스캔
- arXiv에서 새 버전이 있는 논문 감지
- 추적된 논문의 중요한 업데이트 알림

### 시각화 (선택사항)
- 토픽 분포 차트 생성
- 키워드 동시 출현 그래프
- `requirements-viz.txt` 의존성 필요

## 확장 가이드

### 새 논문 소스 추가

1. `core/sources/`에 새 어댑터 생성:

```python
from core.sources.base import SourceAdapter

class NewSourceAdapter(SourceAdapter):
    def collect(self, agent1_output, categories, window_start, window_end, config):
        papers = []
        # ... 수집 로직 ...
        return papers
```

2. `core/sources/registry.py`에 어댑터 등록

3. `config.yaml`에 구성 추가

### 커스텀 알림기 추가

1. `output/notifiers/`에 알림기 생성:

```python
from output.notifiers.base import NotifierBase

class NewNotifier(NotifierBase):
    def send(self, topic_slug, summary, files):
        # 알림 로직 구현
        pass
```

2. `output/notifiers/registry.py`에 등록

3. 환경변수 매핑 추가 및 `config.yaml`의 토픽별 `notify` 섹션 업데이트

## arXiv API 공지

이 프로젝트는 arXiv API를 사용합니다. arXiv API 이용 약관을 검토하십시오:

https://info.arxiv.org/help/api/

주요 사항:
- 속도 제한: 3초당 최대 1개의 요청
- 대량 요청은 최소 지연을 포함해야 함
- 공개 애플리케이션에서 출처 표시 필요
- robots.txt 및 API 가이드라인 준수

Paper Scout는 arXiv API 약관을 준수하기 위해 속도 제한(config에서 `delay_seconds: 3`)을 구현합니다.

## License

MIT License

Copyright (c) 2026 Paper Scout Contributors

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 기여

기여를 환영합니다. 다음을 수행하십시오:
1. 리포지토리 포크
2. 기능 브랜치 생성
3. 새 기능에 대한 테스트 추가
4. 모든 테스트 통과 확인
5. 명확한 설명과 함께 풀 리퀘스트 제출

## 지원

문제, 질문 또는 기능 요청 사항:
- GitHub Issue 열기
- 기존 이슈에서 솔루션 확인
- 상세한 오류 메시지 및 로그 제공

## 감사의 말

- 연구 논문에 대한 오픈 액세스를 제공하는 arXiv
- LLM API 액세스를 위한 OpenRouter
- 서버리스 실행 인프라를 위한 GitHub Actions
