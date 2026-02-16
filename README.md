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
- **하이브리드 필터링**: 규칙 기반 필터링과 선택적 embedding 유사도 순위 결합
- **한국어 리포트**: 영어 논문에서 포괄적인 한국어 리포트 생성
- **다중 알림 채널**: 파일 첨부가 가능한 Discord 및 Telegram 통합
- **GitHub Pages 배포**: 반응형 디자인의 정적 HTML 리포트
- **로컬 관리**: 토픽 관리 및 파이프라인 테스트를 위한 웹 UI 및 CLI

### 기술적 특징

- **무료 모델 지원**: OpenAI SDK를 통한 OpenRouter 무료 모델과 호환
- **결정론적 점수 매기기**: 엣지 연구, 실시간 애플리케이션, 코드 가용성에 대한 보너스 시스템을 갖춘 구성 가능한 가중치
- **교차 실행 중복 제거**: GitHub Actions 캐시 지속성을 갖춘 SQLite 데이터베이스
- **확장 가능한 아키텍처**: 소스 어댑터 패턴으로 arXiv 이외의 추가 논문 소스 지원 가능
- **주간 자동화**: 데이터베이스 유지보수, 트렌드 분석, 업데이트 스캔, 선택적 시각화
- **포괄적인 테스트**: 1283개의 테스트 케이스를 다루는 123개의 테스트 파일

## 아키텍처

### 파이프라인 흐름

```
사전 점검
    ↓
토픽 루프 (활성화된 각 토픽에 대해)
    ├─ 에이전트 1: 키워드 확장
    ├─ 쿼리 빌더: arXiv 검색 쿼리
    ├─ 수집: arXiv에서 논문 가져오기
    ├─ 필터: 규칙 필터 + 선택적 embedding 정렬
    ├─ 에이전트 2: 논문 점수 매기기 (0-100 + 플래그)
    ├─ 순위: 보너스 시스템 적용
    ├─ 에이전트 3: 상위 논문 요약
    ├─ 클러스터: 관련 논문 그룹화
    ├─ 렌더링: JSON, Markdown, HTML 생성
    └─ 알림: Discord, Telegram
    ↓
루프 후 처리
    ├─ GitHub Issue: 일일 요약 업데이트
    └─ 완료
    ↓
주간 작업 (일요일 UTC 02:00)
    ├─ 데이터베이스 정리 (90일)
    ├─ 트렌드 요약
    ├─ 업데이트 스캔
    └─ 선택적 시각화
```

### 시스템 구성요소

**핵심 파이프라인**:
- `core/pipeline/preflight.py` - 구성 및 API 검증
- `core/pipeline/orchestrator.py` - 토픽 루프 조정
- `core/pipeline/post_loop.py` - 후처리 작업
- `core/pipeline/weekly_tasks.py` - 유지보수 및 분석

**LLM 에이전트**:
- `agents/keyword_expander.py` - 에이전트 1: 한국어 설명에서 키워드 확장
- `agents/scorer.py` - 에이전트 2: base_score 및 플래그로 논문 점수 매기기
- `agents/summarizer.py` - 에이전트 3: 한국어 요약 생성

**데이터 수집**:
- `core/sources/arxiv.py` - arXiv API 통합
- `core/sources/arxiv_query_builder.py` - 검색 쿼리 구성
- `core/sources/registry.py` - 소스 어댑터 레지스트리

**처리**:
- `core/scoring/ranker.py` - 보너스 시스템을 사용한 점수 매기기
- `core/embeddings/embedding_ranker.py` - 선택적 embedding 유사도
- `core/clustering/clusterer.py` - 논문 그룹화
- `core/parsing/json_parser.py` - 괄호 균형을 맞춘 강력한 JSON 파싱

**스토리지**:
- `core/storage/db_manager.py` - SQLite 데이터베이스 작업
- `core/storage/dedup.py` - 교차 실행 중복 제거
- `core/storage/usage_tracker.py` - LLM 토큰 사용량 추적

**출력**:
- `output/render/json_exporter.py` - 구조화된 JSON 출력
- `output/render/md_generator.py` - Markdown 리포트
- `output/render/html_generator.py` - 정적 HTML 사이트
- `output/notifiers/discord.py` - Discord 웹훅
- `output/notifiers/telegram.py` - Telegram 봇 API
- `output/github_issue.py` - GitHub Issue 통합

**관리**:
- `cli/topic_commands.py` - 토픽 관리 명령
- `cli/commands.py` - 파이프라인 실행
- `local_ui/app.py` - Flask 웹 인터페이스

## 빠른 시작

### 사전 요구사항

- Python 3.11 이상
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
pip install -r requirements-core.txt
```

3. 선택사항: 추가 기능 설치:
```bash
pip install -r requirements-embed.txt  # Embedding 지원
pip install -r requirements-ui.txt     # 로컬 웹 UI
pip install -r requirements-viz.txt    # 주간 시각화
```

4. 시스템 구성:
```bash
cp config.example.yaml config.yaml
```

`config.yaml`을 편집하여 API 키와 선호사항을 설정합니다.

5. 첫 번째 토픽 추가:
```bash
python main.py topic add
```

대화형 프롬프트를 따라 한국어로 연구 관심사를 정의합니다.

6. 파이프라인 테스트:
```bash
python main.py --mode dry-run
```

이는 논문을 수집하지 않고 에이전트 1 키워드 확장 및 쿼리 빌드만 실행합니다.

### GitHub Actions 설정

1. 이 리포지토리를 GitHub 계정으로 포크

2. GitHub Secrets 구성:
   - `OPENROUTER_API_KEY`: OpenRouter API 키
   - `DISCORD_WEBHOOK_TOPIC_SLUG`: Discord 웹훅 URL (선택사항, 토픽별)
   - `TELEGRAM_BOT_TOKEN`: Telegram 봇 토큰 (선택사항)
   - `TELEGRAM_CHAT_ID_TOPIC_SLUG`: Telegram 채팅 ID (선택사항, 토픽별)

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
   - 선택사항으로 날짜 범위, 모드 또는 중복 제거 설정 지정

## 구성

### 기본 구성 구조

```yaml
llm:
  provider: openrouter
  base_url: https://openrouter.ai/api/v1
  api_key: ${OPENROUTER_API_KEY}
  model: google/gemini-flash-1.5-8b  # 무료 티어 모델
  temperature: 0.7
  max_tokens: 2048
  timeout_seconds: 60
  retry_attempts: 3

sources:
  arxiv:
    enabled: true
    base_url: https://export.arxiv.org/api/query
    results_per_query: 50
    rate_limit_delay_seconds: 3

filtering:
  use_embeddings: false
  embed_model: sentence-transformers/all-MiniLM-L6-v2
  embed_topk: 100
  min_score_threshold: 60

scoring:
  weights:
    llm_score: 0.80
    recency_score: 0.20
  bonus:
    is_edge: 5
    is_realtime: 5
    has_code: 3

output:
  formats:
    - json
    - markdown
    - html
  html:
    theme: modern
    include_clusters: true

notifications:
  discord:
    enabled: true
    include_attachment: true
  telegram:
    enabled: false

topics:
  - name: Edge AI Research
    name_en: Edge AI Research
    slug: edge-ai
    enabled: true
    description: |
      On-device AI model optimization techniques
      Real-time inference on resource-constrained devices
      TinyML and model compression methods
    preference: |
      Prefer papers with code implementation
      Focus on practical deployment techniques
      Include benchmark comparisons
    keywords:
      - edge computing
      - model compression
      - TinyML
    date_from: 2024-01-01
```

### 토픽 구성

각 토픽은 다음을 필요로 합니다:
- `name`: 한국어 표시 이름
- `name_en`: 파일 명명을 위한 영어 이름
- `slug`: URL에 안전한 식별자 (secrets에 사용됨)
- `enabled`: 활성화/비활성화를 위한 불리언
- `description`: 한국어 연구 관심 설명 (에이전트 1 입력)
- `preference`: 한국어 평가 기준 (에이전트 2 가이드)
- `keywords`: 쿼리 빌드를 위한 핵심 검색어
- `date_from`: 논문 수집 시작 날짜 (YYYY-MM-DD)

선택 필드:
- `custom_scoring`: 토픽별 전역 점수 가중치 재정의
- `notification_override`: 토픽별 알림 설정

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
python main.py
# 또는 명시적으로
python main.py --mode full
```

드라이 런 (에이전트 1 + QueryBuilder만):
```bash
python main.py --mode dry-run
```

날짜 범위 지정:
```bash
python main.py --date-from 2024-01-01 --date-to 2024-01-31
```

교차 실행 중복 제거 비활성화:
```bash
python main.py --dedup none
```

embedding 필터 활성화:
```bash
python main.py --use-embeddings
```

옵션 결합:
```bash
python main.py --mode full --date-from 2024-01-01 --use-embeddings
```

### 로컬 웹 UI

Flask 웹 인터페이스 시작:
```bash
python main.py ui
```

`http://127.0.0.1:5000`에서 접근

기능:
- 토픽 관리 (추가, 편집, 활성화/비활성화)
- 라이브 로그가 있는 파이프라인 실행
- 구성 뷰어 및 편집기
- 통계 대시보드

## 점수 시스템

### 기본 점수 매기기 (에이전트 2)

에이전트 2는 각 논문을 평가하고 반환합니다:
- `base_score`: 0-100 관련성 점수
- `is_edge`: 엣지 컴퓨팅/배포 집중을 위한 불리언 플래그
- `is_realtime`: 실시간 시스템 애플리케이션을 위한 불리언 플래그
- `has_code`: 코드 가용성을 위한 불리언 플래그

### 보너스 시스템 (Ranker)

ranker는 구성 가능한 보너스를 적용합니다:
- 엣지 연구: +5점 (기본값)
- 실시간 애플리케이션: +5점 (기본값)
- 코드 가용성: +3점 (기본값)

총 보너스: 최대 +13점

### 최종 점수 계산

**embedding 활성화 시** (use_embeddings: true):
```
final_score = (0.55 × llm_score) + (0.35 × embed_score × 100) + (0.10 × recency_score)
```

**embedding 비활성화 시** (use_embeddings: false, 기본값):
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

이는 틈새 토픽에 대해서도 최소한의 논문이 포함되도록 보장합니다.

## 주간 작업

매주 일요일 UTC 02:00에 시스템은 유지보수 작업을 실행합니다:

### 데이터베이스 정리
- 90일 이상 된 논문 제거 (구성 가능)
- SQLite 데이터베이스 압축
- GitHub Actions 캐시 공간 확보

### 트렌드 요약
- 주간 상위 키워드 분석
- 신흥 연구 트렌드 식별
- 주간 트렌드 리포트 생성

### 업데이트 스캔
- arXiv에서 새 버전이 있는 논문 감지
- 추적된 논문의 중요한 업데이트 알림
- 이전 버전 링크

### 시각화 (선택사항)
- 토픽 분포 차트 생성
- 키워드 동시 출현 그래프 생성
- UMAP 및 matplotlib으로 대화형 시각화 생성
- `requirements-viz.txt` 의존성 필요

주간 리포트는 일일 리포트와 동일한 알림 채널에 게시됩니다.

## 확장 가이드

### 새 논문 소스 추가

1. `core/sources/`에 새 어댑터 생성:

```python
from core.sources.base_adapter import SourceAdapter

class NewSourceAdapter(SourceAdapter):
    def fetch(self, query, date_from, date_to, max_results):
        # API 통합 구현
        papers = []
        # ... 가져오기 로직 ...
        return papers
```

2. `core/sources/registry.py`에 어댑터 등록:

```python
from core.sources.new_source_adapter import NewSourceAdapter

def create_source(source_name, config):
    if source_name == "new_source":
        return NewSourceAdapter(config)
    # ... 기존 소스 ...
```

3. `config.yaml`에 구성 추가:

```yaml
sources:
  new_source:
    enabled: true
    api_url: https://api.newsource.org
    # ... 소스별 설정 ...
```

### 커스텀 알림기 추가

1. `output/notifiers/`에 알림기 생성:

```python
from output.notifiers.base_notifier import BaseNotifier

class NewNotifier(BaseNotifier):
    def send(self, topic_slug, summary, files):
        # 알림 로직 구현
        pass
```

2. `output/notifiers/registry.py`에 등록:

```python
from output.notifiers.new_notifier import NewNotifier

def create_notifier(notifier_type, config):
    if notifier_type == "new_notifier":
        return NewNotifier(config)
    # ... 기존 알림기 ...
```

3. GitHub Actions에 secrets 추가하고 `config.yaml` 업데이트

### 점수 매기기 로직 커스터마이징

`core/scoring/ranker.py`를 편집하여 수정:
- 보너스 점수 값
- 점수 가중치 분배
- 임계값 완화 전략
- 논문 메타데이터 기반 커스텀 점수 규칙

커스텀 규칙 예시:
```python
if paper.author_count > 10:
    bonus += 2  # 대규모 협업에 대한 보너스
```

## 테스트

### 테스트 실행

모든 테스트 실행:
```bash
pytest
```

특정 테스트 파일 실행:
```bash
pytest tests/test_scorer.py
```

커버리지와 함께 실행:
```bash
pytest --cov=. --cov-report=html
```

### 테스트 커버리지

프로젝트는 포괄적인 테스트 커버리지를 유지합니다:
- 123개의 Python 테스트 파일
- 총 1283개의 테스트 케이스
- 커버리지 영역:
  - LLM 에이전트 (키워드 확장, 점수 매기기, 요약)
  - 파이프라인 오케스트레이션
  - 데이터 수집 및 필터링
  - 점수 매기기 및 순위
  - 출력 생성 (JSON, Markdown, HTML)
  - 알림 전달
  - CLI 명령
  - 데이터베이스 작업
  - 오류 처리 및 엣지 케이스

### CI/CD 테스트

GitHub Actions 워크플로우 포함 사항:
- 푸시 및 풀 리퀘스트 시 자동화된 테스트 실행
- 테스트 결과 리포팅
- 커버리지 추적

## arXiv API 공지

이 프로젝트는 arXiv API를 사용합니다. arXiv API 이용 약관을 검토하십시오:

https://info.arxiv.org/help/api/

주요 사항:
- 속도 제한: 3초당 최대 1개의 요청
- 대량 요청은 최소 지연을 포함해야 함
- 공개 애플리케이션에서 출처 표시 필요
- robots.txt 및 API 가이드라인 준수

Paper Scout는 arXiv API 약관을 준수하기 위해 속도 제한(config에서 `rate_limit_delay_seconds: 3`)을 구현합니다.

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
- 개발 프레임워크 및 도구를 위한 MoAI-ADK
