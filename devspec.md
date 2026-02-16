# Paper Scout -- 상세 기술 스펙 문서 v1.5

> 문서 유형: Technical Specification
> 작성일: 2026-02-16
> 버전: v1.5 (최종본)

---

# 목차

1. 목표 및 비목표
2. 핵심 제약 및 품질 목표
3. 시스템 아키텍처
4. 데이터 모델
5. 입력(TopicSpec) 설계
6. 소스 어댑터 설계
7. LLM 에이전트 설계
8. 스코어링 및 랭킹
9. 파이프라인 상세 프로세스
10. 출력물 설계
11. 알림(Notifier) 설계
12. 저장소 및 성장 제어
13. GitHub Actions 워크플로
14. 보안 및 컴플라이언스
15. 관측성 및 재현성
16. 에러 대응 정책
17. 테스트 계획
18. 로컬 관리 도구
19. 확장 가이드
20. 부록

---

# 1. 목표 및 비목표

## 1-1. 목표

사용자가 한국어로 관심 주제를 서술하면, arXiv에서 관련 논문을 매일 자동 수집하고, LLM이 읽고 평가하여 우선순위가 매겨진 한국어 리포트를 생성한다. 결과물은 HTML 웹 리포트(gh-pages)로 배포하고, 선택한 알림 채널(Discord/Telegram)로 파일을 전송한다.

## 1-2. 비목표 (의도적 제외)

- 실시간 스트리밍 / 상시 서버 운영 (항상 GitHub Actions 서버리스)
- PDF 전문 요약 / 인용 생성 (초록 기반, PDF는 링크 제공)
- 사용자별 권한 / 세션 / 배포형 웹 앱 (리포트는 정적 HTML, 관리는 로컬 도구)
- 피드백 루프 / 사용자 반응 기반 자동 보정
- 북마크 / 즐겨찾기 (서버 필요, 향후 버전)

---

# 2. 핵심 제약 및 품질 목표

## 2-1. 제약

| 항목 | 제약 | 비고 |
|------|------|------|
| 인프라 | GitHub Actions (서버리스) | public repo 무료. private는 월 2,000분 무료 후 과금 |
| LLM | OpenRouter (:free 모델 기본) | RPM/일일 한도는 크레딧 종속, Preflight에서 자동 감지 |
| 알림 | Discord webhook / Telegram bot | 토픽별 채널 분리 가능, 혼용 가능 |
| 저장 | main 브랜치에 리포트 커밋 금지 | 리포트는 gh-pages + 알림 파일 전송 |
| 비용 | $10/월 (OpenRouter) + $0 (Actions, public) | |

## 2-2. 품질 목표 (SLO)

| 항목 | 목표 |
|------|------|
| 일일 실행 성공률 | 99%+ (알림 실패가 전체 실패로 전파되지 않음) |
| 토픽 5개 기준 실행 시간 | 25분 이내 |
| 재현성 | 동일 입력/윈도우에서 결과 변동 최소화 (RunMeta 기록) |
| 저장소 성장 | main: 월 10MB 이하, gh-pages: 최근 90일만 보존 |

---

# 3. 시스템 아키텍처

## 3-1. 전체 구조

```
GitHub Actions Runner (cron KST 11:00 / manual)
  |
  +-- [Preflight]
  |     config 검증, API 키 상태, 알림 채널 연결, RPM/한도 감지
  |     run_id 발급, 검색 윈도우(UTC) 확정
  |
  +-- [토픽 루프] (토픽마다 반복)
  |     Agent 1 (키워드 확장)
  |       -> concepts + cross_domain_keywords + exclude_keywords + topic_embedding_text
  |     SourceAdapter.search() (arXiv)
  |       QueryBuilder: concepts/cross_domain_keywords -> arXiv 쿼리 DSL
  |       -> Paper[] (정규화)
  |     Dedup (in_run: run 내 중복 제거 / cross_run: seen_items topic별 스킵 / papers: 메타데이터 재활용)
  |     Hybrid Filter (규칙 -> cap 2000 -> 임베딩 정렬(optional) -> 상위 200편)
  |     Agent 2 (점수화: base_score + flags)
  |     Ranker (결정론적 가산 + 임계값 완화)
  |     Clusterer (임베딩 있을 때만)
  |     Agent 3 (한국어 요약)
  |     다시 보기 선정 (80점+, 최대 2회)
  |     리포트 생성 -> tmp/
  |
  +-- [토픽 간 중복 태깅]
  |
  +-- [출력/배포]
  |     HTML 빌드 -> gh-pages 배포 (keep_files + pruning)
  |     GitHub Issue upsert (선택)
  |     알림 전송 (토픽별 채널, 파일 첨부 + 폴백)
  |
  +-- [주간 작업] (일요일)
  |     DB purge + VACUUM
  |     주간 트렌드 요약 생성
  |     업데이트 논문 스캔
  |     시각화 생성 (optional)
  |
  +-- [git commit] (경량 텍스트 메타만)
        data/seen_items.jsonl
        data/issue_map.json
        data/usage/YYYYMMDD.json
        data/keyword_cache.json (변경 시)
        data/model_caps.json (변경 시)
        data/last_success.json
        data/weekly_done.flag (주 1회)
  +-- [cache save] paper_scout.db → actions/cache
  +-- [backup] (주 1회) paper_scout.db → Release asset
```

## 3-2. 컴포넌트 역할

| 컴포넌트 | LLM 사용 | 역할 |
|---------|:--------:|------|
| Preflight | X | 설정 검증, API 상태 확인, 실행 조건 확정 |
| Agent 1 | O (1회/토픽) | 한국어 서술 -> 영문 개념 + 키워드 + 임베딩 텍스트 (소스 독립적) |
| SourceAdapter (arXiv) | X | QueryBuilder로 쿼리 생성, 실행, 재시도, truncation 감지 |
| Hybrid Filter | X | 규칙 필터(카테고리/키워드) 후 cap(2000편) 후 임베딩 정렬(optional). 상위 200편 선택 |
| Agent 2 | O (배치) | base_score(0~100) + flags. 가산 안 함 |
| Ranker | X | 결정론적 가산 + final_score + 정렬 + 임계값 완화 |
| Clusterer | X | 유사 논문 그룹핑. 임베딩 없으면 비활성화 |
| Agent 3 | O (배치) | 한국어 요약/근거/인사이트 |
| Output | X | HTML/MD/JSON 생성, Issue upsert, 알림 전송 |

## 3-3. 플러그인 경계

교체 가능한 어댑터:
- SourceAdapter: arXiv(기본). 내부에 QueryBuilder 포함 (Agent 1 키워드 → 소스별 쿼리 DSL). 향후 Semantic Scholar 등 추가 시 새 어댑터 + QueryBuilder만 구현
- Notifier: Discord webhook / Telegram bot. 토픽별 선택

고정 코어:
- 파이프라인 순서: Preflight -> Collect -> Filter -> Score -> Rank -> Render -> Notify
- 데이터 모델: Paper, Evaluation, RunMeta
- 스코어링: 가중치 공식, 가산 규칙, 임계값 정책

---

# 4. 데이터 모델

## 4-1. Paper (내부 표준)

| 필드 | 타입 | 설명 |
|------|------|------|
| source | string | "arxiv" 등 |
| native_id | string | 소스 고유 ID (arXiv: "2602.12345") |
| paper_key | string | "{source}:{native_id}" (유일키) |
| canonical_id | string? | DOI (있으면). 교차 소스 dedup용 |
| url | string | 랜딩 페이지 URL |
| title | string | 논문 제목 |
| abstract | string | 초록 |
| authors | string[] | 저자 목록 |
| categories | string[] | 카테고리 |
| published_at_utc | datetime | 제출/게시 시각 (UTC) |
| updated_at_utc | datetime? | 업데이트 시각 |
| pdf_url | string? | PDF 링크 |
| has_code | boolean | 코드 공개 여부 (정규식 정본) |
| has_code_source | string | "regex" / "llm" / "both" / "none" |
| code_url | string? | GitHub 등 코드 링크 |
| comment | string? | arXiv comment 필드 (코드 링크, 페이지 수 등 포함) |

유일키 정책:
- 같은 소스 내: paper_key로 dedup
- 교차 소스: canonical_id(DOI) 우선, 없으면 normalized_title + first_author (4-5절 참조)

### 데이터 모델 → DB 테이블 매핑

| 데이터 모델 | DB 테이블명 | 비고 |
|------------|-----------|------|
| Paper | papers | 수집된 논문 원본. 365일 TTL (Dedup 장기 소스 + arXiv 재호출 방지) |
| Evaluation | paper_evaluations | run별 평가 스냅샷. 90일 purge |
| RunMeta | runs | 실행 메타. 90일 purge |
| QueryStats | query_stats | 쿼리별 통계. 90일 purge |
| RemindTracking | remind_tracking | 다시 보기 추적. 별도 테이블 (4-6절 참조). 대응 paper_evaluations 삭제 시 함께 정리 |

### papers 테이블 스키마

| 컬럼 | 타입 | 제약 | 설명 |
|------|------|------|------|
| paper_key | TEXT | PK | "{source}:{native_id}" |
| source | TEXT | NOT NULL | "arxiv" 등 |
| native_id | TEXT | NOT NULL | 소스 고유 ID |
| canonical_id | TEXT | NULL | DOI (교차 소스 dedup용) |
| url | TEXT | NOT NULL | 랜딩 페이지 URL |
| title | TEXT | NOT NULL | 논문 제목 |
| abstract | TEXT | NOT NULL | 초록 |
| authors | TEXT | NOT NULL | JSON 배열 직렬화 |
| categories | TEXT | NOT NULL | JSON 배열 직렬화 |
| published_at_utc | TEXT | NOT NULL | ISO 8601 |
| updated_at_utc | TEXT | NULL | ISO 8601 |
| pdf_url | TEXT | NULL | PDF 링크 |
| has_code | INTEGER | NOT NULL DEFAULT 0 | 0/1 |
| has_code_source | TEXT | NOT NULL DEFAULT 'none' | "regex"/"llm"/"both"/"none" |
| code_url | TEXT | NULL | GitHub 등 코드 링크 |
| comment | TEXT | NULL | arXiv comment 필드 |
| first_seen_run_id | INTEGER | NOT NULL | 최초 수집 실행 ID |
| created_at | TEXT | NOT NULL | 레코드 생성 시각 |

papers 테이블은 365일 TTL로 관리한다. 주간 purge 시 365일 초과 레코드를 삭제한다.

## 4-2. Evaluation

복합 PK: (run_id, paper_key).

Agent 2가 discard: true로 판정한 논문도 discarded=true로 기록한다 (폐기 통계 추적용).

| 필드 | 타입 | 설명 |
|------|------|------|
| run_id | integer | 실행 식별자 |
| paper_key | string | Paper 참조 |
| embed_score | float? | 임베딩 유사도 (0~1). 임베딩 미사용 시 NULL |
| llm_base_score | integer | Agent 2 순수 점수 (0~100) |
| flags | object | is_edge, is_realtime, mentions_code, is_metaphorical |
| bonus_score | integer? | Ranker가 계산한 가산점. discarded 시 NULL |
| final_score | float? | 최종 점수. discarded 시 NULL |
| rank | integer? | 순위. discarded 시 NULL |
| tier | integer? | 1 또는 2. discarded 시 NULL |
| discarded | boolean | 폐기 여부 |
| score_lowered | boolean? | 임계값 완화로 포함된 논문. discarded 시 NULL |
| multi_topic | string? | 다른 토픽에도 등장 시 태그 |
| is_remind | boolean | 다시 보기 탭으로 노출된 논문인지 |
| summary_ko | string? | 한국어 요약. discarded 시 NULL |
| reason_ko | string? | 선정 근거. discarded 시 NULL |
| insight_ko | string? | 활용 인사이트. discarded 시 NULL |
| brief_reason | string? | Agent 2 한국어 1문장 근거 |
| prompt_ver_score | string | Agent 2 프롬프트 버전 |
| prompt_ver_summ | string? | Agent 3 프롬프트 버전. discarded 시 NULL |

discarded=true인 레코드는 embed_score(임베딩 사용 시), llm_base_score, flags, brief_reason, prompt_ver_score만 유효하다. embed_score는 Hybrid Filter([4])에서 이미 계산된 후 Agent 2([5])에서 discard가 결정되므로 값이 존재한다. Ranker/Agent 3을 거치지 않으므로 bonus_score, final_score, rank, tier, score_lowered, summary_ko, reason_ko, insight_ko, prompt_ver_summ은 NULL이다.

## 4-3. RunMeta

| 필드 | 타입 | 설명 |
|------|------|------|
| run_id | integer | 자동 증가 |
| topic_slug | string | 토픽 식별 |
| window_start_utc | datetime | 검색 시작 (UTC) |
| window_end_utc | datetime | 검색 종료 (UTC) |
| display_date_kst | string | 표시용 날짜 (KST) |
| embedding_mode | string | "disabled" / "en_synthetic" |
| scoring_weights | string | 사용된 가중치 JSON |
| detected_rpm | integer | Preflight에서 감지한 RPM |
| detected_daily_limit | integer? | 감지한 일일 한도 |
| response_format_supported | boolean | 모델의 response_format 지원 여부 |
| prompt_versions | string | agent1/agent2/agent3 버전 JSON |
| topic_override_fields | string | 사용된 선택 필드 목록 JSON (빈 배열 가능) |
| total_collected | integer | 수집 수 |
| total_filtered | integer | 필터 후 |
| total_scored | integer | 점수화 수 (discard 제외) |
| total_discarded | integer | Agent 2가 discard한 수 |
| total_output | integer | 최종 출력 수 |
| threshold_used | integer | 실제 적용된 임계값 (60/50/40) |
| threshold_lowered | boolean | 임계값 완화 발동 여부 |
| status | string | running / completed / failed |
| errors | string? | 에러 로그 |

## 4-4. QueryStats

| 필드 | 타입 | 설명 |
|------|------|------|
| run_id | integer | 실행 참조 |
| query_text | string | 실제 쿼리 문자열 |
| collected | integer | 수집 수 |
| total_available | integer? | OpenSearch totalResults |
| truncated | boolean | 실제 truncation 여부 |
| retries | integer | 재시도 횟수 |
| duration_ms | integer | 소요 시간 |
| exception | string? | 발생 예외 |

## 4-5. 중복 방지 (Dedup) 전략

구성 요소:
1. seen_items.jsonl (Dedup 캐시): paper_key + topic_slug + date. 매일 커밋. 30일 롤링.
2. papers 테이블 (장기 소스): 수집된 논문 원본. 365일 TTL. arXiv API 재호출 방지용.

Dedup은 (paper_key, topic_slug) 기준으로 작동한다. 같은 논문이 여러 토픽에서 각각 독립 평가된다 (multi_topic 태깅의 전제).

### 2계층 Dedup

Dedup은 두 계층으로 분리된다:

**계층 1: in_run (항상 ON, 모드 무관)**
- 같은 run 안에서 다중 쿼리가 동일 논문을 반환할 수 있다.
- run 시작 시 빈 set 생성. 수집된 paper_key를 set에 추가. 이미 있으면 스킵.
- LLM 호출 낭비를 방지하는 방어 장치이며, 비활성화할 수 없다.

**계층 2: cross_run (모드에 따라 동작)**
- seen_items.jsonl을 읽어 "이전 실행에서 처리된 논문"을 스킵한다.
- 모드에 따라 동작이 달라진다 (아래 표 참조).

체크 순서 (skip_recent 모드):
1. in_run set에서 paper_key 조회 → 히트면 스킵
2. seen_items 메모리 set에서 (paper_key, topic_slug) 조회 → 히트면 스킵 (이 토픽에서 최근 처리됨)
3. 미스면 → in_run set에 즉시 추가 (수집 시점) → 평가 대상으로 진행
   - papers 테이블에서 paper_key 조회 → 히트면 메타데이터 재활용 (arXiv 재호출 불필요)
   - papers에 없으면 → 신규. papers INSERT
4. 실행 완료 후 ([11]): seen_items에 (paper_key, topic_slug, date) 추가 (dedup=none이면 건너뜀)

### Dedup 모드

| 모드 | in_run | cross_run (seen_items 읽기) | seen_items 쓰기 | 용도 |
|------|:------:|:--------------------------:|:---------------:|------|
| skip_recent (기본) | ON | ON | ON | 자동 실행, 일반 수동 실행 |
| none | ON | OFF | OFF | 과거 기간 재검증, 디버깅 |

dedup=none에서는 seen_items를 읽지도 쓰지도 않는다. 따라서 백필 실행이 이후 일일 실행의 cross_run dedup에 영향을 주지 않는다. in_run dedup은 항상 작동하므로 같은 run 내 중복 LLM 호출은 방지된다.

자동 실행(cron)은 항상 skip_recent. 수동 실행(workflow_dispatch, CLI, 로컬 UI)에서 `--dedup=none` 지정 가능.

향후 교차 소스 dedup 우선순위: DOI > paper_key(같은 소스) > normalized_title + first_author(최후 수단).

## 4-6. 다시 보기 추적

별도 remind_tracking 테이블에서 토픽별 recommend_count를 관리한다. papers 테이블은 수집된 논문 원본 저장 역할에 집중하고, 토픽별 평가 추적은 분리한다 (Evaluation은 run별 스냅샷이므로 cross-run 추적에 부적합).

| 컬럼 | 타입 | 제약 | 설명 |
|------|------|------|------|
| paper_key | TEXT | NOT NULL | Paper 참조 |
| topic_slug | TEXT | NOT NULL | 토픽 |
| recommend_count | INTEGER | NOT NULL DEFAULT 0 | 다시 보기 노출 횟수 (0/1/2) |
| last_recommend_run_id | INTEGER | NOT NULL | 마지막 노출 실행 ID |

복합 PK: (paper_key, topic_slug).

---

# 5. 입력(TopicSpec) 설계

## 5-1. 토픽 정의

config.yaml의 topics 배열에 토픽을 정의한다. description(필수)만 채우면 Agent 1이 나머지를 자동 생성한다. 검색 정밀도를 직접 제어하고 싶으면 선택 필드를 추가하면 되고, Agent 1은 이를 우선 반영한다.

## 5-2. 필드 정의

| 필드 | 필수 | 타입 | 설명 |
|------|:----:|------|------|
| slug | O | string | 토픽 식별자. ASCII, 하이픈 허용. 파일명/URL에 사용 |
| name | O | string | 한국어 표시명 |
| description | O | string | 한국어 자유 서술 (100~300자). 두서없이 써도 된다 |
| arxiv_categories | O | string[] | arXiv 카테고리. 규칙 필터 + Agent 1 힌트로 사용 |
| must_concepts_en | - | string[] | 반드시 포함할 영문 개념. Agent 1이 이를 concepts에 포함 |
| should_concepts_en | - | string[] | 가급적 포함할 영문 개념. Agent 1이 cross_domain_keywords에 반영 |
| must_not_en | - | string[] | 반드시 제외할 영문 키워드. Agent 1의 exclude_keywords에 병합 + 규칙 필터에서도 직접 적용 |
| notify | O | object | 알림 채널 설정 (provider, channel_id, secret_key) |

## 5-3. 예시

### 최소 구성 (선택 필드 없음)

Agent 1이 description을 분석해서 영문 학술 키워드와 임베딩용 영문 텍스트를 자동 생성한다. 이를 기반으로 QueryBuilder가 arXiv 쿼리를 만든다.

```yaml
topics:
  - slug: "ai-sports-device"
    name: "AI 스포츠 디바이스/플랫폼"
    description: |
      자동으로 경기 촬영하고 주요 장면이나 개인별 하이라이트 편집하고
      영상 보정하고 자세/전략 분석하는 AI 스포츠 디바이스 만들건데
      실시간으로 돌아가야 하고 모바일에서도 써야 함. 경기 영상을
      SNS처럼 공유하는 플랫폼도 관심 있음
    arxiv_categories: [cs.CV, cs.AI, cs.MM, cs.LG, eess.IV]
    notify:
      provider: "discord"
      channel_id: "work-research"
      secret_key: "WORK_RESEARCH"
```

### 정밀 제어 (선택 필드 사용)

dry-run으로 Agent 1 결과를 확인한 뒤, 불필요한 키워드를 must_not_en에, 누락된 개념을 must_concepts_en에 추가하는 방식으로 점진적으로 보강한다.

```yaml
topics:
  - slug: "ai-sports-device"
    name: "AI 스포츠 디바이스/플랫폼"
    description: |
      자동으로 경기 촬영하고 주요 장면이나 개인별 하이라이트 편집하고
      영상 보정하고 자세/전략 분석하는 AI 스포츠 디바이스 만들건데
      실시간으로 돌아가야 하고 모바일에서도 써야 함
    arxiv_categories: [cs.CV, cs.AI, cs.MM, cs.LG, eess.IV]
    must_concepts_en:
      - "sports analytics"
      - "camera automation"
    should_concepts_en:
      - "tracking"
      - "video understanding"
    must_not_en:
      - "medical imaging"
      - "satellite"
    notify:
      provider: "discord"
      channel_id: "work-research"
      secret_key: "WORK_RESEARCH"
```

## 5-4. Agent 1과의 연동

| 선택 필드 | Agent 1 동작 |
|----------|-------------|
| must_concepts_en | concepts에 반드시 포함. 추가 개념은 Agent 1이 자체 생성 |
| should_concepts_en | cross_domain_keywords에 병합. Agent 1이 추가 확장 |
| must_not_en | exclude_keywords에 병합. Agent 1이 추가 부정 키워드도 생성 |
| 모두 미지정 | Agent 1이 description + arxiv_categories만으로 전부 자동 생성 |

선택 필드는 Agent 1의 출력을 제약/보강하는 역할이지, Agent 1을 우회하는 것이 아니다. 최종 키워드와 임베딩 텍스트는 항상 Agent 1이 생성하고, 이를 기반으로 QueryBuilder가 소스별 쿼리를 생성한다.

## 5-5. 캐싱

description + 선택 필드 전체의 해시 기준. 어떤 필드라도 변경되면 캐시 무효화. 만료 30일. data/keyword_cache.json.

## 5-6. dry-run을 통한 점진적 보강

1. 선택 필드 없이 최소 구성으로 시작
2. dry-run 실행 (택일):
   - Actions: `gh workflow run paper-scout.yml -f mode=dry-run`
   - 로컬 CLI: `paper-scout dry-run`
   - 로컬 UI: 수동 검색 페이지에서 dry-run 실행 (18절)
3. Agent 1이 생성한 concepts, exclude_keywords와 QueryBuilder가 만든 쿼리를 확인
4. 필요하면 must_concepts_en, must_not_en 등을 config.yaml에 추가 (로컬 UI에서 토픽 수정 가능)
5. 다시 dry-run으로 확인 후 full 실행

---

# 6. 소스 어댑터 설계

## 6-1. 공통 인터페이스

입력: Agent 1 출력(concepts, cross_domain_keywords, exclude_keywords) + 시간 윈도우(UTC) + 설정(max_results 등)
출력: Paper[] (정규화)

각 SourceAdapter는 내부에 QueryBuilder를 갖고, Agent 1의 소스 독립적 키워드를 자신의 쿼리 DSL로 변환한다. Agent 1은 소스별 쿼리 문법을 알 필요가 없다.

## 6-2. arXiv 어댑터

### QueryBuilder

Agent 1이 출력한 concepts + cross_domain_keywords + exclude_keywords + arxiv_categories를 조합하여 arXiv 쿼리 DSL(ti:/abs:/cat:)을 생성한다.

생성 전략:
- 카테고리별 넓은 쿼리 (recall 우선): `cat:cs.CV AND abs:"sports"`
- 키워드 조합 좁은 쿼리 (precision 우선): `abs:"camera selection" AND abs:"real-time"`
- 교차 도메인 쿼리: 서로 다른 concept에서 키워드를 교차 조합
- 목표: 15~25개 쿼리. 넓은 것부터 좁은 것까지.

QueryBuilder는 결정론적 코드이며 LLM을 사용하지 않는다.

### 수집 정책

- Client 설정: num_retries=3, delay_seconds=3. 별도 sleep 없음 (이중 딜레이 방지).
- 쿼리: QueryBuilder가 생성한 15~25개 arXiv 쿼리 순차 실행.
- 날짜 필터: submittedDate 범위. 윈도우 양쪽 +-30분 버퍼.
- 기본 max_results_per_query: 200.
- 동일 실행 내 같은 쿼리 결과는 메모리 캐시.

### 재시도 정책

| 에러 유형 | 재시도 | 백오프 | 실패 시 |
|----------|:------:|--------|---------|
| 네트워크/5xx | 3회 | 지수 + jitter | 쿼리 스킵, 로그 |
| UnexpectedEmptyPageError | 2회 | 3초, 9초 | 쿼리 스킵, 로그 |

### Truncation 감지

- 우선: OpenSearch totalResults 파싱. totalResults > collected이면 truncated=true.
- 폴백: 라이브러리가 totalResults를 미노출하면, collected == max_results를 경고 수준 로깅 (오탐 가능 표기).
- QueryStats 테이블에 기록.

### 코드 공개 감지

수집 단계에서 정규식으로 초록(abstract) + 코멘트(comment)를 검사. 패턴:
- github.com/ 뒤에 경로가 있는 URL
- "code" + "available" (사이 공백/is 허용)
- "our code", "our implementation"
- "open source code", "open-source implementation"
- "repository:" + URL

URL 추출: 닫힘 괄호/공백/따옴표를 끊는 패턴.

정규식이 정본. Agent 2의 mentions_code는 보조 신호. 최종 has_code = regex OR llm. has_code_source에 출처 기록.

### arXiv 특성 참고사항

- 평일 미국 동부시간 14:00 제출 마감, 20:00경 공지. 공지 시각은 KST 09:00(여름, EDT) ~ 10:00(겨울, EST).
- 시스템 크론 및 검색 윈도우 설계는 9-2절 참조.
- arXiv 공지가 지연되면 해당 논문은 다음 실행에서 수집됨 (seen_items로 중복 제거).
- 주말/미국 공휴일에는 공지가 없음. 증분 윈도우(9-2절) 덕분에 월요일 실행 시 금요일 공지분까지 자동으로 커버된다.
- 3초 요청 간 딜레이 권고 준수 (Client delay_seconds=3).
- arXiv API 이용 고지 필수 (리포트 하단, README). 문구는 14-5절 참조.

# 7. LLM 에이전트 설계

## 7-0. 공통 규칙

모든 Agent에 일괄 적용:

1. OpenRouter 호출: OpenAI Python SDK 사용. base_url을 OpenRouter로 지정. OpenRouter 전용 파라미터(reasoning 등)는 extra_body로 전달.
2. 앱 식별 헤더: HTTP-Referer, X-Title 포함.
3. reasoning 필드 무시: content만 사용.
4. think 블록 제거: content에서 항상 수행. 미종결 태그(</think> 없음)도 제거.
5. 구조화 출력: Preflight에서 모델의 response_format 지원 여부 확인 (data/model_caps.json 캐시, 7일 유효). 지원 시 response_format: { type: "json_object" } 우선 적용. 미지원 시 프롬프트에 "JSON만 출력" 지시 + 파싱 방어.
6. 파싱 방어 (response_format 미사용 시): think 제거 -> 브라켓 밸런싱 추출 -> json.loads -> json_repair 폴백.
7. 파싱 실패 원문: tmp/debug/에 저장. 실행 성공 시 tmp/ 전체 삭제. 실패 시 tmp/debug/를 Actions artifact로 업로드 후 삭제 (15-3 참조).

## 7-1. Agent 1: 키워드 확장

### 설정

| 항목 | 값 | 이유 |
|------|-----|------|
| effort | high | 한국어->영문 학술 변환이 핵심. 여기서 틀리면 전체가 틀어짐 |
| max_tokens | 2048 | 개념 목록 + 임베딩 텍스트 |
| temperature | 0.3 | 창의적이되 안정적 |

### 입력
- topic.description (한국어 자유 서술, 필수)
- topic.arxiv_categories (힌트)
- topic.must_concepts_en (선택. 있으면 concepts에 필수 포함)
- topic.should_concepts_en (선택. 있으면 cross_domain_keywords에 병합)
- topic.must_not_en (선택. 있으면 exclude_keywords에 병합)

### 출력 구조

```json
{
  "concepts": [
    {"name_ko": "자동 촬영", "name_en": "automatic cinematography", "keywords": ["camera selection", "view planning"]}
  ],
  "cross_domain_keywords": ["neural radiance field", "attention mechanism"],
  "exclude_keywords": ["medical imaging", "satellite"],
  "topic_embedding_text": "Automatic cinematography and camera selection for sports broadcasting, real-time highlight detection, pose estimation, edge deployment for mobile devices, video understanding and action recognition in athletic contexts"
}
```

- Agent 1은 소스에 독립적인 개념/키워드/임베딩 텍스트만 출력한다. 소스별 쿼리 DSL(arXiv ti:/abs:/cat: 등)은 생성하지 않는다.
- 소스별 쿼리 생성은 SourceAdapter 내부의 QueryBuilder가 담당한다 (6-2절 참조).
- topic_embedding_text: concepts + cross_domain_keywords를 합성한 영문 단락. 임베딩 쿼리로 사용. 한국어 description을 직접 임베딩하지 않음 (언어 공간 불일치 방지).

### 캐싱

description + 선택 필드(must_concepts_en, should_concepts_en, must_not_en) 전체의 해시 기준. 어떤 필드라도 변경되면 캐시 무효화. 만료 30일. data/keyword_cache.json.

### 프롬프트

```
[시스템]
You are an expert research librarian specializing in computer science
and engineering papers. Analyze the project description and generate
comprehensive search concepts and keywords in English.

Also generate a "topic_embedding_text" field: a single English paragraph
combining all key concepts and keywords, suitable for semantic similarity
matching against paper abstracts.

[사용자]
## 프로젝트 설명
{topic.description}

## 참고 카테고리 (힌트)
{topic.arxiv_categories}

## 사용자 지정 제약 (있을 때만 포함)
- 필수 포함 개념: {topic.must_concepts_en}
- 권장 포함 개념: {topic.should_concepts_en}
- 제외 키워드: {topic.must_not_en}

## 작업 지시
1. 개념 분해: 독립 기술 도메인 5~10개
2. 학술 용어 매핑: 도메인별 영문 키워드 5~10개
   (동의어, 상위/하위 개념, 방법론명 포함)
3. 교차 도메인 확장: 도메인 간 복합 키워드
4. 부정 키워드: 혼동 가능한 무관 키워드
5. topic_embedding_text: 위 키워드를 합성한 영문 단락 1개

소스별 검색 쿼리(arXiv 문법 등)는 생성하지 마라. 키워드만 출력하라.

## 출력: JSON만. 설명 없이.
```

버전: agent1-v3

### 실패 시 폴백

파싱 2회 실패 -> arxiv_categories 기반 기본 키워드 생성 (카테고리명을 키워드로 사용).

---

## 7-2. Agent 2: 점수화

### 설정

| 항목 | 값 | 이유 |
|------|-----|------|
| effort | low | 초록 읽고 점수+flag 부여. 복잡한 추론 불필요 |
| max_tokens | 2048 | 10편 배치 응답 |
| temperature | 0.2 | 일관성 우선 |
| 배치 크기 | 10편 | |

### 출력 구조

```json
[{
  "index": 1,
  "base_score": 82,
  "flags": {
    "is_edge": true,
    "is_realtime": true,
    "mentions_code": false,
    "is_metaphorical": false
  },
  "discard": false,
  "brief_reason": "멀티뷰 스포츠 영상에서 실시간 하이라이트 검출"
}]
```

핵심 규칙:
- base_score는 순수 관련성만. 가산/감점 없이.
- flags는 사실 여부 판단만. 점수에 반영하지 않음.
- discard 조건: is_metaphorical=true 또는 base_score < 20. 이 두 경우만 폐기한다.
- 20~59점 논문은 discard하지 않는다. 낮은 점수 그대로 Ranker에 전달되어 임계값(60→50→40) 완화 시 선정 후보가 된다.
- discard: true인 논문은 Evaluation에 discarded=true로 기록된 후 Ranker에 전달되지 않는다.
- brief_reason은 한국어 1문장. Agent 3 요약 전 빠른 판단용.
- 가산은 Ranker가 결정론적으로 수행.

### 프롬프트

```
[시스템]
You are a paper evaluator. Output base_score and boolean flags.
Do NOT apply bonuses. Output ONLY raw JSON.

[사용자]
## 프로젝트
{topic.description}

## 채점 (base_score, 가산 없이 순수 관련성)
- 90~100: 핵심 기술을 직접 다루는 논문
- 70~89: 적용 가능한 기술/방법론 포함
- 50~69: 간접 관련/참고
- 20~49: 관련 약함 (discard: false, 점수만 부여)
- 20 미만: 무관 (discard: true)

## flags (사실 여부만 판단, 점수에 반영하지 말 것)
- is_edge: 엣지/모바일 디바이스에서 실행 가능한 경량 모델인가
- is_realtime: 실시간 또는 near-realtime 처리가 가능한가
- mentions_code: 초록에서 코드 공개/github 링크를 언급하는가
- is_metaphorical: "sport"/"game" 등이 비유적으로 쓰였는가 (true면 discard)

## 논문 목록
{10편: 번호 + 제목 + 초록}

## 출력
[{"index":1,"base_score":82,"flags":{...},"discard":false,"brief_reason":"한국어 1문장"}]
```

버전: agent2-v2

### 검증

- 출력 개수 == 입력 개수인지 확인. 누락 시 누락분만 재호출.
- base_score 범위 0~100. 벗어나면 클램프.
- 파싱 2회 실패 -> 해당 배치 스킵, 로그.

---

## 7-3. Agent 3: 요약/인사이트

### 설정

| 항목 | 값 |
|------|-----|
| effort | low |
| max_tokens | 4096 |
| temperature | 0.4 |
| 배치: Tier 1 | 5편 |
| 배치: Tier 2 | 10편 |

### 수치 정책

- 허용: fps, latency, 추론 속도, 파라미터 수, 모델 크기, 실시간, 모바일, 엣지, on-device, 경량화
- 금지: mAP, BLEU, ROUGE, FID, LPIPS, 벤치마크 테이블, 수식

### Tier 1 프롬프트 (상위 30편)

```
[시스템]
You are a technical writer for a startup CEO. Write in simple Korean.
No thinking, no analysis preamble. Output ONLY raw JSON.

[사용자]
## 프로젝트 컨텍스트
{topic.description}

## 작성 규칙
1. "이 논문은 ~하는 방법을 제안한다. 핵심은 ~이다." 형식
2. 기존 문제 -> 해결책 -> 결과 순서
3. 5초 안에 활용 가능 여부 판단 가능해야 함
4. 선정 근거: 왜 우리 프로젝트에 중요한지
5. 활용 인사이트: 구체적 적용 방법
6. 허용 수치: fps, latency, 추론 속도, 파라미터 수
7. 금지: mAP, BLEU, ROUGE 등 학술 벤치마크 수치 나열

## 논문 목록
{5편: 번호 + 제목 + 초록 + base_score + brief_reason}

## 출력
[{"index":1,"summary_ko":"300~500자","reason_ko":"~150자","insight_ko":"~150자"}]
```

버전: agent3-tier1-v1

### Tier 2 프롬프트 (31~100위)

Tier 1과 동일하되: summary_ko 200자 이내, reason_ko 한 줄, insight_ko 없음.

버전: agent3-tier2-v1

---

# 8. 스코어링 및 랭킹

## 8-1. 결정론적 가산 (Ranker)

Agent 2의 flags 기반으로 Ranker가 코드에서 계산. LLM은 관여하지 않음.

| flag | 가산 | 조건 |
|------|:----:|------|
| is_edge | +5 | 엣지/모바일 경량 모델 |
| is_realtime | +5 | 실시간 처리 가능 |
| has_code | +3 | 코드 공개 (regex OR llm) |

```
bonus = (5 if is_edge) + (5 if is_realtime) + (3 if has_code)
llm_adjusted = min(base_score + bonus, 100)
```

규칙 변경 시 프롬프트 수정 없이 config/코드만 변경.

## 8-2. 최종 점수 공식

### 임베딩 사용 시 (en_synthetic 모드)

```
final_score = 0.55 * llm_adjusted + 0.35 * (embed_score * 100) + 0.10 * recency_score
```

### 임베딩 미사용 시 (disabled 모드)

```
final_score = 0.80 * llm_adjusted + 0.20 * recency_score
```

### recency_score

기준일: 검색 윈도우의 window_end (자동 실행은 당일, 수동 실행은 지정 종료일). 수동 실행에서 과거 날짜를 지정해도 해당 윈도우 내에서 공정하게 평가된다.

경과일 계산: `floor((window_end - published_at_utc) / 24시간)`. 정수 날짜 차이(버림).

| 경과일 (window_end 기준) | 점수 |
|:------:|:----:|
| 0일 (당일) | 100 |
| 1일 | 90 |
| 2일 | 80 |
| 3일 | 70 |
| 4일 | 60 |
| 5일 | 50 |
| 6일 | 40 |
| 7일+ | 30 |

## 8-3. 임베딩 설계

### 언어 정렬 원칙

임베딩 비교는 동일 언어 공간에서만 수행한다.
- 토픽: Agent 1이 생성한 topic_embedding_text (영문)
- 논문: title + abstract (영문)
- 모델: sentence-transformers/all-MiniLM-L6-v2 (영문)
- 한국어 description을 직접 임베딩하지 않음.

### embed_score 정규화

cosine similarity를 `clamp(0, 1)`로 정규화한 후 embed_score에 저장한다. all-MiniLM-L6-v2 + 영문 학술 텍스트 조합에서는 음수가 관측되지 않으므로 clamp는 사실상 no-op이다. 모델을 교체할 경우 cosine 분포가 달라질 수 있으며, 그때 정규화 전략(예: `(cos+1)/2`)을 재검토해야 한다. 클러스터링(cosine ≥ 0.85)에도 동일한 정규화 후 값을 사용한다.

### embedding_mode (RunMeta에 기록, config: embedding.mode)

- disabled: 임베딩 모듈 미설치. 규칙 기반만.
- en_synthetic: Agent 1 영문 텍스트 기반. 기본값.

토픽 임베딩은 최초 1회 계산 후 캐시 (data/topic_embeddings.npy). Agent 1 캐시와 동일한 해시(description + 선택 필드)를 사용하며, Agent 1 출력이 갱신되면 임베딩도 재계산된다.

### 임베딩 미설치 시 자동 처리

- import 실패 -> embedding_mode = disabled
- 가중치 자동 전환 (0.80 llm + 0.20 recency)
- 클러스터링 자동 비활성화
- RunMeta에 모드 기록

## 8-4. 동적 임계값 완화 (자동 백필 없음)

자동 실행에서 결과가 적어도 검색 윈도우를 자동 확장(=백필)하지 않는다. 대신 점수 임계값만 완화한다. 수동 실행에서 사용자가 명시적으로 날짜 범위를 지정하는 것은 백필이 아니라 별도 수집이다 (9-2절 참조).

Agent 2의 discard 기준(base_score < 20 또는 is_metaphorical)은 임계값 완화보다 훨씬 낮게 설정되어 있다. 따라서 20~59점 논문은 Ranker에 도달하며, 완화 시 50점/40점 대역의 논문이 실제로 선정될 수 있다.

| 단계 | 조건 | 동작 |
|:----:|------|------|
| 기본 | -- | final_score 60점 이상만 선정 |
| 1차 완화 | final_score 60점 이상이 5편 미만 | 50점으로 낮춤. [완화] 태그 |
| 2차 완화 | final_score 50점 이상이 5편 미만 | 40점으로 낮춤. [완화] 태그 |

최대 출력: 100편.

## 8-5. Tier 구분

| Tier | 순위 | 요약 수준 |
|:----:|------|----------|
| 1 | 1~30위 | 풀 요약 300~500자 + 근거 ~150자 + 인사이트 ~150자 |
| 2 | 31~100위 | 압축 요약 ~200자 + 한 줄 근거 |

## 8-6. 다시 보기

이전 실행에서 80점 이상이었던 논문을 다시 보기 탭에 추가 노출. 토픽별로 독립 운영한다 (토픽 A의 고득점 논문이 토픽 B에 나타나지 않음).

규칙:
- 같은 토픽의 이전 실행에서 final_score 80점 이상
- recommend_count < 2인 논문만 대상
- 노출할 때마다 recommend_count += 1
- 2회 노출 후 졸업 (더 이상 노출 안 함)
- 다시 보기 논문은 오늘 신규 논문과 별도 탭으로 분리
- 요약은 가장 최근 Evaluation의 summary_ko/reason_ko/insight_ko를 재사용 (Agent 3 재호출 없음)

---

# 9. 파이프라인 상세 프로세스

## 9-1. Preflight

실패를 조기 발견하여 전체 run을 낭비하지 않기 위한 단계.

| 순서 | 검증 항목 | 실패 시 |
|:----:|----------|---------|
| 1 | config.yaml 스키마 검증 | 즉시 중단 |
| 2 | TopicSpec 필드 검증 (필수 필드 + 선택 필드 타입) | 즉시 중단 |
| 3 | OpenRouter API 키 유효성 (GET /api/v1/key) | 즉시 중단 |
| 4 | RPM/일일 한도 감지. 실패 시 보수적 기본값 (RPM=10, daily=200) | 경고 후 계속 |
| 5 | response_format 지원 여부 확인. data/model_caps.json 캐시 (모델명+확인일, 7일 유효). 캐시 히트면 probe 스킵. 미스면 샘플 호출 1회. | 미지원이면 파싱 방어 모드 |
| 6 | 알림 채널 연결성 확인 (토픽별) | 경고 후 계속 |
| 7 | run_id 발급, 검색 윈도우(UTC) 확정. 토픽별 window_start 조회: RunMeta(DB) → data/last_success.json → 72h 폴백 (9-2절) | -- |
| 8 | 당일 usage 파일 로드 -> RateLimiter 초기화 | -- |

## 9-2. 검색 윈도우

### 증분 윈도우 (자동 실행)

고정 24시간이 아닌, 직전 성공 실행 기준 증분 윈도우를 사용한다. 주말/공휴일에 arXiv 공지가 없어도, 월요일 실행 시 금요일 공지분까지 자동으로 커버된다.

- window_start = 직전 성공 실행(같은 토픽)의 window_end_utc. 조회 우선순위:
  1. RunMeta(DB)에서 해당 토픽의 최신 completed run
  2. data/last_success.json (토픽별 last_success_window_end_utc, git 커밋)
  3. 72시간 폴백 (window_end - 72h)
- window_end = 당일 KST 11:00을 UTC로 변환
- 양쪽 +-30분 버퍼 추가
- 중복은 seen_items (topic별) dedup으로 상쇄

폴백 계층 설명:
- RunMeta(DB)는 actions/cache에 저장되므로 cache miss 시 사용할 수 없다.
- data/last_success.json은 main 브랜치에 커밋되므로 actions/cache/Release asset과 독립적으로 존속한다. 매 성공 실행 후 갱신 커밋.
- 갱신 규칙: `last_success_window_end_utc = max(기존값, 이번 run의 window_end_utc)`. 과거 범위 백필이 더 최신 값을 덮어쓰지 않도록 보호한다.
- 72시간 폴백은 첫 실행 또는 last_success.json이 없는 경우(초기 설정)에만 도달한다.
- 72시간이면 금요일 공지 → 월요일 실행까지 안전하게 커버

버퍼 근거: arXiv는 미국 동부시간(ET) 20:00에 공지하며, 이는 KST 09:00(여름) ~ 10:00(겨울)이다. KST 11:00 크론은 공지 후 최소 1시간 여유를 확보한다. +-30분 버퍼는 공지 직후 API 인덱싱 지연 및 제출/공지 경계의 누락을 흡수하기 위한 것이다.

### 수동 실행 (workflow_dispatch, CLI, 로컬 UI)

- 사용자가 date_from/date_to를 직접 지정
- date_from/date_to 모두 미지정 시: 자동 실행과 동일한 증분 윈도우
- 같은 +-30분 버퍼 적용

## 9-3. 토픽 루프 상세

```
[1] runs 레코드 생성 (status: running)

[2] Agent 1: 키워드 확장
    캐시 히트 -> LLM 0회
    캐시 미스 -> LLM 1회 (effort: high)
    출력: concepts + cross_domain_keywords + exclude_keywords + topic_embedding_text (소스 독립적)
    실패: 카테고리 기본 키워드 폴백

[3] arXiv 수집
    QueryBuilder: Agent 1 출력(concepts, cross_domain_keywords) + arxiv_categories -> arXiv 쿼리 DSL 15~25개
    쿼리 순차 실행
    Client(delay_seconds=3), 별도 sleep 없음
    EmptyPage: 2회 재시도(3s, 9s) -> 스킵
    truncation: totalResults 기반 판정
    쿼리별 통계 -> QueryStats 테이블
    코드 감지 (정규식, 수집 단계: abstract + comment)
    Dedup 계층 1 (in_run): run 내 paper_key set 체크 -> 히트면 스킵 -> 미스면 즉시 set에 추가 (항상 ON)
    Dedup 계층 2 (cross_run): seen_items에서 (paper_key, topic_slug) 체크 -> 히트면 스킵 (dedup=none이면 이 계층 건너뜀)
    미스면 papers에서 메타데이터 재활용 또는 신규 INSERT
    UTC 윈도우 필터링 (window_start_utc ≤ published_at_utc ≤ window_end_utc, 버퍼 포함 범위로 판정. display_date_kst는 리포트 제목/파일명 등 표시용으로만 사용하며, 필터링 근거로 쓰지 않는다)

[4] 하이브리드 필터링
    1단계 [규칙 필터]:
      - 부정 키워드 매칭 -> 즉시 제외
        (출처: Agent 1의 exclude_keywords + 사용자의 must_not_en. 합집합)
      - 카테고리 매칭 OR 필수 키워드 매칭 -> 통과
        (카테고리: topic.arxiv_categories, 키워드: Agent 1의 concepts 내 keywords)
      - 결과: 규칙 통과 후보
    1.5단계 [방어 cap]:
      - 규칙 통과 후보가 pre_embed_cap(기본 2,000편)을 초과하면 published_at_utc 내림차순으로 상위 2,000편만 다음 단계에 전달
    2단계 [임베딩 정렬] (optional):
      - 규칙 통과 후보에 대해 topic_embedding_text(영문) vs title+abstract(영문) cosine 유사도 계산
      - 유사도 내림차순 정렬 후 상위 200편 선택
    임베딩 미사용 시:
      - 규칙 통과 후보를 published_at_utc 내림차순 정렬 후 상위 200편 선택
    import 실패 시 규칙만 + published_at_utc 정렬

[5] Agent 2: 점수화 (effort: low)
    10편/배치
    호출 간 딜레이: 60 / detected_rpm + 0.5초 (동적)
    base_score + flags 출력
    has_code 병합: regex OR mentions_code
    discard 조건: is_metaphorical=true 또는 base_score < 20. 해당 논문만 Evaluation에 discarded=true 기록 후 Ranker에 전달하지 않음
    20~59점 논문은 discard 없이 Ranker에 전달 (임계값 완화의 후보)
    파싱 실패 원문 -> tmp/debug/
    누락분 재호출 (1회)

[6] Ranker
    결정론적 가산 (flags 기반)
    final_score 계산 (임베딩 on/off 분기)
    내림차순 정렬
    임계값 완화 (60 -> 50 -> 40, 최대 2단계)
    Tier 배정: 1~30 = Tier 1, 31~100 = Tier 2
    rank, tier, bonus_score, score_lowered 기록

[7] 클러스터링 (임베딩 있을 때만)
    cosine 유사도 0.85 이상 -> 동일 클러스터
    클러스터별 대표 = 최고 점수 논문
    활용: HTML 리포트에서 "같은 클러스터" 앵커 링크 표시 + JSON에 clusters 필드 포함
    랭킹/요약 단계에는 영향 없음 (순수 표시용)

[8] Agent 3: 요약 (effort: low)
    Tier 1: 5편/배치
    Tier 2: 10편/배치
    파싱 실패 원문 -> tmp/debug/

[9] 다시 보기 선정
    이전 실행에서 80점+ && recommend_count < 2인 논문 조회
    recommend_count += 1
    가장 최근 Evaluation의 요약 재사용 (Agent 3 재호출 없음)

[10] 리포트 생성 -> tmp/reports/
     {YYYYMMDD_paper_slug}.md
     {YYYYMMDD_paper_slug}.json
     {YYYYMMDD_paper_slug}.html
     scoring_weights 스냅샷 -> RunMeta

[11] seen_items에 (paper_key, topic_slug, date) 추가 (dedup=none이면 건너뜀) + issue_map 업데이트 + last_success.json 갱신 (max(기존, window_end))
[12] runs.status = completed

[에러] status = failed, 에러 로그, 다음 토픽으로
```

## 9-4. 토픽 루프 종료 후

1. 토픽 간 중복 감지: 같은 paper_key가 여러 토픽의 최종 출력(임계값 통과)에 포함 -> multi_topic 태그. 태그 값은 해당 토픽 slug 목록 (예: "ai-sports-device, video-analytics").
2. HTML 빌드: 전체 토픽 통합 index + 토픽별 리포트 + latest.html
3. gh-pages 배포 (keep_files: true + 90일 pruning)
4. GitHub Issue upsert (선택)
5. 알림 전송 (토픽별 채널, 파일 첨부 + 폴백)
6. 주간 작업 (일요일): DB purge, 트렌드 요약, 업데이트 스캔, 시각화(optional), Release asset 백업
7. actions/cache save (paper_scout.db, topic_embeddings.npy)
8. git commit (텍스트 메타만: seen_items, issue_map, usage, keyword_cache, model_caps, weekly_done.flag, last_success.json)
9. tmp/ 삭제

## 9-5. 주간 작업 (일요일)

Python이 UTC 기준 일요일 판정. 처리 후 data/weekly_done.flag 생성 (내용: iso_year + iso_week, 예: "2026-W07"). 실행 시작 시 flag의 iso_year+iso_week가 오늘과 같으면 주간 작업을 스킵한다.

| 작업 | 내용 |
|------|------|
| DB purge | 90일 초과 Evaluations + QueryStats + Runs 삭제. remind_tracking 정리. 365일 초과 papers 삭제. VACUUM. Release asset 백업 |
| 주간 트렌드 요약 | 한 주치 데이터에서 키워드 빈도 Top 10 + 점수 추세 |
| 업데이트 논문 스캔 | lastUpdatedDate 기반 주간 스캔. 별도 리포트 (아래 상세) |
| 시각화 (optional) | 임베딩 UMAP 클러스터 맵 + 점수 분포 차트 생성. requirements-viz.txt 설치 시만 작동 |

### 주간 트렌드 요약 상세

순수 통계 집계로 생성한다 (LLM 미사용, 호출 수 추정에 포함되지 않음).

출력:
- 파일명: `YYYYMMDD_weekly_summary.html` / `.md`
- gh-pages: reports/YYYY-MM-DD/ 하위에 저장
- 알림: notifications.weekly_summary 채널로 전송 (파일 첨부 + 한 줄 메시지)

내용:
- 토픽별 키워드 출현 빈도 Top 10
- 토픽별 평균 점수 추이 (일별)
- 이번 주 최고 점수 논문 3편 (전체 토픽 통합)
- 다시 보기 졸업 논문 목록 (이번 주 2회차 도달)

### 업데이트 논문 스캔 상세

papers 테이블에서 "최근 업데이트되었으나 최초 공개는 오래된 논문"을 찾아 별도 리포트로 제공한다.

대상 조건:
- updated_at_utc가 지난 7일 이내
- published_at_utc가 7일 이전 (즉, 이번 주 일일 파이프라인에서 신규로 수집된 것이 아닌 논문)
- published_at_utc가 90일 이내 (Evaluation purge 이전. 이보다 오래된 논문은 평가 데이터가 삭제되어 원래 점수를 표시할 수 없으므로 대상에서 제외)
- 해당 토픽에서 기존 Evaluation이 존재하는 논문 (한 번이라도 평가된 적 있음)

수집 방법: arXiv OAI-PMH의 lastUpdatedDate 파라미터 또는 API 쿼리로 updated 논문 목록을 가져온 뒤, papers 테이블과 조인.

출력:
- 파일명: `YYYYMMDD_weekly_updates.html` / `.md`
- gh-pages: reports/YYYY-MM-DD/ 하위에 저장 (weekly_summary와 동일 디렉토리)
- 알림: weekly_summary와 같은 채널로 전송

리포트 내용:
- 논문 제목 + arXiv ID + 원래 평가 점수(llm_base_score)
- "업데이트됨" 표시 + updated_at_utc 날짜
- 초록/제목의 diff는 수행하지 않음 (비용 대비 효과 낮음). 변경 사항이 궁금하면 arXiv 페이지에서 직접 확인.

## 9-6. 호출 수 추정

| 단계 | LLM 호출 / 토픽 | 근거 |
|------|:---------------:|------|
| Agent 1 | 0~1 | 캐시 히트면 0 |
| Agent 2 | 15~20 | 150~200편 / 10편 배치 |
| Agent 3 Tier 1 | 6 | 30편 / 5편 배치 |
| Agent 3 Tier 2 | 7 | 70편 / 10편 배치 |
| 재시도 | 3~5 | 파싱 실패 |
| 소계 | 31~39 | |

| 토픽 수 | 총 호출 | 예상 시간 |
|:-------:|:------:|:---------:|
| 1 | 31~39 | ~4분 |
| 3 | 93~117 | ~12분 |
| 5 | 155~195 | ~20분 |
| 8 | 248~312 | ~32분 |

---

# 10. 출력물 설계

## 10-1. 파일명 규칙

모든 파일명은 ASCII로 고정. 한국어는 HTML title / 메시지 텍스트에만 사용.

- 일일 리포트: `YYYYMMDD_paper_slug.html` / `.md` / `.json` (예: 20260210_paper_ai-sports-device.html)
- 주간 요약: `YYYYMMDD_weekly_summary.html` / `.md`
- 주간 업데이트: `YYYYMMDD_weekly_updates.html` / `.md` (일요일에만 생성)
- 0건 리포트: 일일 리포트와 동일 파일명. 내용만 "0건" 표시.
- latest: `latest.html` (항상 최신으로 덮어쓰기)

## 10-2. HTML 리포트

### 구조

```
index.html          -- 전체 토픽 목록, 날짜별 리포트 네비게이션
latest.html         -- 항상 최신 리포트로 덮어쓰기 (북마크용)
reports/YYYY-MM-DD/
  YYYYMMDD_paper_slug.html
  YYYYMMDD_weekly_summary.html   -- 일요일에만 생성
  YYYYMMDD_weekly_updates.html   -- 일요일에만 생성
```

### 리포트 페이지 구성

1. 헤더:
   - 제목: "YY년 MM월 DD일 N요일 - {토픽명} arXiv 논문 정리"
   - 통계: 수집 N편 / 필터 N편 / 폐기 N편 / 선정 N편
   - 검색 윈도우 표시

2. 사용된 키워드 (접히는 아코디언):
   - Agent 1이 생성한 concepts 전체 목록 + QueryBuilder가 만든 arXiv 쿼리 목록
   - 평소엔 접혀있고, 펼치면 확인 가능

3. 탭 1 - 오늘의 논문:
   - Tier 1 (상위 30편): 논문 카드
     - 순위 + 점수 바 (base + bonus 분리 표시)
     - 제목 (arXiv 링크)
     - 카테고리 태그
     - 플래그 표시 (엣지, 실시간, 코드)
     - 요약 (아코디언)
     - 선정 근거
     - 활용 인사이트
     - PDF 다운로드 버튼
     - 코드 링크 (있을 때만)
     - 클러스터 앵커 링크
     - [완화] 태그 (임계값 완화로 포함된 경우)
   - 구분선
   - Tier 2 (31~100위): 압축 카드

4. 탭 2 - 다시 보기:
   - 이전 실행에서 80점+ 논문 (최대 2회 노출)
   - 기존 요약 표시
   - "N회째 추천" 표시

5. 하단:
   - arXiv API 이용 고지
   - 실행 메타 정보 (run_id, embedding_mode, 가중치)

## 10-3. GitHub Issue (선택)

- upsert: data/issue_map.json 기반 (window_end + slug -> issue_number)
- 제목: "[YYYY-MM-DD] {토픽명} 논문 리포트 (N건)"
- 본문: 상위 10편 요약 + HTML 리포트 링크. 전체 내용은 HTML로 유도.
- 본문 길이 상한: 60,000자
- 인젝션 방지: @멘션 이스케이프, 특수문자 처리
- 라벨: paper-report, {slug}

## 10-4. JSON

tmp/reports/{YYYYMMDD_paper_slug}.json. HTML 빌드용 데이터 소스.

```json
{
  "meta": {
    "topic_name": "AI 스포츠 디바이스/플랫폼",
    "topic_slug": "ai-sports-device",
    "date": "2026-02-10",
    "display_title": "26년 02월 10일 화요일 - AI 스포츠 디바이스 arXiv 논문 정리",
    "window_start_utc": "2026-02-09T01:30:00Z",
    "window_end_utc": "2026-02-10T02:30:00Z",
    "embedding_mode": "en_synthetic",
    "scoring_weights": {"llm":0.55,"embedding":0.35,"recency":0.10},
    "total_collected": 347,
    "total_filtered": 156,
    "total_discarded": 23,
    "total_scored": 133,
    "total_output": 42,
    "threshold_used": 60,
    "threshold_lowered": false,
    "run_id": 128,
    "keywords_used": ["automatic cinematography", "sports highlight", "..."]
  },
  "clusters": [...],
  "papers": [...],
  "remind_papers": [...]
}
```

## 10-5. Markdown

Tier 1:
```markdown
## 1위: [제목]

- arXiv: https://arxiv.org/abs/...
- PDF: https://arxiv.org/pdf/...
- 코드: https://github.com/...
- 발행일: 2026-02-09
- 카테고리: cs.CV, cs.AI
- 점수: final 87.5 (llm_adjusted:95 = base:82 + bonus:+13)
- 플래그: 엣지, 실시간, 코드 공개

**개요**
[300~500자]

**선정 근거**
[~150자]

**활용 인사이트**
[~150자]

같은 클러스터: #3위, #7위
```

Tier 2:
```markdown
## 31위: [제목]

- arXiv: ... | 2026-02-09 | final 68.2

[~200자]

-> [한 줄 근거]
```

리포트 하단: arXiv API 이용 고지 (문구는 14-5절 참조).

토픽 관리, dry-run, 수동 검색 등은 로컬 관리 도구(18절)에서 수행한다.

---

# 11. 알림(Notifier) 설계

## 11-1. 공통 계약

입력: topic_slug, display_date, keywords_summary, file_paths(md, html), gh_pages_url
출력: 1회 전송. 실패 시 1회 재시도. 최종 실패는 경고 기록 (전체 실패로 전파 금지).

## 11-2. 메시지 형식 (공통)

```
26년 02월 10일 화요일, 오늘의 키워드인 "sports camera", "highlight detection", "pose estimation" 외 12개에 대한 arXiv 논문 정리입니다.
```

키워드: Agent 1이 생성한 concepts에서 name_en 상위 3개 표시 + "외 N개".
메시지 뒤에 HTML/MD 파일 첨부. 별도 요약이나 논문 목록 없음.

## 11-3. 파일 첨부 정책

기본: 파일 첨부 (attachments.enabled: true).

방어 로직:
1. 전송 전 파일 크기 확인 (HTML/MD는 보통 수십 KB)
2. Discord 안전선: 8MB. Telegram 안전선: 50MB (sendDocument 업로드 한도).
3. 크기 초과 시: 자동으로 링크-only 폴백 (gh-pages latest.html URL 전송)
4. 전송 실패 시: 1회 재시도 -> 실패면 링크-only 폴백

## 11-4. 텔레그램

- sendMessage: 메시지 텍스트. parse_mode 미사용 (plain text). 포맷 인젝션 방지.
- sendDocument: HTML 파일, MD 파일 각각 전송. caption에 파일 설명.
- 논문 제목 등 외부 텍스트는 메시지 본문에 포함하지 않으므로 escape 불필요.

## 11-5. 디스코드

- webhook POST (multipart/form-data): 메시지 + 파일 첨부.
- allowed_mentions: { "parse": [] } -> 멘션 차단.
- flags: 4 (SUPPRESS_EMBEDS) -> 링크 프리뷰 억제 (선택).
- wait: true -> 전송 결과 확인.

## 11-6. 토픽별 알림 채널 분리

토픽마다 provider + channel 지정 가능. 회사 토픽은 디스코드, 개인 토픽은 텔레그램 등 혼용 가능.

```yaml
topics:
  - slug: "ai-sports-device"
    notify:
      provider: "discord"
      channel_id: "work-research"
      secret_key: "WORK_RESEARCH"

  - slug: "prompt-engineering"
    notify:
      provider: "telegram"
      channel_id: "personal"
      secret_key: "PERSONAL"
```

Secrets에서 채널별 매핑 (영문 대문자 + 언더스코어만, 하이픈 불가):
- DISCORD_WEBHOOK_WORK_RESEARCH
- TELEGRAM_CHAT_ID_PERSONAL

channel_id는 표시용 슬러그, secret_key는 환경변수 suffix. GitHub Secrets는 영문 대문자/숫자/언더스코어만 허용하므로 반드시 분리한다.

주간 트렌드 요약은 별도 알림 채널 지정 (config의 notifications.weekly_summary). 주간 업데이트 스캔 리포트도 같은 채널로 전송한다.

## 11-7. 0건 처리

논문 0건이어도 최소 리포트("오늘은 0건입니다" 페이지)를 생성한다. 이유: latest.html이 항상 "오늘"을 가리켜야 한다. 0건인데 리포트를 안 만들면 latest.html이 이전 날짜를 가리키게 되어 알림과 리포트의 정합성이 깨진다.

- 0건 리포트: 통계 표시 + "해당 기간에 관련 논문이 없습니다" 메시지
- latest.html: 0건 리포트로 덮어쓰기
- index.html: 0건 날짜도 네비게이션에 포함. "(0건)" 표시로 구분.
- 알림: "26년 02월 15일 일요일, 오늘은 {토픽명} 관련 신규 논문이 없습니다." (주말/공휴일에 arXiv 공지가 없으면 정상)
- 파일 첨부: 0건 리포트 HTML/MD도 전송 (리포트 자체는 존재하므로)

## 11-8. 알림 실패 격리

알림 실패는 run 전체 실패로 전파하지 않는다. HTML 배포 성공이 주요 성공 기준. 알림 실패는 RunMeta에 기록하고 Actions는 warning 처리.

gh-pages 배포도 실패하면: RunMeta에 기록 + 에러 알림 시도 (알림 채널이 살아있다면). 최악의 경우에도 tmp/에 리포트가 남아있고 다음 실행 시 정리됨.

## 11-9. 에러 알림 형식

config의 notifications.on_error가 true일 때, 파이프라인 실패 시 첫 번째 토픽의 알림 채널로 에러 알림 전송:

```
[Paper Scout 오류] 26년 02월 10일 화요일 실행 중 오류가 발생했습니다.
- 실패 토픽: ai-sports-device
- 단계: Agent 2 점수화
- 원인: OpenRouter 429 Too Many Requests
- 완료된 토픽: prompt-engineering (42편)
```

## 11-10. 토픽 실행 순서

config.yaml의 topics 배열 순서대로 실행한다. 일일 한도가 부족하면 뒤쪽 토픽이 스킵되므로, 중요한 토픽을 배열 앞에 배치한다.

# 12. 저장소 및 성장 제어

## 12-1. main 브랜치 (경량 텍스트 메타만 커밋)

| 파일 | 커밋 주기 | 롤링 |
|------|----------|------|
| data/seen_items.jsonl | 매일 | 30일 초과 항목 삭제 (캐시 역할) |
| data/issue_map.json | 매일 | 삭제 안 함 (경량) |
| data/usage/YYYYMMDD.json | 매일 | 30일 초과 파일 삭제 |
| data/keyword_cache.json | 변경 시 | 30일 만료 항목 자동 제거 |
| data/model_caps.json | response_format 확인 시 | 7일 만료 후 재확인 |
| data/weekly_done.flag | 주 1회 (일요일) | iso_year + iso_week 저장. 같은 주면 스킵 |
| data/last_success.json | 매 성공 실행 | 토픽별 last_success_window_end_utc. 증분 윈도우 폴백용 |

리포트 파일(HTML/MD/JSON)은 main 브랜치에 커밋하지 않는다.
바이너리 파일(paper_scout.db, topic_embeddings.npy)은 main 브랜치에 커밋하지 않는다 (12-4절 참조).

## 12-2. gh-pages 브랜치

- 배포 방식: keep_files: true + Python pruning
- 보존 기간: 최근 90일 (config로 변경 가능)
- 배포 시: 90일 초과 reports/YYYY-MM-DD/ 디렉토리 삭제 후 배포
- latest.html: 항상 최신으로 덮어쓰기 (고정 URL)
- index.html: 전체 날짜/토픽 네비게이션 갱신

## 12-3. tmp/ (임시, gitignore)

매 실행 시작 시 이전 잔여물 삭제 후 생성. 실행 종료 시 삭제.

```
tmp/
  reports/           -- MD, JSON, HTML
  html/              -- gh-pages 배포용 빌드
  debug/             -- 파싱 실패 원문
  dry-run/           -- 수동 dry-run 결과
```

## 12-4. DB 영속화 (actions/cache + Release asset)

paper_scout.db와 topic_embeddings.npy는 바이너리 파일이므로 git에 커밋하지 않는다. git은 바이너리 diff를 지원하지 않아 매 커밋마다 전체 복사본이 히스토리에 쌓이고, 저장소 크기가 급격히 증가한다.

### 저장 전략

| 파일 | 영속 방식 | 키 | 비고 |
|------|----------|-----|------|
| paper_scout.db | actions/cache | `db-{branch}-{hash of schema version}` | 매 실행 종료 시 save. 7일 미사용 시 자동 삭제 |
| topic_embeddings.npy | actions/cache | `embed-{branch}-{hash of agent1 cache key}` | Agent 1 캐시 해시 변경 시 갱신 |
| paper_scout.db 백업 | GitHub Release asset | `paper-scout-db-{YYYYMMDD}.sqlite` | 주 1회 (일요일). 최근 4주분만 유지 |

### Cache miss 복구

1. actions/cache miss → Release asset에서 최신 DB 다운로드
2. Release asset도 없음 → 빈 DB 생성. seen_items.jsonl로 단기(30일) dedup 유지. 31일 이전 논문은 재수집될 수 있으나, 리포트 품질에는 영향 없음 (같은 논문이 다시 수집되면 Agent 2가 다시 평가)
3. topic_embeddings.npy cache miss → Agent 1 출력으로 재계산 (1회 연산, 수 초)

### Release asset 관리

주간 작업(일요일)에서 DB purge + VACUUM 후 Release asset 업로드. 4주(28일) 초과 asset은 삭제.

## 12-5. DB 유지보수 (주간)

일요일 실행 시:
1. paper_evaluations에서 90일 초과 레코드 삭제
2. query_stats에서 90일 초과 레코드 삭제
3. runs에서 90일 초과 레코드 삭제
4. remind_tracking에서 대응하는 paper_key가 paper_evaluations에 없는 레코드 삭제 (졸업 완료 + 90일 경과)
5. papers에서 365일 초과 레코드 삭제 (TTL 경과 논문. 이후 같은 논문이 수집되면 신규로 처리)
6. VACUUM
7. Release asset 업로드 (purge 완료 후 깨끗한 DB)
8. data/weekly_done.flag 생성 (내용: iso_year + iso_week, 예: "2026-W07")

---

# 13. GitHub Actions 워크플로

## 13-1. 자동 실행 (cron)

```yaml
name: Paper Scout Daily

on:
  schedule:
    - cron: '0 2 * * *'      # UTC 02:00 = KST 11:00
  workflow_dispatch:
    inputs:
      date_from:
        description: 'YYYY-MM-DD (KST)'
        required: false
        type: string
      date_to:
        description: 'YYYY-MM-DD (KST)'
        required: false
        type: string
      mode:
        description: 'full / dry-run'
        required: false
        default: 'full'
        type: string
      dedup:
        description: 'skip_recent / none'
        required: false
        default: 'skip_recent'
        type: string

concurrency:
  group: paper-scout-${{ github.ref }}
  cancel-in-progress: false
```

> 참고: cancel-in-progress: false이므로 cron과 수동 실행이 겹치면 후행 실행이 대기한다. GitHub Actions 대기 제한(기본 360분)을 초과하면 후행 실행이 취소되므로, 토픽 수가 많아 실행 시간이 길어질 경우 수동 실행 시점에 유의한다.

```yaml
permissions:
  contents: write
  issues: write
  pages: write
```

## 13-2. 단계

| 단계 | 내용 | 실패 시 |
|------|------|---------|
| Checkout | actions/checkout@v4 | 중단 |
| Restore DB cache | actions/cache → paper_scout.db, topic_embeddings.npy | miss면 Release asset 폴백 → 빈 DB 생성 |
| Setup Python | 3.11, pip 캐시 | 중단 |
| Install core | requirements-core.txt | 중단 |
| Install embed | requirements-embed.txt, continue-on-error | 경고, 임베딩 비활성 |
| Install viz | requirements-viz.txt, continue-on-error | 경고, 시각화 비활성 |
| Run main.py | Preflight + 파이프라인 | 토픽별 격리 |
| Save DB cache | actions/cache save (항상) | 경고 |
| Commit metadata | seen_items, issue_map, usage, keyword_cache, model_caps, weekly_done.flag, last_success.json | 커밋 없으면 pass |
| Upload DB backup (weekly) | weekly_done.flag 갱신 시 → Release asset 업로드 | 경고 |
| Deploy gh-pages | keep_files: true | 경고, 알림 시도 |
| Upload debug (실패 시만) | tmp/debug/ → Actions artifact | -- |

## 13-3. 수동 실행

두 가지 방식:

**방식 A: CLI (기본, 권장)**
```bash
gh workflow run paper-scout.yml \
  -f date_from=2026-02-01 \
  -f date_to=2026-02-07 \
  -f mode=full
```

dry-run (키워드 확인만):
```bash
gh workflow run paper-scout.yml -f mode=dry-run
```
dry-run 결과는 GitHub Issue 또는 알림 채널로 키워드 + QueryBuilder 쿼리 목록 전송. 실제 수집/점수/요약은 하지 않음.
dry-run에서 생성된 Agent 1 캐시는 정상 캐시에 저장된다 (의도적). dry-run으로 키워드를 확인하고 description을 수정하면 해시가 변경되어 캐시가 갱신되므로 오염 문제는 발생하지 않는다.

**방식 B: 로컬 관리 도구 (18절)**

`paper-scout ui`로 로컬 웹 UI를 열고, 수동 검색 탭에서 날짜 범위 선택 + 실행. 토픽 관리, dry-run 키워드 확인도 같은 UI에서 수행.

## 13-4. 의존성 분리

requirements-core.txt (필수):
- arxiv, openai, json-repair, jinja2, requests, pyyaml, numpy

requirements-embed.txt (선택):
- sentence-transformers, torch

requirements-viz.txt (선택, 주간):
- umap-learn, matplotlib

requirements-ui.txt (선택, 로컬 관리):
- flask (또는 fastapi + uvicorn)

Actions에서는 requirements-ui.txt를 설치하지 않는다. 로컬 관리 도구는 로컬 환경에서만 사용.
모든 optional 의존성은 continue-on-error로 설치. Python에서 import 성공 여부로 기능 on/off 판정.

---

# 14. 보안 및 컴플라이언스

## 14-1. Secrets 관리

| Secret | 용도 | 비고 |
|--------|------|------|
| OPENROUTER_API_KEY | LLM 호출 | |
| DISCORD_WEBHOOK_{SECRET_KEY} | Discord 알림 | 토픽별. 대문자+언더스코어만 |
| TELEGRAM_BOT_TOKEN | Telegram 봇 | |
| TELEGRAM_CHAT_ID_{SECRET_KEY} | Telegram 채팅 | 토픽별. 대문자+언더스코어만 |
| GITHUB_TOKEN | 자동 제공 | Issue, Pages |

## 14-2. 로그 마스킹

- API 키, 봇 토큰, webhook URL은 로그에 절대 출력 금지
- URL 출력 시 마스킹: https://discord.com/api/webhooks/***

## 14-3. Discord webhook 보안

- 분기별 로테이션 권장
- 최소 권한 채널에서 운영

## 14-4. 로컬 관리 도구 보안

- 로컬 서버는 127.0.0.1만 바인딩 (외부 접속 차단)
- API 키는 .env 또는 환경변수에서 로드. config.yaml에 평문 저장 금지
- 로컬에서만 실행되므로 PAT 불필요

## 14-5. arXiv 이용 고지

- 리포트 하단: "This report uses the arXiv API. See https://info.arxiv.org/help/api/"
- README에도 동일 고지

## 14-6. HTML 보안

- Jinja2 autoescape=True
- |safe 필터 사용 금지
- arXiv 제목/초록의 LaTeX, 특수문자에 의한 XSS 방지

---

# 15. 관측성 및 재현성

## 15-1. RunMeta

매 실행마다 DB runs 테이블 + tmp/run_meta.json에 기록.

필수 기록 항목:
- run_id, topic_slug
- window_start/end (UTC), display_date (KST)
- embedding_mode, scoring_weights (가중치 + 보너스 설정 스냅샷)
- detected_rpm, detected_daily_limit
- response_format_supported (boolean)
- prompt_versions (agent1, agent2, agent3)
- topic_override_fields (사용된 선택 필드 목록. 예: ["must_concepts_en", "must_not_en"]. 없으면 빈 배열)
- 토픽별 통계: collected, filtered, discarded, scored, output, threshold_used, threshold_lowered
- 에러 요약

## 15-2. QueryStats

쿼리별 상세:
- query_text, collected, total_available, truncated
- retries, duration_ms, exception

DB 테이블에 기록. 90일 purge 대상.

## 15-3. 디버그 샘플

파싱 실패 원문을 tmp/debug/에 저장:
- {agent_name}_{batch_index}.txt
- 실행 성공 시: tmp/ 전체 삭제
- 실패 시: tmp/debug/를 GitHub Actions artifact로 업로드 (기본 90일 보존). 장애 원인 추적에 필수. 이후 tmp/ 삭제.

---

# 16. 에러 대응 정책

## 16-1. 대응표

| 실패 유형 | 감지 | 복구 | 최종 폴백 |
|----------|------|------|----------|
| config 검증 실패 | Preflight | -- | 즉시 중단 |
| API 키 무효 | Preflight | -- | 즉시 중단 |
| RPM/한도 감지 실패 | Preflight | 보수적 기본값 (RPM=10, daily=200) | 경고 후 계속 |
| arXiv EmptyPage | 예외 | 2회 재시도 (3s, 9s) | 쿼리 스킵, 로그 |
| arXiv 타임아웃/5xx | HTTP 에러 | Client(num_retries=3) | 쿼리 스킵, 로그 |
| arXiv truncation | totalResults | 로그 + 경고 | 진행 (정보 로깅만) |
| Agent 1 파싱 | ParseError | 재시도 2회 | 카테고리 기본 키워드 |
| Agent 2 파싱 | ParseError | 재시도 2회 + 디버그 저장 | 배치 스킵 |
| Agent 2 출력 누락 | len 검증 | 누락분 별도 배치 재호출 (1회) | 임베딩 on: embed_score만으로 정렬. 임베딩 off: recency만으로 정렬 |
| Agent 3 파싱 | ParseError | 재시도 2회 + 디버그 저장 | 요약 없이 출력 |
| OpenRouter 429 | HTTP 429 | 지수 백오프 (2, 6, 18초 + jitter) | 부분 리포트 |
| OpenRouter 5xx | HTTP 5xx | 3회 재시도 | 배치 스킵 |
| 일일 한도 초과 | usage 영속 카운터 | -- | 남은 토픽 스킵, 완료분만 리포트 |
| 임베딩 import 실패 | ImportError | -- | 규칙 기반만 + 클러스터링 비활성 |
| GitHub Issue API | API 에러 | 1회 재시도 | 스킵 (HTML/알림은 전송됨) |
| 알림 전송 실패 | API 에러 | 1회 재시도 -> 링크-only 폴백 | 경고 기록 (전체 실패 전파 금지) |
| gh-pages 배포 실패 | Actions 에러 | -- | 경고 + 에러 알림 시도 |

## 16-2. RateLimiter 일일 사용량 영속화

data/usage/YYYYMMDD.json:
```json
{
  "date": "2026-02-10",
  "api_calls": 234,
  "topics_completed": ["ai-sports-device", "prompt-engineering"],
  "topics_skipped": []
}
```

프로세스 시작 시 당일 파일 로드. 없으면 0. RateLimiter 초기화에 사용.
같은 날 수동 실행 여러 번 해도 누적치 유지.

## 16-3. RateLimiter 동적 딜레이

Preflight에서 감지한 RPM 기반:
```
delay = 60 / detected_rpm + 0.5
```

감지 실패 시 보수적 기본값: delay = 60 / 10 + 0.5 = 6.5초.

---

# 17. 테스트 계획

## 17-1. 단위 테스트

| 모듈 | 테스트 항목 | 우선순위 |
|------|-----------|:--------:|
| json_parser | 브라켓 밸런싱, 중첩 괄호, 문자열 내 괄호, 미종결 think, response_format 모드 전환 | P0 |
| rule_filter | 키워드 매칭, 부정 키워드, 빈 초록 | P0 |
| rate_limiter | RPM 슬라이딩 윈도우, 일일 한도 영속화, 재시작 복원, 동적 딜레이 | P0 |
| code_detector | github URL, "code available" 변형, 오탐 방지, regex+llm 병합, provenance | P0 |
| ranker | 가산 (flags), final_score (임베딩 on/off), 정렬, 임계값 완화 순서 (60->50->40), Tier 배정 | P0 |
| query_builder | concepts→쿼리 변환, 카테고리 조합, 키워드 교차, 15~25개 생성, 빈 키워드 처리 | P0 |
| notifier | 파일 첨부, 크기 초과 폴백, 링크-only, 멘션 차단, 0건 메시지 | P0 |
| dedup | 2계층: in_run(항상 ON, run 내 중복 제거) + cross_run(seen_items 기준, 모드 선택). papers 메타데이터 재활용, 30일 롤링, multi_topic 허용 | P1 |
| embedding_ranker | cosine, import 실패, 영문 토픽 텍스트, 모드 전환 | P1 |
| clusterer | 그룹핑, 대표 선정, 임베딩 없을 때 비활성 | P1 |
| remind | 80점+ 필터, recommend_count 증가, 2회 졸업 | P1 |
| issue_publisher | upsert (issue_map), 이스케이프 (@방지), 길이 상한 | P1 |
| db_manager | CRUD, purge (90일 eval/query/runs + 365일 papers), VACUUM, seen_items 롤링, usage 롤링, remind_tracking CRUD, remind_tracking purge, cache miss 복구 | P1 |
| topic_spec | 필수 필드 검증, 선택 필드 타입 검증, 선택 필드 미지정 시 정상 동작, 캐시 해시 변경 감지 | P2 |
| html_generator | 유효성, XSS 이스케이프, 아코디언, 탭, latest 덮어쓰기 | P2 |

## 17-2. 통합 테스트

| 시나리오 | 검증 |
|---------|------|
| E2E 토픽 1개 | 파이프라인 완료, HTML 생성, 알림 전송 |
| 토픽 3개 (디스코드+텔레그램 혼용) | RateLimiter 공유, 독립 리포트, 채널별 전송 |
| EmptyPage | 재시도 -> 스킵 -> 나머지 정상 |
| truncation | totalResults 기반 로깅 |
| 임베딩 미설치 | 규칙만, 가중치 자동 전환, 클러스터 비활성 |
| 중복 실행 | concurrency 대기, Issue upsert |
| 미종결 think | 파싱 성공 |
| 중첩 괄호 JSON | 밸런싱 파싱 성공 |
| flags 가산 | Ranker 결정론 검증 |
| 임계값 완화 | 60 -> 50 -> 40 순서 |
| has_code 병합 | regex OR llm, provenance 기록 |
| 일일 한도 영속 | 재시작 후 누적 유지 |
| 0건 | 최소 리포트 생성, latest.html 덮어쓰기, 알림 "없습니다" + 파일 첨부 |
| 파일 크기 초과 | 링크-only 폴백 |
| dry-run | Agent 1 + QueryBuilder만 실행, 키워드 + 쿼리 목록 전송 |
| 다시 보기 | 80점+, 2회 졸업, 별도 탭 |
| 주간 트렌드 | 키워드 빈도, 점수 추세, 파일 생성, weekly_summary 채널 전송 |
| 수동 실행 (과거 날짜) | cross_run(seen_items) 30일 내면 스킵(skip_recent), 초과면 재평가 가능, 정상 리포트 |
| 수동 실행 (--dedup=none) | cross_run 건너뜀 + seen_items 쓰기 안 함. in_run은 작동. 이후 일일 실행에 영향 없음 검증 |
| in_run dedup | 같은 run 내 다중 쿼리 중복 → in_run set이 제거. LLM 중복 호출 없음 |
| DB cache miss | Release asset 폴백 → 빈 DB → last_success.json 폴백 → 72h 최종 폴백 → 정상 실행 |
| 증분 윈도우 | 직전 성공 run의 window_end → 당일 KST 11:00. DB miss → last_success.json → 72h 폴백 |
| last_success.json | 성공 실행 후 토픽별 window_end_utc 갱신 + git 커밋 검증 |
| discard (base_score < 20) | is_metaphorical=true 또는 base_score < 20만 discard. 20~59점은 Ranker 전달 |
| QueryBuilder | Agent 1 concepts → arXiv 쿼리 15~25개 생성 → 수집 정상 |
| pre_embed_cap | 규칙 통과 2,000편 초과 시 recency 순 상위 2,000편만 임베딩 |
| 주간 업데이트 스캔 | updated_at 7일 내 + published_at 7일~90일 이전 + Evaluation 존재 → weekly_updates 리포트 생성 |

## 17-3. 회귀 테스트 (Golden)

- 임베딩 on/off 시 결과 순위 상관도 최소 기준
- 임계값 완화가 "5편 미만일 때만" 작동하는지
- truncation 판정 오탐 방지

---

# 18. 로컬 관리 도구

로컬에서 실행되는 Python 웹 서버. 토픽 관리, dry-run, 수동 검색을 브라우저 UI로 수행한다. gh-pages와 무관하며, GitHub Actions에도 배포되지 않는다.

## 18-1. 실행

```bash
paper-scout ui                  # 기본: 127.0.0.1:8585
paper-scout ui --port 9090      # 포트 변경
```

브라우저가 자동으로 열린다 (config: local_ui.open_browser).

## 18-2. 기술 스택

- Python (Flask 또는 FastAPI). 별도 Node.js/npm 불필요
- 프론트엔드: 단일 HTML + vanilla JS (또는 경량 번들). 빌드 도구 없음
- 127.0.0.1만 바인딩 (외부 접속 차단, 14-4절)

## 18-3. 페이지 구성

### 페이지 1 - 토픽 관리

config.yaml을 직접 읽고 쓴다.

**토픽 목록**
- 등록된 토픽 카드: slug, name, description 요약, arxiv_categories
- 선택 필드 표시: must_concepts_en, should_concepts_en, must_not_en (설정된 것만)
- 캐시 상태: Agent 1 키워드 캐시 유무, 만료까지 남은 일수
- 최근 실행 통계: 마지막 실행일, 수집/선정 수 (DB에서 조회)

**토픽 추가**
- 폼: slug, name, description(한국어, 자유 서술), arxiv_categories
- 선택 필드: must_concepts_en, should_concepts_en, must_not_en (접히는 고급 설정)
- 알림 채널: provider(discord/telegram) + channel_id + secret_key
- 저장 → config.yaml에 직접 추가

**토픽 수정/삭제**
- 기존 토픽의 모든 필드 수정 가능
- description 변경 시 캐시 해시가 달라지므로, 다음 실행에서 Agent 1이 키워드를 재생성한다는 안내 표시
- 삭제 시 확인 다이얼로그

### 페이지 2 - Dry-run / 수동 검색

**dry-run**
- 토픽 선택 (또는 전체)
- 실행 버튼 → Python 파이프라인을 로컬에서 직접 호출 (Agent 1 + QueryBuilder까지 실행)
- 결과 인라인 표시: concepts, cross_domain_keywords, exclude_keywords, QueryBuilder가 만든 arXiv 쿼리 목록
- "이 키워드로 실제 검색" 버튼 → 수동 검색으로 이동

**수동 검색**
- 토픽 선택
- 날짜 범위 (date_from / date_to)
- Dedup 모드 선택: skip_recent(기본) / none
- 실행 → 로컬에서 파이프라인 full 실행
- 진행 상황 실시간 표시 (수집 중 → 필터 중 → 점수화 중 → 요약 중)
- 완료 후 결과 리포트를 브라우저에서 바로 확인

### 페이지 3 - 설정

- config.yaml의 비-토픽 설정 편집: LLM 모델, 가중치, 임계값, 알림 채널 등
- .env 파일의 API 키 상태 확인 (값은 마스킹, 유효성만 표시)
- DB 상태: 레코드 수, 용량, 마지막 purge 날짜

## 18-4. 로컬 vs Actions 실행 차이

| 항목 | 로컬 (paper-scout ui / CLI) | GitHub Actions (cron / dispatch) |
|------|---------------------------|--------------------------------|
| 트리거 | 사용자 직접 | cron 자동 / dispatch |
| config 접근 | 파일시스템 직접 읽기/쓰기 | 저장소 체크아웃 후 읽기만 |
| DB 영속화 | 로컬 파일 (항상 존재) | actions/cache + Release asset |
| 결과 배포 | 로컬에서 바로 확인 | gh-pages + 알림 |
| 리포트 커밋 | 하지 않음 (로컬 tmp/) | gh-pages에 배포 |

로컬 실행 결과는 gh-pages에 배포되지 않는다. 로컬은 테스트/관리용이고, 일일 자동 리포트 배포는 Actions가 담당한다.

## 18-5. CLI 명령어

UI 없이도 CLI로 동일 기능 수행 가능:

```bash
paper-scout topic list                          # 등록 토픽 목록
paper-scout topic add                           # 대화형 토픽 추가
paper-scout topic edit ai-sports-device          # 토픽 수정 ($EDITOR)
paper-scout topic remove ai-sports-device        # 토픽 삭제
paper-scout dry-run                             # 전체 토픽 dry-run
paper-scout dry-run --topic ai-sports-device     # 특정 토픽만
paper-scout run --date-from 2026-02-01 --date-to 2026-02-07  # 수동 검색
paper-scout run --dedup=none --date-from 2026-02-01 --date-to 2026-02-07  # 재검증 (dedup 건너뜀)
paper-scout run                                 # 당일 증분 윈도우 실행
paper-scout ui                                  # 웹 UI 실행
```

---

# 19. 확장 가이드

## 19-1. 새 소스 추가

1. core/sources/{source}.py에 SourceAdapter 구현
2. 내부에 QueryBuilder 구현: Agent 1의 concepts/cross_domain_keywords/exclude_keywords를 해당 소스의 쿼리 DSL로 변환
3. 입력: Agent 1 출력(concepts, cross_domain_keywords, exclude_keywords) + 시간 윈도우(UTC) + 설정
4. 출력: Paper[] (정규화, UTC 시각 필수)
5. paper_key = "{source}:{native_id}"
6. registry에 등록
7. TopicSpec에 해당 소스용 카테고리/설정 필드 추가 (5절 참조)

교차 소스 Dedup 우선순위: 4-5절 참조.

## 19-2. 새 알림 채널 추가

1. output/notifiers/{provider}.py 구현
2. 공통 계약 준수: 메시지 + 파일 첨부 + 폴백
3. 실패는 경고 (전체 실패 전파 금지)
4. config의 notify.provider에 추가
5. Secrets 매핑

## 19-3. TopicSpec 선택 필드 추가

1. config.yaml topics 항목에 새 선택 필드 정의
2. Preflight validator에 타입 검증 추가
3. Agent 1 프롬프트에 해당 필드 전달 로직 추가
4. 캐시 해시 계산에 새 필드 포함

---

# 20. 부록

## A. config.yaml 전체 스키마

```yaml
app:
  display_timezone: "Asia/Seoul"
  report_retention_days: 90

llm:
  provider: "openrouter"
  base_url: "https://openrouter.ai/api/v1"
  model: "tngtech/deepseek-r1t2-chimera:free"
  app_url: "https://github.com/{username}/paper-scout"
  app_title: "Paper Scout"
  retry:
    max_attempts: 3
    backoff_base_seconds: 2
    jitter: true
  preflight:
    fallback_rpm: 10
    fallback_daily: 200
    model_caps_path: "data/model_caps.json"
    model_caps_ttl_days: 7

agents:
  common:
    always_strip_think: true
    response_field: "content"
  keyword_expander:
    effort: "high"
    max_tokens: 2048
    temperature: 0.3
    cache_ttl_days: 30
    prompt_version: "agent1-v3"
  scorer:
    effort: "low"
    max_tokens: 2048
    temperature: 0.2
    batch_size: 10
    prompt_version: "agent2-v2"
  summarizer:
    effort: "low"
    max_tokens: 4096
    temperature: 0.4
    batch_size_tier1: 5
    batch_size_tier2: 10
    prompt_version_tier1: "agent3-tier1-v1"
    prompt_version_tier2: "agent3-tier2-v1"

sources:
  - type: "arxiv"
    enabled: true
    max_results_per_query: 200
    client:
      num_retries: 3
      delay_seconds: 3
    empty_page_retries: 2
    empty_page_backoff: [3, 9]

filter:
  pre_embed_cap: 2000           # 규칙 통과 후보 상한. 초과 시 recency 순 상위만 임베딩

embedding:
  model: "sentence-transformers/all-MiniLM-L6-v2"
  device: "cpu"
  optional: true
  mode: "en_synthetic"

scoring:
  weights:
    llm: 0.55
    embedding: 0.35
    recency: 0.10
  weights_no_embedding:
    llm: 0.80
    recency: 0.20
  bonus:
    is_edge: 5
    is_realtime: 5
    has_code: 3
  thresholds:
    default: 60
    relax_steps: [50, 40]
    min_papers_before_relax: 5
  discard_cutoff: 20              # base_score가 이 값 미만이면 discard
  max_output: 100
  tier1_count: 30

remind:
  enabled: true
  min_score: 80
  max_expose_count: 2

clustering:
  enabled: true
  similarity_threshold: 0.85

topics:
  - slug: "ai-sports-device"
    name: "AI 스포츠 디바이스/플랫폼"
    description: |
      자동으로 경기 촬영하고 주요 장면이나 개인별 하이라이트 편집하고
      영상 보정하고 자세/전략 분석하는 AI 스포츠 디바이스 만들건데
      실시간으로 돌아가야 하고 모바일에서도 써야 함
    arxiv_categories: [cs.CV, cs.AI, cs.MM, cs.LG, eess.IV]
    must_concepts_en:                          # 선택
      - "sports analytics"
      - "camera automation"
    should_concepts_en:                        # 선택
      - "tracking"
      - "video understanding"
    must_not_en:                               # 선택
      - "medical imaging"
      - "satellite"
    notify:
      provider: "discord"
      channel_id: "work-research"
      secret_key: "WORK_RESEARCH"

  - slug: "prompt-engineering"
    name: "프롬프트 엔지니어링 / LLM"
    description: |
      프롬프트 엔지니어링 최신 기법, LLM 추론 최적화,
      경량화, 파인튜닝, RAG, 에이전트 아키텍처 관련 논문
    arxiv_categories: [cs.CL, cs.AI, cs.LG]
    notify:
      provider: "telegram"
      channel_id: "personal"
      secret_key: "PERSONAL"

output:
  gh_pages:
    enabled: true
    keep_files: true
    retention_days: 90
    latest_name: "latest.html"
  github_issue:
    enabled: true
    labels: ["paper-report"]
    body_char_limit: 60000
    upsert: true
    escape_mentions: true
  attachments:
    enabled: true
    discord_max_mb: 8
    telegram_max_mb: 50

notifications:
  weekly_summary:
    provider: "discord"
    channel_id: "work-research"
    secret_key: "WORK_RESEARCH"
  on_zero_results: true
  on_error: true

database:
  path: "data/paper_scout.db"
  retention_days: 90
  papers_ttl_days: 365
  vacuum_on_purge: true
  seen_items_path: "data/seen_items.jsonl"
  seen_items_rolling_days: 30
  issue_map_path: "data/issue_map.json"
  usage_dir: "data/usage"
  persistence: "actions_cache"
  release_backup:
    enabled: true
    keep_weeks: 4

weekly:
  trend_summary: true
  update_scan: true
  visualization: true

local_ui:
  host: "127.0.0.1"
  port: 8585
  open_browser: true
```

## B. 디렉토리 구조

```
paper-scout/
  .github/workflows/paper-scout.yml
  .gitignore                     # tmp/ 포함

  config.yaml
  main.py
  requirements-core.txt
  requirements-embed.txt
  requirements-viz.txt
  requirements-ui.txt
  README.md

  agents/
    base_agent.py                # think 제거, 헤더, 파싱 방어, 디버그 저장
    keyword_expander.py          # Agent 1: topic_embedding_text 생성
    scorer.py                    # Agent 2: base_score + flags
    summarizer.py                # Agent 3: 한국어 요약

  core/
    pipeline/
      preflight.py               # 설정 검증, API 상태, RPM 감지
      orchestrator.py             # 토픽 루프, 주간 작업, tmp 관리
    sources/
      base.py                    # SourceAdapter 인터페이스
      arxiv.py                   # arXiv 어댑터
      arxiv_query_builder.py     # concepts/cross_domain_keywords -> arXiv 쿼리 DSL
      registry.py
    llm/
      openrouter_client.py       # 앱 식별 헤더, extra_body
      rate_limiter.py            # 동적 딜레이, 일일 영속화
    embeddings/
      embedding_ranker.py          # optional import, 영문 토픽 텍스트
    scoring/
      ranker.py                  # 결정론적 가산, 임계값 완화
      code_detector.py           # 정규식 정본 + LLM 보조 병합
    parsing/
      json_parser.py             # 브라켓 밸런싱, response_format 분기
    storage/
      db_manager.py              # purge, VACUUM
      dedup.py                   # 2계층: in_run(run 내 중복) + cross_run(seen_items topic별 스킵) + papers(메타데이터 재활용)
      usage_tracker.py           # 일일 사용량 영속화
    clustering/
      clusterer.py               # cosine 0.85, 임베딩 없으면 비활성

  output/
    render/
      html_generator.py          # autoescape, 탭, 아코디언, 키워드 패널
      md_generator.py            # Tier 1/2 포맷
      json_exporter.py
    notifiers/
      base.py                    # 공통 계약
      discord.py                 # webhook, 멘션 차단, 파일 첨부
      telegram.py                # sendDocument, plain text
      registry.py
    github_issue.py              # upsert, 이스케이프

  prompts/
    agent1_keyword_expansion.txt
    agent2_scoring.txt
    agent3_summary_tier1.txt
    agent3_summary_tier2.txt

  templates/
    report.html.j2
    index.html.j2
    latest.html.j2
    weekly_summary.html.j2
    weekly_updates.html.j2
    base.html.j2

  scripts/
    rebuild_db.py                # JSON 아카이브로 DB 재구축
    prune_gh_pages.py            # 90일 초과 리포트 삭제

  local_ui/                        # 로컬 관리 도구 (18절)
    server.py                    # Flask/FastAPI 로컬 서버
    static/                      # HTML/CSS/JS
    templates/                   # 관리 UI 템플릿

  data/
    keyword_cache.json           # git 추적
    model_caps.json              # git 추적
    seen_items.jsonl             # git 추적
    issue_map.json               # git 추적
    weekly_done.flag             # git 추적
    last_success.json            # git 추적 (토픽별 last_success_window_end_utc)
    usage/                       # git 추적
      YYYYMMDD.json
    paper_scout.db               # .gitignore (actions/cache + Release asset 백업)
    topic_embeddings.npy          # .gitignore (actions/cache. miss 시 재계산)

  tmp/                           # gitignore
    reports/
    html/
    debug/
    dry-run/

  tests/
    test_json_parser.py
    test_rule_filter.py
    test_embedding_ranker.py
    test_rate_limiter.py
    test_code_detector.py
    test_ranker.py
    test_query_builder.py
    test_notifier.py
    test_dedup.py
    test_remind.py
    test_issue_upsert.py
    test_clusterer.py
    test_topic_spec.py
```

## C. 점수 산출 공식

```
[discard]
discard = (is_metaphorical=true) OR (base_score < 20)
discarded 논문은 Ranker에 전달되지 않음. 20~59점은 discard하지 않음 (임계값 완화의 후보).

[embed_score 정규화]
embed_score = clamp(cosine_similarity, 0, 1)

[가산]
bonus = (5 if is_edge) + (5 if is_realtime) + (3 if has_code)
llm_adjusted = min(base_score + bonus, 100)

[최종 점수 - 임베딩 사용]
final_score = 0.55 * llm_adjusted + 0.35 * (embed_score * 100) + 0.10 * recency_score

[최종 점수 - 임베딩 미사용]
final_score = 0.80 * llm_adjusted + 0.20 * recency_score

[recency_score] (기준일: window_end, 경과일 = floor((window_end - published_at_utc) / 24h))
0일=100, 1일=90, 2일=80, 3일=70, 4일=60, 5일=50, 6일=40, 7일+=30

[임계값 완화]
기본: final_score 60점 -> 5편 미만이면 50점 -> 그래도 5편 미만이면 40점
```

## D. arXiv API 참고

- 매뉴얼: https://info.arxiv.org/help/api/user-manual.html
- submittedDate 형식: [YYYYMMDDHHMM TO YYYYMMDDHHMM] (분 단위, GMT). 예: [202602090130 TO 202602100230]
- URL 직접 구성 시 urllib.parse.quote_plus 필수. arxiv.py 라이브러리 사용 시 자동.
- 3초 요청 간 딜레이 권고 (Client delay_seconds=3).
- 이용 고지: https://info.arxiv.org/help/api/

## E. OpenRouter API 참고

- 한도: https://openrouter.ai/docs/api/reference/limits
- :free 모델은 크레딧 잔량에 따라 일일 한도 변동
- /api/v1/key: 키 상태/한도 조회
- 전용 파라미터(reasoning 등)는 OpenAI SDK extra_body로 전달

---

문서 끝 -- v1.5
