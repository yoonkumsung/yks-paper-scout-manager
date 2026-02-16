# DB Cache Manager

GitHub Actions용 DB 캐시 관리 스크립트입니다. `actions/cache`와 GitHub Release asset을 활용하여 DB를 저장하고 복원합니다.

## 기능

- **Cache Key 생성**: 브랜치와 스키마 버전 기반 캐시 키 생성
- **Release 복원**: GitHub Release asset에서 DB 다운로드
- **Release 업로드**: 주간 백업을 Release asset으로 업로드
- **자동 정리**: 4주 이상 된 Release asset 삭제
- **DB 보장**: DB가 없으면 복원 또는 생성

## 캐시 전략 (devspec 12-4)

| 파일 | 저장 방법 | 캐시 키 | 비고 |
|------|----------|---------|------|
| paper_scout.db | actions/cache | `db-{branch}-{schema_hash}` | 매 실행마다 저장, 7일 미사용 시 자동 삭제 |
| topic_embeddings.npy | actions/cache | `embed-{branch}-{agent1_cache_hash}` | Agent 1 캐시 해시 변경 시 갱신 |
| paper_scout.db 백업 | GitHub Release asset | `paper-scout-db-{YYYYMMDD}.sqlite` | 주간 (일요일), 최근 4주만 보관 |

## 캐시 미스 복구 체인

1. `actions/cache` 미스 → 최신 Release asset에서 DB 다운로드
2. Release asset 없음 → 빈 DB 생성. 30일 중복 제거는 `seen_items.jsonl`로 계속 작동
3. `topic_embeddings.npy` 미스 → Agent 1 출력에서 재계산 (수 초)

## 사용법

### 1. Cache Key 생성

```bash
python3 scripts/cache_manager.py cache-key --branch main
# 출력: db-main-abc12345
```

GitHub Actions에서 사용:
```yaml
- name: Generate cache key
  id: cache-key
  run: |
    CACHE_KEY=$(python3 scripts/cache_manager.py cache-key --branch ${{ github.ref_name }})
    echo "cache-key=$CACHE_KEY" >> $GITHUB_OUTPUT
```

### 2. DB 복원

```bash
python3 scripts/cache_manager.py restore \
  --repo owner/repo \
  --db-path data/paper_scout.db
```

GitHub Actions에서 사용:
```yaml
- name: Restore DB from cache
  uses: actions/cache/restore@v4
  with:
    path: data/paper_scout.db
    key: ${{ steps.cache-key.outputs.cache-key }}

- name: Fallback to Release asset
  if: steps.cache-db.outputs.cache-hit != 'true'
  run: |
    python3 scripts/cache_manager.py restore \
      --repo ${{ github.repository }} \
      --db-path data/paper_scout.db
```

### 3. DB 저장

GitHub Actions에서 사용:
```yaml
- name: Save DB to cache
  uses: actions/cache/save@v4
  if: always()
  with:
    path: data/paper_scout.db
    key: ${{ steps.cache-key.outputs.cache-key }}
```

### 4. Release 업로드 (주간 백업)

```bash
python3 scripts/cache_manager.py upload \
  --repo owner/repo \
  --db-path data/paper_scout.db \
  --tag weekly-backup
```

GitHub Actions에서 사용 (일요일):
```yaml
- name: Upload DB to Release
  if: github.event_name == 'schedule' && github.event.schedule == '0 0 * * 0'
  run: |
    python3 scripts/cache_manager.py upload \
      --repo ${{ github.repository }} \
      --db-path data/paper_scout.db \
      --tag weekly-backup
```

### 5. 오래된 Release asset 정리

```bash
python3 scripts/cache_manager.py cleanup \
  --repo owner/repo \
  --keep-weeks 4
```

GitHub Actions에서 사용:
```yaml
- name: Cleanup old Release assets
  if: github.event_name == 'schedule'
  run: |
    python3 scripts/cache_manager.py cleanup \
      --repo ${{ github.repository }} \
      --keep-weeks 4
```

### 6. DB 보장 (자동 복원/생성)

```bash
python3 scripts/cache_manager.py ensure \
  --repo owner/repo \
  --db-path data/paper_scout.db
```

복구 체인:
1. DB 파일이 존재하면 → 경로 반환
2. 존재하지 않으면 → Release asset에서 복원 시도
3. Release asset도 없으면 → 빈 DB 생성

## GitHub Actions 워크플로우 예제

```yaml
name: Daily Paper Collection

on:
  schedule:
    - cron: '0 0 * * *'  # 매일 자정
  workflow_dispatch:

jobs:
  collect:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Generate cache key
        id: cache-key
        run: |
          CACHE_KEY=$(python3 scripts/cache_manager.py cache-key --branch ${{ github.ref_name }})
          echo "cache-key=$CACHE_KEY" >> $GITHUB_OUTPUT

      - name: Restore DB from cache
        id: cache-db
        uses: actions/cache/restore@v4
        with:
          path: data/paper_scout.db
          key: ${{ steps.cache-key.outputs.cache-key }}

      - name: Fallback to Release asset
        if: steps.cache-db.outputs.cache-hit != 'true'
        run: |
          python3 scripts/cache_manager.py restore \
            --repo ${{ github.repository }} \
            --db-path data/paper_scout.db || true

      - name: Ensure DB exists
        run: |
          python3 scripts/cache_manager.py ensure \
            --repo ${{ github.repository }} \
            --db-path data/paper_scout.db

      - name: Run paper collection
        run: python3 main.py collect

      - name: Save DB to cache
        uses: actions/cache/save@v4
        if: always()
        with:
          path: data/paper_scout.db
          key: ${{ steps.cache-key.outputs.cache-key }}

  weekly-backup:
    runs-on: ubuntu-latest
    if: github.event_name == 'schedule' && github.event.schedule == '0 0 * * 0'
    needs: collect
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Restore DB
        uses: actions/cache/restore@v4
        with:
          path: data/paper_scout.db
          key: ${{ needs.collect.outputs.cache-key }}

      - name: Purge old data
        run: python3 scripts/purge_db.py --days 90

      - name: VACUUM DB
        run: sqlite3 data/paper_scout.db "VACUUM;"

      - name: Upload to Release
        run: |
          python3 scripts/cache_manager.py upload \
            --repo ${{ github.repository }} \
            --db-path data/paper_scout.db \
            --tag weekly-backup

      - name: Cleanup old Release assets
        run: |
          python3 scripts/cache_manager.py cleanup \
            --repo ${{ github.repository }} \
            --keep-weeks 4
```

## 함수 API

### `get_schema_hash(db_path: str | None = None) -> str`
스키마 버전 해시 반환 (8자)

### `get_cache_key(branch: str, schema_hash: str) -> str`
캐시 키 생성: `db-{branch}-{schema_hash}`

### `restore_from_release(repo: str, db_path: str) -> bool`
GitHub Release asset에서 DB 복원

### `upload_to_release(repo: str, db_path: str, tag: str) -> bool`
DB를 GitHub Release asset으로 업로드

### `cleanup_old_releases(repo: str, keep_weeks: int = 4) -> int`
오래된 Release asset 삭제, 삭제된 개수 반환

### `ensure_db(db_path: str, repo: str | None = None) -> str`
DB 존재 보장 (복원 또는 생성), DB 경로 반환

## 테스트

```bash
python3 -m pytest tests/test_cache_manager.py -v
```

## 의존성

- Python 3.9+
- `gh` CLI (GitHub Actions runner에 사전 설치됨)
- `core.storage.db_manager.DBManager` (빈 DB 생성용)

## 보안 고려사항

- Release asset 업로드/삭제에는 `gh` CLI 인증 필요
- GitHub Actions에서는 `GITHUB_TOKEN` 자동 제공
- 로컬 테스트 시 `gh auth login` 필요

## 문제 해결

### gh CLI 오류
```bash
# gh CLI 버전 확인
gh --version

# 인증 확인
gh auth status

# 재인증
gh auth login
```

### DB 복원 실패
```bash
# 로그 확인
python3 scripts/cache_manager.py restore --repo owner/repo --db-path data/paper_scout.db

# Release 목록 확인
gh release list -R owner/repo
```

### 캐시 키 미스매치
```bash
# 현재 캐시 키 확인
python3 scripts/cache_manager.py cache-key --branch main

# 스키마 해시 확인
python3 -c "from scripts.cache_manager import get_schema_hash; print(get_schema_hash())"
```
