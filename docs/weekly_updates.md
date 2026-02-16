# Weekly Update Scan

## Overview

The Weekly Update Scan feature identifies papers that were recently updated but originally published earlier. This helps track significant revisions to existing papers that might warrant re-evaluation.

## Target Conditions

A paper is included in the weekly update report if it meets ALL of the following conditions:

1. **Recently Updated**: `updated_at_utc` is within the last 7 days
2. **Not Newly Published**: `published_at_utc` is more than 7 days ago
3. **Within Data Retention**: `published_at_utc` is within 90 days (still has Evaluation data before purge)
4. **Has Evaluation**: Existing Evaluation record exists in `paper_evaluations` table with `discarded = 0`

## Output Format

### File Naming

- Markdown: `YYYYMMDD_weekly_updates.md`
- HTML: `YYYYMMDD_weekly_updates.html`

### gh-pages Structure

Reports are stored under `reports/YYYY-MM-DD/` directory, same as `weekly_summary`.

### Report Content

Each paper entry includes:

- Paper title
- arXiv ID (linked to paper URL)
- Original evaluation score (`llm_base_score`)
- "Updated" marker with `updated_at_utc` date
- Topic slug

**Note**: Abstract and title diffs are NOT included (cost vs benefit too low per devspec).

## Usage

### Quick Start

```python
from core.pipeline.weekly_updates import generate_update_report

# Generate reports for today
result = generate_update_report(
    db_path="data/paper_scout.db",
    date_str="2024-02-01",
    output_dir="tmp/reports"
)

if result is None:
    print("No updated papers found")
else:
    print(f"Reports generated in: {result}")
```

### Advanced Usage

```python
from core.pipeline.weekly_updates import (
    scan_updated_papers,
    render_updates_md,
    render_updates_html,
)

# Step 1: Scan for papers
papers = scan_updated_papers("data/paper_scout.db", "2024-02-01")

# Step 2: Generate custom reports
if papers:
    # Markdown report
    md_content = render_updates_md(papers, "2024-02-01")

    # HTML report
    html_content = render_updates_html(papers, "2024-02-01")

    # Process results
    for paper in papers:
        print(f"Updated: {paper['title']}")
        print(f"  Score: {paper['llm_base_score']}")
        print(f"  Topic: {paper['topic_slug']}")
```

## API Reference

### `scan_updated_papers(db_path, reference_date)`

Scan database for papers matching weekly update criteria.

**Parameters:**
- `db_path` (str): Path to SQLite database file
- `reference_date` (str): Reference date in YYYY-MM-DD format (UTC)

**Returns:**
- `list[dict]`: List of paper dictionaries with keys:
  - `paper_key`: Unique paper identifier
  - `title`: Paper title
  - `url`: Paper URL
  - `native_id`: Source-specific ID (e.g., arXiv ID)
  - `llm_base_score`: Original evaluation score
  - `updated_at_utc`: Update timestamp
  - `published_at_utc`: Publication timestamp
  - `topic_slug`: Topic identifier

**SQL Query:**
```sql
SELECT DISTINCT
    p.paper_key,
    p.title,
    p.url,
    p.native_id,
    pe.llm_base_score,
    p.updated_at_utc,
    p.published_at_utc,
    r.topic_slug
FROM papers p
JOIN paper_evaluations pe ON p.paper_key = pe.paper_key
JOIN runs r ON pe.run_id = r.run_id
WHERE p.updated_at_utc >= :seven_days_ago
  AND p.published_at_utc < :seven_days_ago
  AND p.published_at_utc >= :ninety_days_ago
  AND pe.discarded = 0
ORDER BY p.updated_at_utc DESC
```

### `render_updates_md(papers, date_str)`

Render update scan results to Markdown format.

**Parameters:**
- `papers` (list[dict]): Paper list from `scan_updated_papers`
- `date_str` (str): Date string for report heading (YYYY-MM-DD)

**Returns:**
- `str`: Markdown formatted report

**Example Output:**
```markdown
# Weekly Updates Report - 2024-02-01

Found 2 paper(s) with recent updates.

---

## Deep Learning for Edge Computing

**arXiv ID**: [2401.12345](https://arxiv.org/abs/2401.12345)
**Original Score**: 85
**Updated**: 2024-01-29T10:30:00+00:00
**Topic**: edge-computing

---
```

### `render_updates_html(papers, date_str)`

Render update scan results to HTML format with inline styles.

**Parameters:**
- `papers` (list[dict]): Paper list from `scan_updated_papers`
- `date_str` (str): Date string for report heading (YYYY-MM-DD)

**Returns:**
- `str`: HTML formatted report with inline CSS

**Styling:**
- Responsive design (mobile-friendly)
- Clean card-based layout
- Color-coded badges for score and update status
- Clickable arXiv links

### `generate_update_report(db_path, date_str, output_dir)`

Full workflow: scan → render both formats → write files.

**Parameters:**
- `db_path` (str): Path to SQLite database file
- `date_str` (str): Report date in YYYY-MM-DD format
- `output_dir` (str): Directory to save reports (default: "tmp/reports")

**Returns:**
- `str | None`: Output directory path if papers found, None otherwise

**Side Effects:**
- Creates `output_dir` if it doesn't exist
- Writes `YYYYMMDD_weekly_updates.md`
- Writes `YYYYMMDD_weekly_updates.html`

## Database Schema

### Tables Used

**papers:**
- `paper_key` (PRIMARY KEY)
- `native_id`
- `url`
- `title`
- `published_at_utc` (ISO 8601 string)
- `updated_at_utc` (ISO 8601 string)

**paper_evaluations:**
- `run_id` (part of composite PK)
- `paper_key` (part of composite PK)
- `llm_base_score`
- `discarded`

**runs:**
- `run_id` (PRIMARY KEY)
- `topic_slug`

## Integration

### With Weekly Pipeline

The weekly update scan can be integrated into the weekly pipeline after the main summary generation:

```python
# After weekly summary generation
from core.pipeline.weekly_updates import generate_update_report

# Generate update report
update_result = generate_update_report(
    db_path=db.db_path,
    date_str=reference_date,
    output_dir="tmp/reports"
)

if update_result:
    # Optionally publish to gh-pages
    # Optionally send notification
    pass
```

### Notification

Use the same channel as `weekly_summary` for notifications:

```python
if update_result:
    # Send notification with link to update report
    notifier.send_message(
        f"Weekly Updates Report: {len(papers)} papers updated\n"
        f"View report: https://example.com/reports/{date_str}/weekly_updates.html"
    )
```

## Testing

Comprehensive test coverage in `tests/test_weekly_updates.py`:

- ✅ Scan finds papers matching all 4 conditions
- ✅ Scan excludes newly published papers (published within 7 days)
- ✅ Scan excludes old papers (published > 90 days)
- ✅ Scan excludes papers without evaluations
- ✅ Scan excludes discarded evaluations
- ✅ Scan handles papers in multiple topics
- ✅ Markdown render with empty and populated lists
- ✅ HTML render with empty and populated lists
- ✅ File generation with None return when empty
- ✅ File generation with proper output when papers exist
- ✅ Empty database returns empty list
- ✅ Date boundary edge cases (exactly 7 days, exactly 90 days)

Run tests:
```bash
python3 -m pytest tests/test_weekly_updates.py -v
```

## Performance

- **Query Complexity**: O(n) with indexed joins on `paper_key` and `run_id`
- **Memory Usage**: Linear with number of matching papers (typically < 100)
- **File I/O**: Two writes per execution (markdown + HTML)
- **Database Load**: Single SELECT query with time-based filtering

## Limitations

1. **No Abstract Diffs**: Cost vs benefit too low per devspec 9-5
2. **No Title Diffs**: Would require storing historical versions
3. **90-Day Window**: Papers older than 90 days excluded (data purge policy)
4. **Single Evaluation**: Returns first evaluation per paper (may have multiple topics)

## Future Enhancements

Potential improvements (not in current scope):

- Paper update changelog tracking
- Abstract diff highlighting
- Title change detection
- Update frequency analysis
- Automated significance scoring for updates
- Email digest format
- RSS feed for updates
