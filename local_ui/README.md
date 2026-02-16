# Paper Scout Local UI

Flask-based local web interface for managing Paper Scout configuration and running pipelines.

## Features

- **Topics Management**: Create, update, and delete topics with real-time stats
- **Settings**: View environment variables and database status
- **Pipeline Control**: Run dry-runs and full pipeline executions (TASK-052)

## Installation

```bash
pip install -r requirements-ui.txt
```

## Usage

### Start Server (Programmatic)

```python
from local_ui.app import start_server

start_server(
    host="127.0.0.1",  # Localhost only (security)
    port=8585,
    open_browser=True,
    config_path="config.yaml",
    db_path="data/paper_scout.db"
)
```

### Start Server (CLI)

```bash
python3 -c "from local_ui.app import start_server; start_server()"
```

## Security

**CRITICAL**: The server binds to `127.0.0.1` only (localhost) by default. This prevents external network access and is a security requirement.

## API Endpoints

### Topics API

- `GET /api/topics` - List all topics with stats
- `POST /api/topics` - Create new topic
- `PUT /api/topics/<slug>` - Update topic
- `DELETE /api/topics/<slug>` - Delete topic

### Pipeline API

- `POST /api/pipeline/dry-run` - Execute dry-run (TASK-052)
- `POST /api/pipeline/run` - Execute full run
- `GET /api/pipeline/status` - Get pipeline status

### Settings API

- `GET /api/settings` - Get non-topic config
- `PUT /api/settings` - Update settings
- `GET /api/settings/env-status` - Get env vars (masked)
- `GET /api/settings/db-status` - Get DB stats

## Configuration

The UI uses `config.yaml` for all configuration. The `local_ui` section controls server behavior:

```yaml
local_ui:
  host: "127.0.0.1"  # Localhost only
  port: 8585
  open_browser: true
```

## Testing

```bash
python3 -m pytest tests/test_local_ui.py -v
```

## Architecture

```
local_ui/
├── __init__.py
├── app.py              # Flask app factory
├── config_io.py        # YAML read/write
├── api/
│   ├── __init__.py
│   ├── topics.py       # Topics CRUD
│   ├── pipeline.py     # Pipeline control
│   └── settings.py     # Settings management
├── static/             # Static assets (empty)
└── templates/
    └── index.html      # Single-page app
```

## Development Notes

- Uses Flask 3.1+ with JSON API and vanilla JS frontend
- Bootstrap 5 CDN for styling (no build tools needed)
- SQLite for stats queries (read-only in UI context)
- PyYAML for config operations (safe_load/safe_dump)

## Future Enhancements (TASK-052)

- Pipeline execution implementation
- Real-time progress tracking
- Search functionality
- Advanced filtering
