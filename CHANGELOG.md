# Changelog

All notable changes to Paper Scout will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-02-16

### Added

#### Phase 1-3: Core Infrastructure

**Configuration System**
- Configuration loader with YAML parsing and environment variable substitution
- Schema validation for configuration structure
- Multi-environment configuration support
- Secure API key management through environment variables

**Database Infrastructure**
- SQLite database manager with connection pooling
- Schema migration system for version management
- Robust error handling and transaction management
- Automatic database initialization and schema validation

**JSON Parsing**
- Bracket-balancing JSON parser for robust LLM output parsing
- Handles malformed JSON from LLM responses
- Graceful error recovery with partial content extraction
- Support for streaming JSON parsing

**Deduplication System**
- Cross-run deduplication manager using paper identifiers
- Efficient duplicate detection algorithm
- Configurable deduplication strategies (full, partial, none)
- Database-backed persistence across workflow runs

**LLM Integration**
- OpenRouter client with OpenAI SDK compatibility
- Rate limiting and retry logic for API reliability
- Token usage tracking and monitoring
- Support for free-tier models
- Base agent class with prompt template system

#### Phase 4: Paper Discovery

**Preflight Checks**
- Configuration validation before pipeline execution
- API connectivity verification for LLM and arXiv services
- Resource availability checking
- Database integrity validation

**Agent 1: Keyword Expansion**
- Korean research description to English keyword expansion
- LLM-powered semantic keyword generation
- Domain-specific terminology extraction
- Configurable expansion parameters

**arXiv Integration**
- arXiv QueryBuilder for search query construction
- Boolean query composition with AND/OR operators
- Category filtering and date range specification
- Result limit configuration

#### Phase 5: Paper Collection and Filtering

**Code Detection**
- Repository URL pattern matching in paper abstracts
- GitHub, GitLab, and Bitbucket URL extraction
- Code availability flag for scoring system
- Configurable detection patterns

**Source Registry**
- Pluggable source adapter architecture
- arXiv source adapter implementation
- Extensible design for additional paper sources
- Unified paper metadata format

**Hybrid Filtering**
- Rule-based filtering with keyword matching
- Optional embedding-based similarity ranking
- Sentence-transformers integration for embeddings
- Configurable embedding models and topK selection
- Efficient filtering pipeline

**Agent 2: Paper Scoring**
- Relevance scoring on 0-100 scale
- Edge computing flag detection
- Real-time application flag detection
- Code availability confirmation
- Korean preference-based evaluation

#### Phase 6: Ranking and Summarization

**Ranker System**
- Deterministic scoring with configurable weights
- Bonus system for edge research, real-time applications, and code availability
- Threshold relaxation strategy (60 -> 50 -> 40)
- Minimum paper guarantee (5 papers per topic)

**Embedding Ranker**
- Cosine similarity calculation with topic embeddings
- Hybrid scoring combining LLM score, embedding score, and recency
- Optional embedding mode with graceful fallback
- Configurable weight distribution

**Agent 3: Summarization**
- Korean-language summary generation from English papers
- Key findings extraction
- Research significance explanation
- Concise summary formatting

**REMIND Selection**
- Top paper selection for detailed reporting
- Configurable selection count
- Score-based priority ranking
- Diversity consideration in selection

#### Phase 7: Output Generation

**JSON Generation**
- Structured JSON output with complete paper metadata
- Topic-specific result organization
- Timestamp and version information
- Machine-readable format for further processing

**Markdown Generation**
- Human-readable Markdown reports
- Table formatting for paper listings
- Score and flag visualization
- Cluster grouping in output

**HTML Generation**
- Static HTML site generation with Jinja2 templates
- Modern responsive design
- Paper clustering visualization
- Score-based color coding
- Mobile-friendly layout
- GitHub Pages deployment support

**GitHub Issue Integration**
- Daily summary issue creation and update
- Multi-topic aggregation in single issue
- Automatic issue closing and reopening
- Comment-based update history

**Notification System**
- Discord notifier with webhook integration
- Telegram notifier with bot API integration
- File attachment support for reports
- Per-topic notification channel configuration
- Error handling and retry logic

**Notifier Registry**
- Pluggable notifier architecture
- Easy addition of new notification channels
- Unified notifier interface
- Configuration-driven notifier selection

#### Phase 8: Pipeline Orchestration

**Topic Loop Orchestrator**
- Multi-topic pipeline execution
- Per-topic configuration and state management
- Error isolation between topics
- Progress tracking and logging

**Post-Loop Processing**
- GitHub Issue upsert after all topics complete
- Cross-topic statistics aggregation
- Final cleanup and validation
- Error summary reporting

**Main Entry Point**
- Command-line argument parsing
- Mode selection (full, dry-run)
- Date range specification
- Deduplication control
- Embedding toggle

**GitHub Actions Workflow**
- Scheduled daily execution at UTC 02:00 (KST 11:00)
- Manual workflow dispatch with parameters
- Secrets management for API keys and webhooks
- GitHub Pages deployment step
- Actions cache for database persistence

#### Phase 9: Maintenance and Analytics

**Database Cache Management**
- GitHub Actions cache integration for SQLite persistence
- Automatic cache key generation
- Cache restoration and saving logic
- Cache size optimization

**gh-pages Pruning**
- Old report cleanup script
- Configurable retention period
- Disk space management
- Orphaned file removal

**Weekly Tasks System**
- Weekly task scheduler for Sunday UTC 02:00
- Database purge for papers older than 90 days
- Trend summary generation across topics
- Update scan for new paper versions
- Optional visualization generation

**Database Maintenance**
- Old paper removal with configurable retention period
- Database vacuum and optimization
- Index rebuilding for performance
- Statistics recalculation

**Trend Analysis**
- Cross-topic keyword frequency analysis
- Emerging trend identification
- Weekly trend report generation
- Visualization of research trends

**Update Scanning**
- Detection of paper version updates on arXiv
- Comparison with previously collected versions
- Notification of significant updates
- Version history tracking

**Visualization (Optional)**
- Topic distribution charts with matplotlib
- Keyword co-occurrence graphs
- UMAP dimensionality reduction for paper clustering
- Interactive HTML visualizations
- Heatmap generation for trend analysis

#### Phase 10: Local Management

**CLI Commands**
- Topic list command with status display
- Topic add command with interactive prompts
- Topic edit command for existing topics
- Topic remove command with confirmation
- Dry-run command for testing without paper collection
- Full run command with all options

**Flask Web UI**
- Topic management interface with CRUD operations
- Pipeline execution with live log streaming
- Configuration viewer and editor
- Statistics dashboard with charts
- Topic enable/disable toggles
- Date range selection for runs
- Mode and deduplication controls

### Testing

**Comprehensive Test Suite**
- 123 Python test files covering all components
- 1283 total test cases for thorough validation
- Unit tests for individual functions and classes
- Integration tests for pipeline workflows
- Mock-based testing for external API calls
- Fixture-based test data management
- Test coverage reporting
- Continuous integration testing in GitHub Actions

### Documentation

**Project Documentation**
- Comprehensive README with Quick Start guide
- Architecture overview with pipeline diagram
- Configuration guide with examples
- CLI usage documentation
- Scoring system explanation
- Extension guide for developers
- arXiv API compliance notice

**Code Documentation**
- Docstrings for all public functions and classes
- Type hints for function signatures
- Inline comments for complex logic
- Configuration schema documentation

---

## [Unreleased]

### Planned

- Support for additional paper sources (Google Scholar, PubMed, Semantic Scholar)
- Advanced clustering algorithms (DBSCAN, hierarchical clustering)
- Multi-language summary generation (English, Japanese)
- Interactive data visualization dashboard
- Email notification support
- Slack integration
- Custom scoring rule configuration UI
- Paper recommendation system
- Citation network analysis
- Research collaboration detection

---

[1.0.0]: https://github.com/your-username/yks-paper-collector/releases/tag/v1.0.0
