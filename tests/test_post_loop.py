"""Tests for PostLoopProcessor (TASK-037).

Covers:
- Cross-topic duplicate tagging (multi_topic)
- HTML build with mocked generators
- Notification dispatch with mock notifiers
- Git commit metadata (subprocess mocked)
- Cleanup of tmp directories
- Error isolation for each step
- Empty results handling
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, call, patch

import pytest

from core.pipeline.post_loop import PostLoopProcessor


# ---------------------------------------------------------------------------
# Test fixtures and helpers
# ---------------------------------------------------------------------------


@dataclass
class FakeNotifyConfig:
    """Minimal NotifyConfig for testing."""

    provider: str = "discord"
    channel_id: str = "123"
    secret_key: str = "TEST"


@dataclass
class FakeTopicSpec:
    """Minimal TopicSpec for testing."""

    slug: str = "topic-a"
    name: str = "Topic A"
    description: str = "Test topic A" * 10
    arxiv_categories: list = field(default_factory=lambda: ["cs.AI"])
    notify: FakeNotifyConfig = field(default_factory=FakeNotifyConfig)


@dataclass
class FakeEvaluation:
    """Minimal Evaluation for testing."""

    run_id: int = 1
    paper_key: str = "arxiv:2401.00001"
    llm_base_score: int = 80
    discarded: bool = False
    multi_topic: Optional[str] = None


@dataclass
class FakeRunMeta:
    """Minimal RunMeta for testing."""

    run_id: Optional[int] = 1
    topic_slug: str = "topic-a"
    display_date_kst: str = "2026-02-17"


@dataclass
class FakeAppConfig:
    """Minimal AppConfig for testing."""

    topics: list = field(default_factory=list)
    output: dict = field(default_factory=dict)
    notifications: dict = field(default_factory=dict)


def _make_config(topics: list | None = None) -> FakeAppConfig:
    """Create a minimal config with optional topics."""
    return FakeAppConfig(
        topics=topics or [],
        output={"template_dir": "templates", "report_dir": "tmp/reports"},
        notifications={},
    )


def _make_db_manager() -> MagicMock:
    """Create a mock DBManager."""
    db = MagicMock()
    db._conn = MagicMock()
    db.get_latest_completed_run = MagicMock(return_value=None)
    db.get_evaluations_by_run = MagicMock(return_value=[])
    return db


def _make_topic_results(
    completed: list | None = None,
    skipped: list | None = None,
    failed: list | None = None,
) -> Dict[str, Any]:
    """Create topic_results dict."""
    return {
        "topics_completed": completed or [],
        "topics_skipped": skipped or [],
        "topics_failed": failed or [],
    }


# ---------------------------------------------------------------------------
# Tests: Cross-topic duplicate tagging (_tag_multi_topic)
# ---------------------------------------------------------------------------


class TestTagMultiTopic:
    """Tests for the cross-topic duplicate tagging step."""

    def test_single_topic_no_tagging(self) -> None:
        """A single completed topic should produce zero tags."""
        db = _make_db_manager()
        processor = PostLoopProcessor(
            config=_make_config(), db_manager=db
        )

        completed = [{"slug": "topic-a", "total_output": 5}]
        result = processor._tag_multi_topic(completed)

        assert result == 0
        db._conn.execute.assert_not_called()

    def test_no_overlap(self) -> None:
        """Two topics with disjoint papers should produce zero tags."""
        db = _make_db_manager()

        run_a = FakeRunMeta(run_id=1, topic_slug="topic-a")
        run_b = FakeRunMeta(run_id=2, topic_slug="topic-b")

        db.get_latest_completed_run.side_effect = lambda slug: {
            "topic-a": run_a,
            "topic-b": run_b,
        }.get(slug)

        ev_a = FakeEvaluation(run_id=1, paper_key="arxiv:001")
        ev_b = FakeEvaluation(run_id=2, paper_key="arxiv:002")

        db.get_evaluations_by_run.side_effect = lambda rid: {
            1: [ev_a],
            2: [ev_b],
        }.get(rid, [])

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=db
        )

        completed = [
            {"slug": "topic-a", "total_output": 1},
            {"slug": "topic-b", "total_output": 1},
        ]
        result = processor._tag_multi_topic(completed)

        assert result == 0

    def test_single_paper_two_topics(self) -> None:
        """Same paper_key in two topics should get multi_topic tag."""
        db = _make_db_manager()

        run_a = FakeRunMeta(run_id=1, topic_slug="topic-a")
        run_b = FakeRunMeta(run_id=2, topic_slug="topic-b")

        db.get_latest_completed_run.side_effect = lambda slug: {
            "topic-a": run_a,
            "topic-b": run_b,
        }.get(slug)

        shared_key = "arxiv:shared001"
        ev_a = FakeEvaluation(run_id=1, paper_key=shared_key)
        ev_b = FakeEvaluation(run_id=2, paper_key=shared_key)

        db.get_evaluations_by_run.side_effect = lambda rid: {
            1: [ev_a],
            2: [ev_b],
        }.get(rid, [])

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=db
        )

        completed = [
            {"slug": "topic-a", "total_output": 1},
            {"slug": "topic-b", "total_output": 1},
        ]
        result = processor._tag_multi_topic(completed)

        assert result == 1
        # Verify the SQL updates were called
        assert db._conn.execute.call_count >= 2
        db._conn.commit.assert_called_once()

    def test_three_topics_overlap(self) -> None:
        """Paper in 3 topics should get all slugs in multi_topic."""
        db = _make_db_manager()

        runs = {
            "topic-a": FakeRunMeta(run_id=1, topic_slug="topic-a"),
            "topic-b": FakeRunMeta(run_id=2, topic_slug="topic-b"),
            "topic-c": FakeRunMeta(run_id=3, topic_slug="topic-c"),
        }
        db.get_latest_completed_run.side_effect = lambda s: runs.get(s)

        shared_key = "arxiv:triple001"
        db.get_evaluations_by_run.side_effect = lambda rid: [
            FakeEvaluation(run_id=rid, paper_key=shared_key)
        ]

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=db
        )

        completed = [
            {"slug": "topic-a", "total_output": 1},
            {"slug": "topic-b", "total_output": 1},
            {"slug": "topic-c", "total_output": 1},
        ]
        result = processor._tag_multi_topic(completed)

        assert result == 1

        # Verify the multi_topic value contains all three slugs (sorted)
        execute_calls = db._conn.execute.call_args_list
        update_calls = [
            c for c in execute_calls
            if "UPDATE" in str(c.args[0])
        ]
        assert len(update_calls) == 3

        # All updates should use the same comma-joined value
        expected_value = "topic-a, topic-b, topic-c"
        for uc in update_calls:
            assert uc.args[1][0] == expected_value

    def test_discarded_papers_excluded(self) -> None:
        """Discarded evaluations should not be considered for multi_topic."""
        db = _make_db_manager()

        runs = {
            "topic-a": FakeRunMeta(run_id=1, topic_slug="topic-a"),
            "topic-b": FakeRunMeta(run_id=2, topic_slug="topic-b"),
        }
        db.get_latest_completed_run.side_effect = lambda s: runs.get(s)

        shared_key = "arxiv:discarded001"
        ev_a = FakeEvaluation(
            run_id=1, paper_key=shared_key, discarded=False
        )
        ev_b = FakeEvaluation(
            run_id=2, paper_key=shared_key, discarded=True
        )

        db.get_evaluations_by_run.side_effect = lambda rid: {
            1: [ev_a],
            2: [ev_b],
        }.get(rid, [])

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=db
        )

        completed = [
            {"slug": "topic-a", "total_output": 1},
            {"slug": "topic-b", "total_output": 0},
        ]
        result = processor._tag_multi_topic(completed)

        # Paper only appears in one non-discarded topic
        assert result == 0


# ---------------------------------------------------------------------------
# Tests: HTML build (_build_html)
# ---------------------------------------------------------------------------


class TestBuildHtml:
    """Tests for the HTML build step."""

    @patch("core.pipeline.post_loop.PostLoopProcessor._load_latest_report_data")
    @patch("core.pipeline.post_loop.PostLoopProcessor._find_report_entry")
    def test_build_html_calls_generators(
        self, mock_find_entry: MagicMock, mock_load_data: MagicMock
    ) -> None:
        """HTML generators should be called with correct arguments."""
        config = _make_config(
            topics=[FakeTopicSpec(slug="topic-a", name="Topic A")]
        )
        db = _make_db_manager()

        mock_find_entry.return_value = {
            "topic_slug": "topic-a",
            "topic_name": "Topic A",
            "date": "2026-02-17",
            "filepath": "/tmp/reports/2026-02-17/20260217_paper_topic-a.html",
        }
        mock_load_data.return_value = {
            "meta": {"topic_slug": "topic-a", "date": "2026-02-17"},
            "papers": [],
            "clusters": [],
            "remind_papers": [],
        }

        processor = PostLoopProcessor(
            config=config, db_manager=db, report_dir="tmp/reports"
        )

        with patch(
            "output.render.html_generator.generate_index_html"
        ) as mock_index, patch(
            "output.render.html_generator.generate_latest_html"
        ) as mock_latest:
            mock_index.return_value = "/tmp/reports/index.html"
            mock_latest.return_value = "/tmp/reports/latest.html"

            processor._build_html(
                [{"slug": "topic-a", "total_output": 5}],
                "tmp/reports",
            )

            mock_index.assert_called_once()
            mock_latest.assert_called_once()

    @patch("core.pipeline.post_loop.PostLoopProcessor._load_latest_report_data")
    @patch("core.pipeline.post_loop.PostLoopProcessor._find_report_entry")
    def test_build_html_empty_results(
        self, mock_find_entry: MagicMock, mock_load_data: MagicMock
    ) -> None:
        """Empty completed list should not call generators."""
        config = _make_config()
        db = _make_db_manager()

        processor = PostLoopProcessor(
            config=config, db_manager=db, report_dir="tmp/reports"
        )

        with patch(
            "output.render.html_generator.generate_index_html"
        ) as mock_index, patch(
            "output.render.html_generator.generate_latest_html"
        ) as mock_latest:
            processor._build_html([], "tmp/reports")

            mock_index.assert_not_called()
            mock_latest.assert_not_called()


# ---------------------------------------------------------------------------
# Tests: Notification dispatch (_send_notifications)
# ---------------------------------------------------------------------------


class TestSendNotifications:
    """Tests for the notification dispatch step."""

    def test_notification_success(self) -> None:
        """Successful notification should increment sent counter."""
        topic = FakeTopicSpec(slug="topic-a", name="Topic A")
        config = _make_config(topics=[topic])
        db = _make_db_manager()
        db.get_latest_completed_run.return_value = FakeRunMeta(
            run_id=1, topic_slug="topic-a", display_date_kst="2026-02-17"
        )

        processor = PostLoopProcessor(
            config=config, db_manager=db, report_dir="tmp/reports"
        )

        with patch(
            "output.notifiers.registry.NotifierRegistry.get_notifier"
        ) as mock_get:
            mock_notifier = MagicMock()
            mock_notifier.notify.return_value = True
            mock_get.return_value = mock_notifier

            with patch.object(
                processor, "_load_latest_report_data", return_value=None
            ), patch.object(
                processor, "_find_report_files", return_value={}
            ):
                sent, failed = processor._send_notifications(
                    [{"slug": "topic-a", "total_output": 5}]
                )

        assert sent == 1
        assert failed == 0

    def test_notification_failure_isolated(self) -> None:
        """Failed notification should not prevent processing other topics."""
        topic_a = FakeTopicSpec(slug="topic-a", name="Topic A")
        topic_b = FakeTopicSpec(
            slug="topic-b",
            name="Topic B",
            notify=FakeNotifyConfig(
                provider="discord", channel_id="456", secret_key="TEST2"
            ),
        )
        config = _make_config(topics=[topic_a, topic_b])
        db = _make_db_manager()
        db.get_latest_completed_run.return_value = FakeRunMeta(
            run_id=1, display_date_kst="2026-02-17"
        )

        processor = PostLoopProcessor(
            config=config, db_manager=db, report_dir="tmp/reports"
        )

        call_count = 0

        def notifier_side_effect(notify_config: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            mock = MagicMock()
            if call_count == 1:
                # First topic fails
                mock.notify.side_effect = Exception("Discord API error")
            else:
                # Second topic succeeds
                mock.notify.return_value = True
            return mock

        with patch(
            "output.notifiers.registry.NotifierRegistry.get_notifier",
            side_effect=notifier_side_effect,
        ), patch.object(
            processor, "_load_latest_report_data", return_value=None
        ), patch.object(
            processor, "_find_report_files", return_value={}
        ):
            sent, failed = processor._send_notifications(
                [
                    {"slug": "topic-a", "total_output": 5},
                    {"slug": "topic-b", "total_output": 3},
                ]
            )

        assert sent == 1
        assert failed == 1

    def test_notification_zero_result(self) -> None:
        """Zero output should still send notification (zero-result message)."""
        topic = FakeTopicSpec(slug="topic-a", name="Topic A")
        config = _make_config(topics=[topic])
        db = _make_db_manager()
        db.get_latest_completed_run.return_value = FakeRunMeta(
            run_id=1, topic_slug="topic-a", display_date_kst="2026-02-17"
        )

        processor = PostLoopProcessor(
            config=config, db_manager=db, report_dir="tmp/reports"
        )

        with patch(
            "output.notifiers.registry.NotifierRegistry.get_notifier"
        ) as mock_get:
            mock_notifier = MagicMock()
            mock_notifier.notify.return_value = True
            mock_get.return_value = mock_notifier

            with patch.object(
                processor, "_load_latest_report_data", return_value=None
            ), patch.object(
                processor, "_find_report_files", return_value={}
            ):
                sent, failed = processor._send_notifications(
                    [{"slug": "topic-a", "total_output": 0}]
                )

        assert sent == 1
        assert failed == 0
        # Verify payload had total_output=0
        payload = mock_notifier.notify.call_args[0][0]
        assert payload.total_output == 0

    def test_notification_missing_topic_spec(self) -> None:
        """Missing topic spec should log warning and count as failed."""
        config = _make_config(topics=[])  # No topics configured
        db = _make_db_manager()

        processor = PostLoopProcessor(
            config=config, db_manager=db, report_dir="tmp/reports"
        )

        sent, failed = processor._send_notifications(
            [{"slug": "nonexistent", "total_output": 5}]
        )

        assert sent == 0
        assert failed == 1


# ---------------------------------------------------------------------------
# Tests: Git commit metadata (_git_commit_metadata)
# ---------------------------------------------------------------------------


class TestGitCommitMetadata:
    """Tests for the git commit metadata step."""

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_git_commit_success(
        self, mock_exists: MagicMock, mock_run: MagicMock
    ) -> None:
        """Successful git add + commit should return True."""
        mock_exists.return_value = True

        # git add succeeds, diff shows changes, commit succeeds
        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add
            MagicMock(returncode=1),  # git diff --cached --quiet (changes exist)
            MagicMock(returncode=0),  # git commit
        ]

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=_make_db_manager()
        )
        result = processor._git_commit_metadata()

        assert result is True
        assert mock_run.call_count == 3

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_git_commit_nothing_to_commit(
        self, mock_exists: MagicMock, mock_run: MagicMock
    ) -> None:
        """No staged changes should return True without committing."""
        mock_exists.return_value = True

        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add
            MagicMock(returncode=0),  # git diff --cached --quiet (no changes)
        ]

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=_make_db_manager()
        )
        result = processor._git_commit_metadata()

        assert result is True
        assert mock_run.call_count == 2

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_git_commit_failure_non_fatal(
        self, mock_exists: MagicMock, mock_run: MagicMock
    ) -> None:
        """Failed git commit should return False, not raise."""
        mock_exists.return_value = True

        mock_run.side_effect = [
            MagicMock(returncode=0),  # git add
            MagicMock(returncode=1),  # changes exist
            MagicMock(
                returncode=1, stderr="error: commit failed"
            ),  # commit fails
        ]

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=_make_db_manager()
        )
        result = processor._git_commit_metadata()

        assert result is False

    @patch("os.path.exists")
    def test_git_commit_no_files_to_commit(
        self, mock_exists: MagicMock
    ) -> None:
        """When no metadata files exist, should return True."""
        mock_exists.return_value = False

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=_make_db_manager()
        )
        result = processor._git_commit_metadata()

        assert result is True

    @patch("subprocess.run")
    @patch("os.path.exists")
    def test_git_timeout_returns_false(
        self, mock_exists: MagicMock, mock_run: MagicMock
    ) -> None:
        """Subprocess timeout should return False."""
        import subprocess as sp

        mock_exists.return_value = True
        mock_run.side_effect = sp.TimeoutExpired(cmd="git", timeout=30)

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=_make_db_manager()
        )
        result = processor._git_commit_metadata()

        assert result is False


# ---------------------------------------------------------------------------
# Tests: Cleanup (_cleanup_tmp)
# ---------------------------------------------------------------------------


class TestCleanupTmp:
    """Tests for the tmp cleanup step."""

    def test_cleanup_removes_debug_dir(self) -> None:
        """tmp/debug should be removed if it exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = os.path.join(tmpdir, "reports")
            debug_dir = os.path.join(tmpdir, "debug")
            os.makedirs(report_dir, exist_ok=True)
            os.makedirs(debug_dir, exist_ok=True)

            # Create a file inside debug dir
            with open(os.path.join(debug_dir, "test.log"), "w") as f:
                f.write("debug log")

            processor = PostLoopProcessor(
                config=_make_config(), db_manager=_make_db_manager()
            )
            processor._cleanup_tmp(report_dir)

            assert not os.path.exists(debug_dir)
            assert os.path.exists(report_dir)  # preserved

    def test_cleanup_no_debug_dir(self) -> None:
        """Should not raise when debug dir does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = os.path.join(tmpdir, "reports")
            os.makedirs(report_dir, exist_ok=True)

            processor = PostLoopProcessor(
                config=_make_config(), db_manager=_make_db_manager()
            )
            # Should not raise
            processor._cleanup_tmp(report_dir)

            assert os.path.exists(report_dir)

    def test_cleanup_preserves_report_dir(self) -> None:
        """Report directory should never be deleted."""
        with tempfile.TemporaryDirectory() as tmpdir:
            report_dir = os.path.join(tmpdir, "reports")
            os.makedirs(report_dir, exist_ok=True)

            # Create a file inside report dir
            with open(
                os.path.join(report_dir, "index.html"), "w"
            ) as f:
                f.write("<html></html>")

            processor = PostLoopProcessor(
                config=_make_config(), db_manager=_make_db_manager()
            )
            processor._cleanup_tmp(report_dir)

            assert os.path.exists(report_dir)
            assert os.path.exists(
                os.path.join(report_dir, "index.html")
            )


# ---------------------------------------------------------------------------
# Tests: Full process() integration
# ---------------------------------------------------------------------------


class TestProcessIntegration:
    """Tests for the full process() method."""

    @patch.object(PostLoopProcessor, "_cleanup_tmp")
    @patch.object(PostLoopProcessor, "_git_commit_metadata")
    @patch.object(PostLoopProcessor, "_send_notifications")
    @patch.object(PostLoopProcessor, "_build_html")
    @patch.object(PostLoopProcessor, "_tag_multi_topic")
    def test_process_calls_all_steps(
        self,
        mock_tag: MagicMock,
        mock_html: MagicMock,
        mock_notify: MagicMock,
        mock_git: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        """All 5 post-loop steps should be called in order."""
        mock_tag.return_value = 3
        mock_notify.return_value = (2, 1)
        mock_git.return_value = True

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=_make_db_manager()
        )
        topic_results = _make_topic_results(
            completed=[
                {"slug": "topic-a", "total_output": 5},
                {"slug": "topic-b", "total_output": 3},
            ]
        )

        summary = processor.process(topic_results)

        mock_tag.assert_called_once()
        mock_html.assert_called_once()
        mock_notify.assert_called_once()
        mock_git.assert_called_once()
        mock_cleanup.assert_called_once()

        assert summary["multi_topic_tagged"] == 3
        assert summary["html_built"] is True
        assert summary["notifications_sent"] == 2
        assert summary["notifications_failed"] == 1
        assert summary["git_committed"] is True
        assert summary["cleanup_done"] is True

    @patch.object(PostLoopProcessor, "_cleanup_tmp")
    @patch.object(PostLoopProcessor, "_git_commit_metadata")
    @patch.object(PostLoopProcessor, "_send_notifications")
    @patch.object(PostLoopProcessor, "_build_html")
    @patch.object(PostLoopProcessor, "_tag_multi_topic")
    def test_process_empty_results(
        self,
        mock_tag: MagicMock,
        mock_html: MagicMock,
        mock_notify: MagicMock,
        mock_git: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        """Empty results should still call all steps without error."""
        mock_tag.return_value = 0
        mock_notify.return_value = (0, 0)
        mock_git.return_value = True

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=_make_db_manager()
        )
        topic_results = _make_topic_results()

        summary = processor.process(topic_results)

        assert summary["multi_topic_tagged"] == 0
        assert summary["notifications_sent"] == 0
        assert summary["notifications_failed"] == 0

    @patch.object(PostLoopProcessor, "_cleanup_tmp")
    @patch.object(PostLoopProcessor, "_git_commit_metadata")
    @patch.object(PostLoopProcessor, "_send_notifications")
    @patch.object(PostLoopProcessor, "_build_html")
    @patch.object(PostLoopProcessor, "_tag_multi_topic")
    def test_process_step_failure_continues(
        self,
        mock_tag: MagicMock,
        mock_html: MagicMock,
        mock_notify: MagicMock,
        mock_git: MagicMock,
        mock_cleanup: MagicMock,
    ) -> None:
        """Failure in one step should not prevent other steps."""
        mock_tag.side_effect = Exception("DB error")
        mock_html.side_effect = Exception("Template error")
        mock_notify.return_value = (1, 0)
        mock_git.return_value = True

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=_make_db_manager()
        )
        topic_results = _make_topic_results(
            completed=[{"slug": "topic-a", "total_output": 5}]
        )

        summary = processor.process(topic_results)

        # Steps 1 and 2 failed but 3, 4, 5 should still run
        assert summary["multi_topic_tagged"] == 0  # default on error
        assert summary["html_built"] is False  # failed
        assert summary["notifications_sent"] == 1  # succeeded
        assert summary["git_committed"] is True
        assert summary["cleanup_done"] is True

        # All steps were called
        mock_tag.assert_called_once()
        mock_html.assert_called_once()
        mock_notify.assert_called_once()
        mock_git.assert_called_once()
        mock_cleanup.assert_called_once()


# ---------------------------------------------------------------------------
# Tests: Helper methods
# ---------------------------------------------------------------------------


class TestHelpers:
    """Tests for internal helper methods."""

    def test_find_topic_name_found(self) -> None:
        """Should return topic name when slug matches."""
        topic = FakeTopicSpec(slug="topic-a", name="My Topic")
        config = _make_config(topics=[topic])
        db = _make_db_manager()

        processor = PostLoopProcessor(config=config, db_manager=db)
        assert processor._find_topic_name("topic-a") == "My Topic"

    def test_find_topic_name_not_found(self) -> None:
        """Should return slug when no topic matches."""
        config = _make_config(topics=[])
        db = _make_db_manager()

        processor = PostLoopProcessor(config=config, db_manager=db)
        assert processor._find_topic_name("missing") == "missing"

    def test_find_topic_spec_returns_none(self) -> None:
        """Should return None when slug not in config."""
        config = _make_config(topics=[])
        db = _make_db_manager()

        processor = PostLoopProcessor(config=config, db_manager=db)
        assert processor._find_topic_spec("missing") is None

    def test_find_report_files_with_temp_dir(self) -> None:
        """Should discover report files by format extension."""
        with tempfile.TemporaryDirectory() as tmpdir:
            date_dir = os.path.join(tmpdir, "2026-02-17")
            os.makedirs(date_dir)

            # Create mock report files
            for ext in ["html", "md", "json"]:
                path = os.path.join(
                    date_dir, f"20260217_paper_topic-a.{ext}"
                )
                with open(path, "w") as f:
                    f.write(f"content.{ext}")

            processor = PostLoopProcessor(
                config=_make_config(), db_manager=_make_db_manager()
            )
            result = processor._find_report_files("topic-a", tmpdir)

            assert "html" in result
            assert "md" in result
            assert "json" in result

    def test_get_display_date_from_run(self) -> None:
        """Should return display_date_kst from latest run."""
        db = _make_db_manager()
        db.get_latest_completed_run.return_value = FakeRunMeta(
            run_id=1,
            topic_slug="topic-a",
            display_date_kst="2026-02-17",
        )

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=db
        )
        result = processor._get_display_date("topic-a")
        assert result == "2026-02-17"

    def test_get_display_date_no_run(self) -> None:
        """Should return empty string when no completed run exists."""
        db = _make_db_manager()
        db.get_latest_completed_run.return_value = None

        processor = PostLoopProcessor(
            config=_make_config(), db_manager=db
        )
        result = processor._get_display_date("topic-a")
        assert result == ""
