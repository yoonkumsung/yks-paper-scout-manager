"""Post-loop processing for Paper Scout (TASK-037).

Handles all work after the per-topic loop completes (DevSpec Section 9-4):
1. Cross-topic duplicate tagging (multi_topic field)
2. HTML build (index.html + latest.html)
3. Notification dispatch (per-topic channel, file attachment + fallback)
4. Git commit (text metadata only)
5. tmp/ cleanup

GitHub Issue upsert is handled per-topic in step 11 and skipped here.
gh-pages deploy and cache save are handled by CI, not Python.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
from collections import defaultdict
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class PostLoopProcessor:
    """Process all work after the per-topic loop completes.

    Receives the aggregated topic_results from
    ``TopicLoopOrchestrator.run_all_topics()`` and performs cross-topic
    operations that require knowledge of all completed topics.

    All steps are best-effort: failures are logged but do not propagate
    to the caller (except in ``_tag_multi_topic`` which is data-critical).
    """

    def __init__(
        self,
        config: Any,  # AppConfig
        db_manager: Any,  # DBManager
        report_dir: str = "tmp/reports",
    ) -> None:
        """Initialize the post-loop processor.

        Args:
            config: Full application configuration (AppConfig).
            db_manager: Database manager for CRUD operations.
            report_dir: Directory containing generated reports.
        """
        self._config = config
        self._db = db_manager
        self._report_dir = report_dir

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(self, topic_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run all post-loop processing steps.

        Args:
            topic_results: Output from ``TopicLoopOrchestrator.run_all_topics()``
                with keys ``topics_completed``, ``topics_skipped``,
                ``topics_failed``.

        Returns:
            Summary dict with per-step results:
            {
                "multi_topic_tagged": int,
                "html_built": bool,
                "notifications_sent": int,
                "notifications_failed": int,
                "git_committed": bool,
                "cleanup_done": bool,
            }
        """
        summary: Dict[str, Any] = {
            "multi_topic_tagged": 0,
            "html_built": False,
            "notifications_sent": 0,
            "notifications_failed": 0,
            "git_committed": False,
            "cleanup_done": False,
        }

        completed = topic_results.get("topics_completed", [])

        # Step 1: Cross-topic duplicate tagging (data-critical)
        try:
            tagged = self._tag_multi_topic(completed)
            summary["multi_topic_tagged"] = tagged
            logger.info(
                "Post-loop step 1: tagged %d papers as multi_topic", tagged
            )
        except Exception:
            logger.error(
                "Post-loop step 1: multi_topic tagging failed "
                "(data-critical step)",
                exc_info=True,
            )
            raise

        # Step 2: HTML build (index + latest)
        try:
            self._build_html(completed, self._report_dir)
            summary["html_built"] = True
            logger.info("Post-loop step 2: HTML build complete")
        except Exception:
            logger.error(
                "Post-loop step 2: HTML build failed", exc_info=True
            )

        # Step 2.5: Deploy to gh-pages (before notifications so link is live)
        gh_pages_cfg = self._config.output.get("gh_pages", {})
        if gh_pages_cfg.get("enabled") and summary["html_built"]:
            try:
                deployed = self._deploy_gh_pages(self._report_dir, gh_pages_cfg)
                summary["gh_pages_deployed"] = deployed
                logger.info(
                    "Post-loop step 2.5: gh-pages deploy %s",
                    "succeeded" if deployed else "skipped",
                )
            except Exception:
                summary["gh_pages_deployed"] = False
                logger.warning(
                    "Post-loop step 2.5: gh-pages deploy failed (non-fatal)",
                    exc_info=True,
                )

        # Step 3: Notification dispatch
        try:
            sent, failed = self._send_notifications(completed)
            summary["notifications_sent"] = sent
            summary["notifications_failed"] = failed
            logger.info(
                "Post-loop step 3: notifications sent=%d, failed=%d",
                sent,
                failed,
            )
        except Exception:
            logger.error(
                "Post-loop step 3: notification dispatch failed",
                exc_info=True,
            )

        # Step 4: Git commit metadata
        try:
            committed = self._git_commit_metadata()
            summary["git_committed"] = committed
            logger.info(
                "Post-loop step 4: git commit %s",
                "succeeded" if committed else "skipped (nothing to commit)",
            )
        except Exception:
            logger.warning(
                "Post-loop step 4: git commit failed (non-fatal)",
                exc_info=True,
            )

        # Step 5: Cleanup
        try:
            self._cleanup_tmp(self._report_dir)
            summary["cleanup_done"] = True
            logger.info("Post-loop step 5: cleanup complete")
        except Exception:
            logger.warning(
                "Post-loop step 5: cleanup failed (non-fatal)",
                exc_info=True,
            )

        return summary

    # ------------------------------------------------------------------
    # Step 1: Cross-topic duplicate tagging
    # ------------------------------------------------------------------

    def _tag_multi_topic(self, completed: List[Dict[str, Any]]) -> int:
        """Tag papers that appear in multiple topics with multi_topic field.

        Builds a paper_key -> list[slug] mapping across all completed
        topics.  Papers appearing in 2+ topics get multi_topic set to a
        comma-joined slug list.

        Args:
            completed: List of dicts with ``slug`` and ``total_output``.

        Returns:
            Number of papers tagged as multi_topic.
        """
        if len(completed) < 2:
            return 0

        # Collect paper_key -> list of slugs across all completed topics
        paper_to_slugs: Dict[str, List[str]] = defaultdict(list)

        for topic_info in completed:
            slug = topic_info["slug"]
            # Get the latest completed run for this topic
            run_meta = self._db.get_latest_completed_run(slug)
            if run_meta is None or run_meta.run_id is None:
                continue

            evaluations = self._db.get_evaluations_by_run(run_meta.run_id)
            for ev in evaluations:
                if not ev.discarded:
                    paper_to_slugs[ev.paper_key].append(slug)

        # Find papers appearing in multiple topics
        tagged_count = 0
        for paper_key, slugs in paper_to_slugs.items():
            if len(slugs) < 2:
                continue

            multi_topic_value = ", ".join(sorted(slugs))

            # Update evaluations for this paper_key across all runs
            for slug in slugs:
                run_meta = self._db.get_latest_completed_run(slug)
                if run_meta is None or run_meta.run_id is None:
                    continue

                self._db.update_evaluation_multi_topic(
                    run_meta.run_id, paper_key, multi_topic_value
                )

            tagged_count += 1

        self._db.commit()
        return tagged_count

    # ------------------------------------------------------------------
    # Step 2: HTML build
    # ------------------------------------------------------------------

    def _build_html(
        self,
        completed: List[Dict[str, Any]],
        report_dir: str,
    ) -> None:
        """Build index.html and latest.html from completed topic reports.

        Per-topic HTML files are already generated in step 10 of the
        topic loop.  This step creates the cross-topic index page and
        the latest redirect page.

        Args:
            completed: List of dicts with ``slug`` and ``total_output``.
            report_dir: Directory containing report files.
        """
        from output.render.html_generator import (
            generate_index_html,
            generate_latest_html,
        )

        template_dir = self._config.output.get("template_dir", "templates")

        # Build reports list for index page
        reports: List[Dict[str, str]] = []
        latest_report_data: Optional[Dict[str, Any]] = None

        for topic_info in completed:
            slug = topic_info["slug"]
            # Find the topic spec for the name
            topic_name = self._find_topic_name(slug)

            # Look for the most recent report file for this topic
            report_entry = self._find_report_entry(slug, topic_name, report_dir)
            if report_entry:
                reports.append(report_entry)

            # Use the last completed topic's report data for latest.html
            report_data = self._load_latest_report_data(slug, report_dir)
            if report_data is not None:
                latest_report_data = report_data

        # Generate index.html
        if reports:
            generate_index_html(
                reports=reports,
                output_dir=report_dir,
                template_dir=template_dir,
            )

        # Generate latest.html from the last completed topic
        if latest_report_data is not None:
            generate_latest_html(
                report_data=latest_report_data,
                output_dir=report_dir,
                template_dir=template_dir,
                read_sync=(self._config.read_sync or None)
                if self._config.output.get("attachments", {}).get(
                    "include_read_sync", True
                )
                else None,
            )

    # ------------------------------------------------------------------
    # Step 3: Notification dispatch
    # ------------------------------------------------------------------

    def _send_notifications(
        self, completed: List[Dict[str, Any]]
    ) -> tuple[int, int]:
        """Send notifications for each completed topic.

        Failures are isolated: a failed notification for one topic does
        not block notifications for other topics.

        Args:
            completed: List of dicts with ``slug`` and ``total_output``.

        Returns:
            Tuple of (sent_count, failed_count).
        """
        from output.notifiers.base import NotifyPayload
        from output.notifiers.registry import NotifierRegistry

        registry = NotifierRegistry()
        sent = 0
        failed = 0

        for topic_info in completed:
            slug = topic_info["slug"]
            total_output = topic_info.get("total_output", 0)

            try:
                # Find the topic spec
                topic_spec = self._find_topic_spec(slug)
                if topic_spec is None:
                    logger.warning(
                        "Topic spec not found for '%s', skipping notification",
                        slug,
                    )
                    failed += 1
                    continue

                if not topic_spec.notify:
                    logger.info(
                        "Topic '%s': no notify config, skipping notification",
                        slug,
                    )
                    continue

                # Get notifiers for the "complete" event
                notifiers = registry.get_notifiers_for_event(
                    topic_spec.notify, "complete"
                )
                if not notifiers:
                    logger.info(
                        "Topic '%s': no notifiers for 'complete' event",
                        slug,
                    )
                    continue

                # Build display date
                display_date = self._get_display_date(slug)

                # Extract keywords from latest run
                keywords = self._get_keywords_for_topic(slug)

                # Find report file paths
                file_paths = self._find_report_files(slug, self._report_dir)

                # Build gh_pages URL if configured
                gh_pages_cfg = self._config.output.get("gh_pages", {})
                gh_pages_url = None
                notify_mode = "file"
                if gh_pages_cfg.get("enabled") and gh_pages_cfg.get("base_url"):
                    base_url = gh_pages_cfg["base_url"].rstrip("/")
                    gh_pages_url = f"{base_url}/latest.html"
                    notify_mode = gh_pages_cfg.get("notify_mode", "file")

                attachments_cfg = self._config.output.get("attachments", {})
                payload = NotifyPayload(
                    topic_slug=slug,
                    topic_name=topic_spec.name,
                    display_date=display_date,
                    keywords=keywords,
                    total_output=total_output,
                    file_paths=file_paths,
                    gh_pages_url=gh_pages_url,
                    notify_mode=notify_mode,
                    allowed_formats=attachments_cfg.get("formats", ["html", "md"]),
                    event_type="complete",
                )

                for notifier in notifiers:
                    success = notifier.notify(payload)
                    if success:
                        sent += 1
                    else:
                        failed += 1

            except Exception:
                logger.warning(
                    "Notification failed for topic '%s'",
                    slug,
                    exc_info=True,
                )
                failed += 1

        return sent, failed

    # ------------------------------------------------------------------
    # Step 4: Git commit metadata
    # ------------------------------------------------------------------

    def _git_commit_metadata(self) -> bool:
        """Commit text metadata files to git.

        Files committed:
        - data/seen_items.jsonl
        - data/issue_map.json
        - data/usage/
        - data/keyword_cache/
        - data/model_caps/
        - data/weekly_done.flag
        - data/last_success.json

        Returns:
            True if commit succeeded or nothing to commit,
            False on failure.
        """
        metadata_paths = [
            "data/seen_items.jsonl",
            "data/issue_map.json",
            "data/usage/",
            "data/keyword_cache.json",
            "data/model_caps.json",
            "data/weekly_done.flag",
            "data/last_success.json",
        ]

        # Filter to existing paths only
        existing = [p for p in metadata_paths if os.path.exists(p)]
        if not existing:
            logger.info("No metadata files to commit")
            return True

        try:
            # Ensure git user config is set (needed in CI)
            subprocess.run(
                ["git", "config", "user.name", "paper-scout[bot]"],
                capture_output=True, text=True, timeout=5,
            )
            subprocess.run(
                ["git", "config", "user.email",
                 "paper-scout[bot]@users.noreply.github.com"],
                capture_output=True, text=True, timeout=5,
            )

            # Stage files
            cmd_add = ["git", "add"] + existing
            result = subprocess.run(
                cmd_add,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode != 0:
                logger.warning(
                    "git add failed: %s", result.stderr.strip()
                )
                return False

            # Check if there are staged changes
            result_diff = subprocess.run(
                ["git", "diff", "--cached", "--quiet"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result_diff.returncode == 0:
                logger.info("No staged changes to commit")
                return True

            # Commit
            result_commit = subprocess.run(
                [
                    "git",
                    "commit",
                    "-m",
                    "chore: update metadata [skip ci]",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result_commit.returncode != 0:
                logger.warning(
                    "git commit failed: %s",
                    result_commit.stderr.strip(),
                )
                return False

            return True

        except subprocess.TimeoutExpired:
            logger.warning("git command timed out")
            return False
        except FileNotFoundError:
            logger.warning("git not found in PATH")
            return False

    # ------------------------------------------------------------------
    # Step 2.5: Deploy to gh-pages
    # ------------------------------------------------------------------

    def _deploy_gh_pages(
        self, report_dir: str, gh_pages_cfg: dict
    ) -> bool:
        """Deploy report files to the gh-pages branch.

        Uses git subtree-based approach:
        1. Create/checkout orphan gh-pages branch worktree
        2. Copy report files
        3. Commit and push

        Falls back gracefully if git operations fail.

        Args:
            report_dir: Directory containing generated HTML reports.
            gh_pages_cfg: gh_pages config section with enabled, base_url, etc.

        Returns:
            True if deploy succeeded, False otherwise.
        """
        if not os.path.isdir(report_dir):
            logger.info("gh-pages: report_dir %s not found, skipping", report_dir)
            return False

        try:
            import tempfile

            branch = "gh-pages"
            keep_files = gh_pages_cfg.get("keep_files", True)

            # Check if gh-pages branch exists on remote
            result = subprocess.run(
                ["git", "ls-remote", "--heads", "origin", branch],
                capture_output=True,
                text=True,
                timeout=30,
            )
            branch_exists = branch in result.stdout

            # Create temp directory for worktree
            with tempfile.TemporaryDirectory(prefix="gh-pages-") as tmpdir:
                worktree_dir = os.path.join(tmpdir, "deploy")

                if branch_exists:
                    # Checkout existing branch
                    subprocess.run(
                        ["git", "worktree", "add", worktree_dir, branch],
                        capture_output=True,
                        text=True,
                        timeout=30,
                        check=True,
                    )
                else:
                    # Create orphan branch
                    subprocess.run(
                        [
                            "git",
                            "worktree",
                            "add",
                            "--orphan",
                            worktree_dir,
                            branch,
                        ],
                        capture_output=True,
                        text=True,
                        timeout=30,
                    )
                    # For older git without --orphan worktree support
                    if not os.path.isdir(worktree_dir):
                        subprocess.run(
                            [
                                "git",
                                "worktree",
                                "add",
                                "-b",
                                branch,
                                worktree_dir,
                            ],
                            capture_output=True,
                            text=True,
                            timeout=30,
                            check=True,
                        )
                        # Clean worktree for fresh start
                        for item in os.listdir(worktree_dir):
                            if item == ".git":
                                continue
                            path = os.path.join(worktree_dir, item)
                            if os.path.isdir(path):
                                shutil.rmtree(path)
                            else:
                                os.remove(path)

                # Run prune script if retention configured
                retention_days = gh_pages_cfg.get("retention_days", 90)
                if retention_days and keep_files:
                    self._prune_old_reports(worktree_dir, retention_days)

                # Copy report files to worktree
                for item in os.listdir(report_dir):
                    src = os.path.join(report_dir, item)
                    dst = os.path.join(worktree_dir, item)
                    if os.path.isdir(src):
                        if os.path.exists(dst):
                            shutil.rmtree(dst)
                        shutil.copytree(src, dst)
                    else:
                        shutil.copy2(src, dst)

                # Git add, commit, push in worktree
                subprocess.run(
                    ["git", "add", "-A"],
                    cwd=worktree_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )

                # Check if there are changes to commit
                diff_result = subprocess.run(
                    ["git", "diff", "--cached", "--quiet"],
                    cwd=worktree_dir,
                    capture_output=True,
                    timeout=10,
                )
                if diff_result.returncode == 0:
                    logger.info("gh-pages: no changes to deploy")
                    return True

                subprocess.run(
                    [
                        "git",
                        "commit",
                        "-m",
                        "deploy: update reports [skip ci]",
                    ],
                    cwd=worktree_dir,
                    capture_output=True,
                    text=True,
                    timeout=30,
                    check=True,
                )

                subprocess.run(
                    ["git", "push", "origin", branch],
                    cwd=worktree_dir,
                    capture_output=True,
                    text=True,
                    timeout=60,
                    check=True,
                )

                logger.info("gh-pages: deployed successfully")

            # Clean up worktree reference
            subprocess.run(
                ["git", "worktree", "prune"],
                capture_output=True,
                text=True,
                timeout=10,
            )

            return True

        except subprocess.CalledProcessError as exc:
            logger.warning(
                "gh-pages deploy git error: %s (stderr: %s)",
                exc,
                exc.stderr if exc.stderr else "",
            )
            # Clean up worktree on failure
            subprocess.run(
                ["git", "worktree", "prune"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return False
        except subprocess.TimeoutExpired:
            logger.warning("gh-pages deploy timed out")
            subprocess.run(
                ["git", "worktree", "prune"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return False
        except FileNotFoundError:
            logger.warning("git not found in PATH, skipping gh-pages deploy")
            return False

    @staticmethod
    def _prune_old_reports(deploy_dir: str, retention_days: int) -> None:
        """Remove date directories older than retention_days from deploy dir."""
        import re
        from datetime import datetime, timedelta

        cutoff = datetime.now() - timedelta(days=retention_days)
        date_pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")

        for item in os.listdir(deploy_dir):
            if not date_pattern.match(item):
                continue
            item_path = os.path.join(deploy_dir, item)
            if not os.path.isdir(item_path):
                continue
            try:
                dir_date = datetime.strptime(item, "%Y-%m-%d")
                if dir_date < cutoff:
                    shutil.rmtree(item_path)
                    logger.info("gh-pages: pruned old report dir %s", item)
            except ValueError:
                continue

    # ------------------------------------------------------------------
    # Step 5: Cleanup
    # ------------------------------------------------------------------

    def _cleanup_tmp(self, report_dir: str) -> None:
        """Clean up temporary files.

        Does NOT delete report_dir (needed for gh-pages deploy).
        Only removes tmp/debug/ if it exists.

        Args:
            report_dir: Report directory to preserve.
        """
        debug_dir = os.path.join(os.path.dirname(report_dir), "debug")
        # Fallback: if report_dir is "tmp/reports", debug is "tmp/debug"
        if not debug_dir or debug_dir == "debug":
            debug_dir = "tmp/debug"

        if os.path.isdir(debug_dir):
            shutil.rmtree(debug_dir, ignore_errors=True)
            logger.info("Removed debug directory: %s", debug_dir)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _find_topic_name(self, slug: str) -> str:
        """Find topic display name from config."""
        for topic in self._config.topics:
            if topic.slug == slug:
                return topic.name
        return slug

    def _find_topic_spec(self, slug: str) -> Any:
        """Find TopicSpec from config by slug."""
        for topic in self._config.topics:
            if topic.slug == slug:
                return topic
        return None

    def _find_report_entry(
        self, slug: str, topic_name: str, report_dir: str
    ) -> Optional[Dict[str, str]]:
        """Find the most recent report file for a topic.

        Returns a dict with topic_slug, topic_name, date, filepath
        or None if not found.
        """
        if not os.path.isdir(report_dir):
            return None

        # Look for date subdirectories containing report files
        best_date = ""
        best_filepath = ""

        for entry in sorted(os.listdir(report_dir), reverse=True):
            date_dir = os.path.join(report_dir, entry)
            if not os.path.isdir(date_dir):
                continue

            # Look for HTML file matching the slug
            for fname in os.listdir(date_dir):
                if fname.endswith(f"_paper_{slug}.html"):
                    best_date = entry
                    best_filepath = os.path.join(date_dir, fname)
                    break

            if best_filepath:
                break

        if not best_filepath:
            return None

        # Convert filesystem path to web-relative path under /reports/
        rel_path = os.path.relpath(best_filepath, report_dir)

        return {
            "topic_slug": slug,
            "topic_name": topic_name,
            "date": best_date,
            "filepath": rel_path,
        }

    def _load_latest_report_data(
        self, slug: str, report_dir: str
    ) -> Optional[Dict[str, Any]]:
        """Load the JSON report data for the latest report of a topic."""
        import json

        if not os.path.isdir(report_dir):
            return None

        for entry in sorted(os.listdir(report_dir), reverse=True):
            date_dir = os.path.join(report_dir, entry)
            if not os.path.isdir(date_dir):
                continue

            for fname in os.listdir(date_dir):
                if fname.endswith(f"_paper_{slug}.json"):
                    json_path = os.path.join(date_dir, fname)
                    try:
                        with open(json_path, "r", encoding="utf-8") as f:
                            return json.load(f)
                    except (json.JSONDecodeError, OSError):
                        logger.warning(
                            "Failed to load report JSON: %s", json_path
                        )
                        return None

        return None

    def _get_display_date(self, slug: str) -> str:
        """Get the display date from the latest completed run."""
        run_meta = self._db.get_latest_completed_run(slug)
        if run_meta is not None:
            return run_meta.display_date_kst
        return ""

    def _get_keywords_for_topic(self, slug: str) -> List[str]:
        """Extract keywords from the latest run's evaluations.

        Returns a list of keyword strings.  Falls back to empty list.
        """
        # Keywords are stored in the JSON report, not directly in DB.
        # Try to load from the latest JSON report.
        report_data = self._load_latest_report_data(slug, self._report_dir)
        if report_data is not None:
            meta = report_data.get("meta", {})
            return meta.get("keywords_used", [])
        return []

    def _find_report_files(
        self, slug: str, report_dir: str
    ) -> Dict[str, str]:
        """Find report file paths for a topic (html, md).

        Returns a dict mapping format key to absolute file path.
        """
        result: Dict[str, str] = {}

        if not os.path.isdir(report_dir):
            return result

        for entry in sorted(os.listdir(report_dir), reverse=True):
            date_dir = os.path.join(report_dir, entry)
            if not os.path.isdir(date_dir):
                continue

            for fname in os.listdir(date_dir):
                if slug not in fname:
                    continue
                abs_path = os.path.abspath(os.path.join(date_dir, fname))
                if fname.endswith(".html"):
                    result["html"] = abs_path
                elif fname.endswith(".md"):
                    result["md"] = abs_path
                elif fname.endswith(".json"):
                    result["json"] = abs_path

            if result:
                break

        return result
