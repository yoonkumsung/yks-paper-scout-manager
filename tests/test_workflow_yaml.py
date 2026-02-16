"""Tests for GitHub Actions workflow YAML structure validation.

Validates that .github/workflows/paper-scout.yml has correct structure,
triggers, inputs, steps, and configuration.
"""

from __future__ import annotations

import yaml
from pathlib import Path


def test_workflow_yaml_is_valid():
    """Verify YAML can be parsed without errors."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    assert workflow_path.exists(), "Workflow file does not exist"

    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    assert workflow is not None, "Workflow YAML is empty"
    assert isinstance(workflow, dict), "Workflow YAML is not a dictionary"


def test_workflow_has_correct_triggers():
    """Verify workflow has schedule and workflow_dispatch triggers."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    # PyYAML may interpret "on:" as boolean True
    triggers = workflow.get("on") or workflow.get(True)
    assert triggers is not None, "Missing 'on' section"

    # Schedule trigger
    assert "schedule" in triggers, "Missing schedule trigger"
    assert isinstance(triggers["schedule"], list), "Schedule must be a list"
    assert len(triggers["schedule"]) > 0, "Schedule list is empty"

    # workflow_dispatch trigger
    assert "workflow_dispatch" in triggers, "Missing workflow_dispatch trigger"


def test_workflow_schedule_cron():
    """Verify cron schedule is '0 2 * * *' (UTC 02:00)."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    # PyYAML may interpret "on:" as boolean True
    triggers = workflow.get("on") or workflow.get(True)
    schedule = triggers["schedule"]
    assert len(schedule) == 1, "Expected exactly one schedule entry"
    assert schedule[0]["cron"] == "0 2 * * *", "Incorrect cron schedule"


def test_workflow_dispatch_inputs():
    """Verify workflow_dispatch has all 4 required inputs."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    # PyYAML may interpret "on:" as boolean True
    triggers = workflow.get("on") or workflow.get(True)
    inputs = triggers["workflow_dispatch"]["inputs"]
    assert "date_from" in inputs, "Missing date_from input"
    assert "date_to" in inputs, "Missing date_to input"
    assert "mode" in inputs, "Missing mode input"
    assert "dedup" in inputs, "Missing dedup input"

    # Validate input types
    assert inputs["date_from"]["type"] == "string", "date_from must be string type"
    assert inputs["date_to"]["type"] == "string", "date_to must be string type"
    assert inputs["mode"]["type"] == "string", "mode must be string type"
    assert inputs["dedup"]["type"] == "string", "dedup must be string type"

    # Validate defaults
    assert inputs["mode"]["default"] == "full", "mode default should be 'full'"
    assert inputs["dedup"]["default"] == "skip_recent", "dedup default should be 'skip_recent'"


def test_workflow_concurrency_group():
    """Verify concurrency configuration is present."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    assert "concurrency" in workflow, "Missing concurrency section"
    concurrency = workflow["concurrency"]

    assert "group" in concurrency, "Missing concurrency group"
    assert "paper-scout" in concurrency["group"], "Concurrency group should include 'paper-scout'"
    assert concurrency["cancel-in-progress"] is False, "cancel-in-progress should be false"


def test_workflow_permissions():
    """Verify workflow has correct permissions."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    assert "permissions" in workflow, "Missing permissions section"
    permissions = workflow["permissions"]

    assert permissions["contents"] == "write", "contents permission should be write"
    assert permissions["issues"] == "write", "issues permission should be write"
    assert permissions["pages"] == "write", "pages permission should be write"


def test_workflow_has_required_steps():
    """Verify workflow has all required steps."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    jobs = workflow["jobs"]
    assert "paper-scout" in jobs, "Missing paper-scout job"

    steps = jobs["paper-scout"]["steps"]
    step_names = [step.get("name", "") for step in steps]

    # Required step names
    required_steps = [
        "Checkout repository",
        "Restore DB cache",
        "Setup Python 3.11",
        "Install core dependencies",
        "Install embed dependencies",
        "Install viz dependencies",
        "Run Paper Scout",
        "Save DB cache",
        "Commit metadata",
        "Deploy gh-pages",
    ]

    for required_step in required_steps:
        assert any(required_step in name for name in step_names), f"Missing step: {required_step}"

    # Verify at least 10 steps total
    assert len(steps) >= 10, f"Expected at least 10 steps, found {len(steps)}"


def test_workflow_checkout_step():
    """Verify checkout step uses correct action."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["paper-scout"]["steps"]
    checkout_step = next((s for s in steps if "Checkout" in s.get("name", "")), None)

    assert checkout_step is not None, "Checkout step not found"
    assert checkout_step["uses"] == "actions/checkout@v4", "Should use actions/checkout@v4"


def test_workflow_python_setup():
    """Verify Python setup step is configured correctly."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["paper-scout"]["steps"]
    python_step = next((s for s in steps if "Setup Python" in s.get("name", "")), None)

    assert python_step is not None, "Python setup step not found"
    assert python_step["uses"] == "actions/setup-python@v5", "Should use actions/setup-python@v5"
    assert python_step["with"]["python-version"] == "3.11", "Python version should be 3.11"


def test_workflow_run_main_py():
    """Verify main.py execution step is present with correct arguments."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["paper-scout"]["steps"]
    run_step = next((s for s in steps if "Run Paper Scout" in s.get("name", "")), None)

    assert run_step is not None, "Run Paper Scout step not found"
    assert "run" in run_step, "Run step must have 'run' field"

    run_command = run_step["run"]
    assert "python main.py" in run_command, "Should execute main.py"
    assert "--mode" in run_command, "Should include --mode argument"
    assert "--dedup" in run_command, "Should include --dedup argument"


def test_workflow_optional_deps_continue_on_error():
    """Verify optional dependency installation steps have continue-on-error."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["paper-scout"]["steps"]

    # Embed dependencies should have continue-on-error
    embed_step = next((s for s in steps if "embed dependencies" in s.get("name", "")), None)
    assert embed_step is not None, "Embed dependencies step not found"
    assert embed_step.get("continue-on-error") is True, "Embed step should continue-on-error"

    # Viz dependencies should have continue-on-error
    viz_step = next((s for s in steps if "viz dependencies" in s.get("name", "")), None)
    assert viz_step is not None, "Viz dependencies step not found"
    assert viz_step.get("continue-on-error") is True, "Viz step should continue-on-error"


def test_workflow_db_cache_save_always():
    """Verify DB cache save runs with 'always()' condition."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["paper-scout"]["steps"]
    cache_save_step = next((s for s in steps if "Save DB cache" in s.get("name", "")), None)

    assert cache_save_step is not None, "Save DB cache step not found"
    assert cache_save_step.get("if") == "always()", "Cache save should run always()"


def test_workflow_commit_metadata_step():
    """Verify commit metadata step is configured correctly."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["paper-scout"]["steps"]
    commit_step = next((s for s in steps if "Commit metadata" in s.get("name", "")), None)

    assert commit_step is not None, "Commit metadata step not found"
    assert "run" in commit_step, "Commit step must have 'run' field"

    run_command = commit_step["run"]
    assert "git config user.name" in run_command, "Should configure git user name"
    assert "paper-scout[bot]" in run_command, "Should use paper-scout[bot] as user"
    assert "git add data/" in run_command, "Should add data/ files"
    assert "git commit" in run_command, "Should commit changes"
    assert "git push" in run_command, "Should push changes"


def test_workflow_deploy_ghpages_step():
    """Verify gh-pages deployment step uses correct action."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["paper-scout"]["steps"]
    deploy_step = next((s for s in steps if "Deploy gh-pages" in s.get("name", "")), None)

    assert deploy_step is not None, "Deploy gh-pages step not found"
    assert deploy_step["uses"] == "peaceiris/actions-gh-pages@v3", "Should use peaceiris/actions-gh-pages@v3"

    deploy_with = deploy_step["with"]
    assert deploy_with["publish_dir"] == "./tmp/reports", "Should publish ./tmp/reports"
    assert deploy_with["keep_files"] is True, "Should keep existing files"


def test_workflow_debug_artifacts_on_failure():
    """Verify debug artifacts upload only runs on failure."""
    workflow_path = Path(".github/workflows/paper-scout.yml")
    with workflow_path.open(encoding="utf-8") as f:
        workflow = yaml.safe_load(f)

    steps = workflow["jobs"]["paper-scout"]["steps"]
    debug_step = next((s for s in steps if "Upload debug" in s.get("name", "")), None)

    assert debug_step is not None, "Upload debug artifacts step not found"
    assert debug_step.get("if") == "failure()", "Debug upload should only run on failure()"
    assert debug_step["uses"].startswith("actions/upload-artifact@"), "Should use upload-artifact action"
