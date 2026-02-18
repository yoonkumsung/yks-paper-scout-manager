"""GitHub Secrets registration for Setup Wizard.

Supports three strategies in priority order:
1. gh CLI (subprocess)
2. GitHub API with PyNaCl encryption
3. Manual instructions fallback
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# Secrets to push to GitHub Actions
GITHUB_SECRET_KEYS = [
    "OPENROUTER_API_KEY",
    "DISCORD_WEBHOOK_URL",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
]


def detect_github_repo() -> dict:
    """Detect GitHub owner/repo from git remote.

    Returns:
        Dict with owner, repo, full_name, and url fields, or error.
    """
    try:
        result = subprocess.run(
            ["git", "remote", "get-url", "origin"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return {"error": "No git remote 'origin' found"}

        url = result.stdout.strip()
        # Match SSH or HTTPS patterns
        match = re.search(r"github\.com[:/](.+?)/(.+?)(?:\.git)?$", url)
        if not match:
            return {"error": f"Not a GitHub URL: {url}"}

        owner, repo = match.group(1), match.group(2)
        return {
            "owner": owner,
            "repo": repo,
            "full_name": f"{owner}/{repo}",
            "url": f"https://github.com/{owner}/{repo}",
        }
    except FileNotFoundError:
        return {"error": "git command not found"}
    except Exception as e:
        return {"error": str(e)}


def _try_gh_cli(owner: str, repo: str, secrets: dict[str, str]) -> dict:
    """Try setting secrets via gh CLI.

    Returns:
        Dict with method, results per key, and success flag.
    """
    try:
        # Check if gh is available
        check = subprocess.run(
            ["gh", "--version"], capture_output=True, text=True, timeout=5,
        )
        if check.returncode != 0:
            return {"method": "gh_cli", "available": False}
    except FileNotFoundError:
        return {"method": "gh_cli", "available": False}

    results = {}
    for key, value in secrets.items():
        if not value:
            results[key] = "skipped (empty)"
            continue
        try:
            proc = subprocess.run(
                ["gh", "secret", "set", key, "--repo", f"{owner}/{repo}"],
                input=value, capture_output=True, text=True, timeout=30,
            )
            results[key] = "ok" if proc.returncode == 0 else proc.stderr.strip()
        except Exception as e:
            results[key] = str(e)

    all_ok = all(v == "ok" or v.startswith("skipped") for v in results.values())
    return {"method": "gh_cli", "available": True, "results": results, "success": all_ok}


def _try_api(owner: str, repo: str, secrets: dict[str, str], token: str) -> dict:
    """Try setting secrets via GitHub API with PyNaCl encryption.

    Returns:
        Dict with method, results per key, and success flag.
    """
    try:
        import requests
        from nacl import encoding, public
    except ImportError as e:
        return {"method": "api", "available": False, "error": str(e)}

    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }

    # Get repo public key
    try:
        pk_resp = requests.get(
            f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/public-key",
            headers=headers, timeout=10,
        )
        pk_resp.raise_for_status()
        pk_data = pk_resp.json()
        public_key = public.PublicKey(
            pk_data["key"].encode("utf-8"), encoding.Base64Encoder,
        )
        key_id = pk_data["key_id"]
    except Exception as e:
        return {"method": "api", "available": True, "error": f"Failed to get public key: {e}"}

    results = {}
    for key, value in secrets.items():
        if not value:
            results[key] = "skipped (empty)"
            continue
        try:
            sealed_box = public.SealedBox(public_key)
            encrypted = sealed_box.encrypt(value.encode("utf-8"))
            encrypted_b64 = encoding.Base64Encoder.encode(encrypted).decode("utf-8")

            resp = requests.put(
                f"https://api.github.com/repos/{owner}/{repo}/actions/secrets/{key}",
                headers=headers, timeout=10,
                json={"encrypted_value": encrypted_b64, "key_id": key_id},
            )
            results[key] = "ok" if resp.status_code in (201, 204) else f"HTTP {resp.status_code}: {resp.text}"
        except Exception as e:
            results[key] = str(e)

    all_ok = all(v == "ok" or v.startswith("skipped") for v in results.values())
    return {"method": "api", "available": True, "results": results, "success": all_ok}


def push_secrets(secrets: dict[str, str]) -> dict:
    """Push secrets to GitHub Actions using best available method.

    Args:
        secrets: Dict of secret name to value.

    Returns:
        Dict with repo info, method used, and per-key results.
    """
    repo_info = detect_github_repo()
    if "error" in repo_info:
        return {
            "success": False,
            "error": repo_info["error"],
            "manual_url": None,
        }

    owner, repo = repo_info["owner"], repo_info["repo"]

    # Filter to only GitHub-relevant secrets
    filtered = {k: v for k, v in secrets.items() if k in GITHUB_SECRET_KEYS and v}

    if not filtered:
        return {
            "success": True,
            "method": "none",
            "message": "No secrets to push",
            "repo": repo_info,
        }

    # Strategy 1: gh CLI
    result = _try_gh_cli(owner, repo, filtered)
    if result.get("available") and result.get("success"):
        return {
            "success": True,
            "method": "gh_cli",
            "results": result["results"],
            "repo": repo_info,
        }

    # Strategy 2: GitHub API + PyNaCl
    token = secrets.get("GITHUB_TOKEN") or os.getenv("GITHUB_TOKEN")
    if token:
        result = _try_api(owner, repo, filtered, token)
        if result.get("available") and result.get("success"):
            return {
                "success": True,
                "method": "api",
                "results": result["results"],
                "repo": repo_info,
            }
        if result.get("available") and "results" in result:
            return {
                "success": False,
                "method": "api",
                "results": result["results"],
                "error": result.get("error"),
                "repo": repo_info,
                "manual_url": f"https://github.com/{owner}/{repo}/settings/secrets/actions",
            }

    # Strategy 3: Manual fallback
    manual_url = f"https://github.com/{owner}/{repo}/settings/secrets/actions"
    gh_available = result.get("available", False) if result.get("method") == "gh_cli" else False

    return {
        "success": False,
        "method": "manual",
        "message": "Automatic methods unavailable. Please set secrets manually.",
        "manual_url": manual_url,
        "repo": repo_info,
        "suggestions": _get_suggestions(gh_available, token),
    }


def _get_suggestions(gh_available: bool, token: str | None) -> list[str]:
    """Generate troubleshooting suggestions."""
    suggestions = []
    if not gh_available:
        suggestions.append("Install GitHub CLI: https://cli.github.com/ then run 'gh auth login'")
    if not token:
        suggestions.append("Set GITHUB_TOKEN in .env for API-based secret registration")
    return suggestions
