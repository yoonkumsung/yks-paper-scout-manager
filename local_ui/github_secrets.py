"""GitHub Secrets registration for Setup Wizard.

Supports three strategies in priority order:
1. gh CLI (subprocess)
2. GitHub API with PyNaCl encryption
3. Manual instructions fallback
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)

# GitHub Secrets size limit: 48 KB for the secret value
_GITHUB_SECRET_MAX_BYTES = 48 * 1024

# Secrets to push to GitHub Actions
GITHUB_SECRET_KEYS = [
    "OPENROUTER_API_KEY",
    "DISCORD_WEBHOOK_URL",
    "TELEGRAM_BOT_TOKEN",
    "TELEGRAM_CHAT_ID",
    "SUPABASE_DB_URL",
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

    # Check if gh is authenticated
    try:
        auth_check = subprocess.run(
            ["gh", "auth", "status"], capture_output=True, text=True, timeout=5,
        )
        if auth_check.returncode != 0:
            return {
                "method": "gh_cli",
                "available": True,
                "success": False,
                "error": "gh CLI not authenticated. Run 'gh auth login' first.",
                "results": {},
            }
    except Exception:
        pass  # Proceed anyway; secret set will fail with a clear error

    results = {}
    for key, value in secrets.items():
        if not value:
            results[key] = "skipped (empty)"
            continue
        try:
            proc = subprocess.run(
                ["gh", "secret", "set", key, "--repo", f"{owner}/{repo}"],
                input=value, capture_output=True, text=True, timeout=60,
            )
            results[key] = "ok" if proc.returncode == 0 else proc.stderr.strip()
        except subprocess.TimeoutExpired:
            results[key] = "timeout (file may be too large)"
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


def sync_data_secrets(config_path: str = "config.yaml", data_dir: str = "data") -> dict:
    """Sync data files (config, keyword_cache, last_success) to GitHub Secrets.

    Encodes each file as base64 and pushes as a GitHub Secret so that
    GitHub Actions can restore them at workflow runtime.

    Args:
        config_path: Path to config.yaml.
        data_dir: Path to the data directory containing cache/state files.

    Returns:
        Dict with repo info, per-file results, and overall success flag.
    """
    repo_info = detect_github_repo()
    if "error" in repo_info:
        return {
            "success": False,
            "error": repo_info["error"],
            "manual_url": None,
        }

    owner, repo = repo_info["owner"], repo_info["repo"]

    # Map: secret name -> local file path
    file_map = {
        "PAPER_SCOUT_CONFIG": Path(config_path),
        "PAPER_SCOUT_KEYWORD_CACHE": Path(data_dir) / "keyword_cache.json",
        "PAPER_SCOUT_LAST_SUCCESS": Path(data_dir) / "last_success.json",
        "PAPER_SCOUT_MODEL_CAPS": Path(data_dir) / "model_caps.json",
    }

    # Load previous deploy hashes to detect actual changes
    hash_file = Path(data_dir) / ".deploy_hashes.json"
    prev_hashes: dict[str, str] = {}
    if hash_file.exists():
        try:
            prev_hashes = json.loads(hash_file.read_text(encoding="utf-8"))
        except Exception:
            pass

    # Build secrets dict: base64-encode each file that exists
    secrets: dict[str, str] = {}
    file_status: dict[str, str] = {}
    current_hashes: dict[str, str] = {}
    warnings: list[str] = []

    for secret_name, file_path in file_map.items():
        if not file_path.exists():
            file_status[secret_name] = "file not found (skipped)"
            warnings.append(f"{file_path.name}: file not found, skipped")
            continue

        try:
            raw = file_path.read_bytes()
        except PermissionError:
            file_status[secret_name] = "permission denied"
            warnings.append(f"{file_path.name}: permission denied")
            continue
        except Exception as e:
            file_status[secret_name] = f"read error: {e}"
            continue

        raw_size = len(raw)

        if raw_size == 0:
            file_status[secret_name] = "empty file (skipped)"
            warnings.append(f"{file_path.name}: empty file, skipped")
            continue

        encoded = base64.b64encode(raw).decode("ascii")
        encoded_size = len(encoded)

        if encoded_size > _GITHUB_SECRET_MAX_BYTES:
            raw_kb = raw_size / 1024
            limit_kb = _GITHUB_SECRET_MAX_BYTES / 1024
            file_status[secret_name] = (
                f"too large ({raw_kb:.1f}KB raw, limit {limit_kb:.0f}KB encoded)"
            )
            warnings.append(
                f"{file_path.name}: {raw_kb:.1f}KB exceeds GitHub Secret limit"
            )
            continue

        # Compare hash to detect actual changes
        file_hash = hashlib.sha256(raw).hexdigest()
        current_hashes[secret_name] = file_hash
        changed = file_hash != prev_hashes.get(secret_name)

        secrets[secret_name] = encoded
        file_status[secret_name] = (
            f"updated ({raw_size:,} bytes)" if changed
            else f"unchanged ({raw_size:,} bytes)"
        )

    if not secrets:
        return {
            "success": False,
            "error": "No valid data files to sync",
            "file_status": file_status,
            "warnings": warnings,
            "repo": repo_info,
        }

    def _save_hashes() -> None:
        """Save current file hashes after successful deploy."""
        try:
            hash_file.write_text(
                json.dumps(current_hashes, indent=2), encoding="utf-8"
            )
        except Exception:
            pass

    # Try gh CLI first
    result = _try_gh_cli(owner, repo, secrets)
    if result.get("available") and result.get("success"):
        _save_hashes()
        return {
            "success": True,
            "method": "gh_cli",
            "results": result["results"],
            "file_status": file_status,
            "warnings": warnings or None,
            "repo": repo_info,
        }

    # Try GitHub API with token
    token = os.getenv("GITHUB_TOKEN")
    if token:
        api_result = _try_api(owner, repo, secrets, token)
        if api_result.get("available") and api_result.get("success"):
            _save_hashes()
            return {
                "success": True,
                "method": "api",
                "results": api_result["results"],
                "file_status": file_status,
                "warnings": warnings or None,
                "repo": repo_info,
            }
        if api_result.get("available") and "results" in api_result:
            return {
                "success": False,
                "method": "api",
                "results": api_result["results"],
                "error": api_result.get("error"),
                "file_status": file_status,
                "warnings": warnings or None,
                "repo": repo_info,
                "manual_url": f"https://github.com/{owner}/{repo}/settings/secrets/actions",
            }

    # Manual fallback
    gh_available = result.get("available", False)
    manual_url = f"https://github.com/{owner}/{repo}/settings/secrets/actions"
    return {
        "success": False,
        "method": "manual",
        "message": "Automatic methods unavailable. Please set secrets manually.",
        "manual_url": manual_url,
        "file_status": file_status,
        "warnings": warnings or None,
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
