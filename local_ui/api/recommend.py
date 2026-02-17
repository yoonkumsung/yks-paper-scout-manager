"""Category recommendation API for Paper Scout local UI.

Provides AI-based or rule-based arXiv category recommendations
based on a topic description.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from flask import Blueprint, current_app, jsonify, request

logger = logging.getLogger(__name__)

recommend_bp = Blueprint("recommend", __name__)

# Rule-based keyword-to-category mapping for fallback
_KEYWORD_CATEGORY_MAP: dict[str, list[str]] = {
    # AI / ML core
    "ai": ["cs.AI"],
    "artificial intelligence": ["cs.AI"],
    "machine learning": ["cs.LG", "stat.ML"],
    "deep learning": ["cs.LG"],
    "neural network": ["cs.LG", "cs.NE"],
    "reinforcement learning": ["cs.LG", "cs.AI"],
    "transformer": ["cs.LG", "cs.CL"],
    "llm": ["cs.CL", "cs.AI", "cs.LG"],
    "language model": ["cs.CL", "cs.AI"],
    "gpt": ["cs.CL", "cs.AI"],
    "nlp": ["cs.CL"],
    "natural language": ["cs.CL"],
    "text": ["cs.CL", "cs.IR"],
    "chatbot": ["cs.CL", "cs.AI"],
    "rag": ["cs.CL", "cs.IR"],
    "prompt": ["cs.CL", "cs.AI"],
    "fine-tuning": ["cs.CL", "cs.LG"],
    "rlhf": ["cs.CL", "cs.LG"],
    # Vision
    "computer vision": ["cs.CV"],
    "image": ["cs.CV", "eess.IV"],
    "video": ["cs.CV", "cs.MM", "eess.IV"],
    "object detection": ["cs.CV"],
    "segmentation": ["cs.CV"],
    "3d": ["cs.CV", "cs.GR"],
    "camera": ["cs.CV", "eess.IV"],
    "photograph": ["cs.CV", "eess.IV"],
    # Robotics
    "robot": ["cs.RO", "cs.AI"],
    "autonomous": ["cs.RO", "cs.AI"],
    "manipulation": ["cs.RO"],
    "navigation": ["cs.RO"],
    "drone": ["cs.RO"],
    "embodied": ["cs.RO", "cs.AI"],
    "physical ai": ["cs.RO", "cs.AI"],
    # Multi-agent
    "multi-agent": ["cs.MA", "cs.AI"],
    "agent": ["cs.MA", "cs.AI"],
    "swarm": ["cs.MA"],
    # Information retrieval
    "search": ["cs.IR"],
    "recommendation": ["cs.IR"],
    "retrieval": ["cs.IR"],
    "ranking": ["cs.IR"],
    # Software / Systems
    "software": ["cs.SE"],
    "code": ["cs.SE", "cs.PL"],
    "distributed": ["cs.DC"],
    "cloud": ["cs.DC"],
    "database": ["cs.DB"],
    "network": ["cs.NI"],
    "security": ["cs.CR"],
    "privacy": ["cs.CR"],
    # Sports / Activity
    "sports": ["cs.CV", "cs.AI"],
    "action recognition": ["cs.CV"],
    "pose estimation": ["cs.CV"],
    "motion": ["cs.CV", "cs.RO"],
    "tracking": ["cs.CV"],
    "highlight": ["cs.CV", "cs.MM"],
    # Platform / Social
    "platform": ["cs.SI", "cs.CY"],
    "social": ["cs.SI"],
    "sns": ["cs.SI"],
    "advertisement": ["cs.IR", "cs.SI"],
    "marketplace": ["cs.SI", "cs.CE"],
    # Signal / Audio
    "signal processing": ["eess.SP"],
    "audio": ["eess.AS", "cs.SD"],
    "speech": ["eess.AS", "cs.CL"],
    # Math / Stats
    "optimization": ["math.OC"],
    "statistics": ["stat.ML", "stat.ME"],
    "probability": ["math.PR"],
    # Graphics / Multimedia
    "graphics": ["cs.GR"],
    "rendering": ["cs.GR"],
    "multimedia": ["cs.MM"],
    "editing": ["cs.CV", "cs.MM"],
    # HCI
    "interface": ["cs.HC"],
    "interaction": ["cs.HC"],
    "ux": ["cs.HC"],
    "user experience": ["cs.HC"],
}

# LLM system prompt for category recommendation
_RECOMMEND_SYSTEM_PROMPT = (
    "You are an expert research librarian specializing in arXiv paper categories. "
    "Given a project description, recommend the most relevant arXiv categories.\n\n"
    "Available arXiv categories:\n"
    "CS: cs.AI, cs.LG, cs.CL, cs.CV, cs.NE, cs.IR, cs.MA, cs.RO, cs.SE, cs.DC, "
    "cs.DB, cs.OS, cs.PL, cs.AR, cs.NI, cs.PF, cs.DS, cs.CC, cs.LO, cs.FL, "
    "cs.DM, cs.IT, cs.NA, cs.SC, cs.GT, cs.CG, cs.CR, cs.HC, cs.CY, cs.CE, "
    "cs.GR, cs.MM, cs.SD, cs.SI, cs.DL, cs.ET, cs.MS, cs.GL, cs.OH, cs.SY\n"
    "Stats: stat.ML, stat.ME, stat.TH, stat.AP, stat.CO\n"
    "EESS: eess.SP, eess.AS, eess.IV, eess.SY\n"
    "Math: math.OC, math.ST, math.PR\n"
    "Q-Bio: q-bio.QM, q-bio.NC\n"
    "Q-Fin: q-fin.ST, q-fin.CP\n"
    "Physics: physics.data-an\n\n"
    "Respond with a JSON object containing:\n"
    '  "categories": ["cs.AI", "cs.CV", ...],\n'
    '  "reasoning": "Brief explanation of why these categories are relevant"\n\n'
    "Select 5-10 most relevant categories. Prioritize precision over recall."
)


@recommend_bp.route("/categories", methods=["POST"])
def recommend_categories() -> tuple[dict, int]:
    """Recommend arXiv categories based on topic description.

    Request body:
        description (str): Topic description text

    Returns:
        JSON with recommended categories and method used
    """
    data = request.get_json(silent=True) or {}
    description = (data.get("description") or "").strip()

    if not description:
        return jsonify({"error": "description is required"}), 400

    api_key = os.environ.get("OPENROUTER_API_KEY", "").strip()

    if api_key:
        result = _recommend_with_llm(description, api_key)
    else:
        result = _recommend_with_rules(description)

    return jsonify(result), 200


def _recommend_with_llm(description: str, api_key: str) -> dict[str, Any]:
    """Use LLM to recommend categories."""
    from local_ui.config_io import read_config

    try:
        config_path = current_app.config.get("CONFIG_PATH", "config.yaml")
        config = read_config(config_path)

        llm_config = config.get("llm", {})
        base_url = llm_config.get("base_url", "https://openrouter.ai/api/v1")
        model = llm_config.get("model", "z-ai/glm-4.5-air:free")

        import requests as req_lib

        resp = req_lib.post(
            f"{base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": _RECOMMEND_SYSTEM_PROMPT},
                    {"role": "user", "content": f"Project description:\n{description}"},
                ],
                "max_tokens": 2048,
                "temperature": 0.3,
            },
            timeout=30,
        )
        resp.raise_for_status()
        resp_data = resp.json()

        content = resp_data.get("choices", [{}])[0].get("message", {}).get("content", "")

        # Parse JSON from response
        parsed = _parse_json_from_text(content)
        if parsed and isinstance(parsed.get("categories"), list):
            return {
                "categories": parsed["categories"],
                "reasoning": parsed.get("reasoning", ""),
                "method": "llm",
            }

        # LLM parse failed, fallback to rules
        logger.warning("recommend: LLM response parse failed, falling back to rules")
        result = _recommend_with_rules(description)
        result["method"] = "rules_fallback"
        return result

    except Exception as e:
        logger.warning("recommend: LLM call failed (%s), falling back to rules", e)
        result = _recommend_with_rules(description)
        result["method"] = "rules_fallback"
        return result


def _recommend_with_rules(description: str) -> dict[str, Any]:
    """Rule-based category recommendation using keyword matching."""
    desc_lower = description.lower()
    category_scores: dict[str, int] = {}

    for keyword, cats in _KEYWORD_CATEGORY_MAP.items():
        if keyword in desc_lower:
            for cat in cats:
                category_scores[cat] = category_scores.get(cat, 0) + 1

    if not category_scores:
        # Default to common AI/ML categories
        return {
            "categories": ["cs.AI", "cs.LG"],
            "reasoning": "No specific keywords matched; defaulting to general AI/ML categories.",
            "method": "rules_default",
        }

    # Sort by score descending, take top 10
    sorted_cats = sorted(category_scores.items(), key=lambda x: x[1], reverse=True)
    recommended = [cat for cat, _ in sorted_cats[:10]]

    return {
        "categories": recommended,
        "reasoning": f"Matched {len(category_scores)} categories based on keyword analysis.",
        "method": "rules",
    }


def _parse_json_from_text(text: str) -> dict | None:
    """Extract and parse JSON object from text that may contain markdown."""
    # Try direct parse
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        pass

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except (json.JSONDecodeError, TypeError):
            pass

    # Try finding JSON object pattern
    match = re.search(r"\{[^{}]*\"categories\"[^{}]*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except (json.JSONDecodeError, TypeError):
            pass

    return None
