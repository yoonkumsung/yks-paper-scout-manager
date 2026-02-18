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
    # Hardware / Device
    "hardware": ["cs.AR", "cs.RO"],
    "device": ["cs.AR", "cs.RO", "eess.SP"],
    # Video / Image processing
    "image processing": ["cs.CV", "eess.IV"],
    "video processing": ["cs.CV", "eess.IV", "cs.MM"],
    "image enhancement": ["cs.CV", "eess.IV"],
    "super resolution": ["cs.CV", "eess.IV"],
    "short form": ["cs.MM", "cs.CV"],
    "content creation": ["cs.MM", "cs.CV"],
    # Generative models
    "generative": ["cs.CV", "cs.LG", "cs.AI"],
    "gan": ["cs.CV", "cs.LG"],
    "diffusion": ["cs.CV", "cs.LG"],
    # Pose / Activity
    "pose": ["cs.CV"],
    # Streaming / Network
    "streaming": ["cs.MM", "cs.NI"],
    # Control / Systems
    "control": ["cs.SY", "eess.SY", "math.OC"],
    "signal": ["eess.SP"],
    # Audio / Sound
    "sound": ["cs.SD", "eess.AS"],
    # Social / Sharing
    "sharing": ["cs.SI", "cs.MM"],
    "advertising": ["cs.IR", "cs.AI"],
    # Strategy / Game theory
    "strategy": ["cs.AI", "cs.GT"],
    "analysis": ["cs.CV", "cs.AI"],
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
    "IMPORTANT: Think about ALL aspects of the description - hardware, software, "
    "algorithms, applications, social aspects, signal processing, multimedia, "
    "networking, user interaction, business models, and more. "
    "For each functional area mentioned in the description, consider which arXiv "
    "categories publish papers on that topic.\n\n"
    "Available arXiv categories (with brief descriptions):\n"
    "CS categories:\n"
    "  cs.AI (Artificial Intelligence), cs.LG (Machine Learning), "
    "cs.CL (NLP/Language Processing), cs.CV (Computer Vision), "
    "cs.NE (Neural/Evolutionary Computing), cs.IR (Information Retrieval), "
    "cs.MA (Multi-Agent Systems), cs.RO (Robotics), "
    "cs.SE (Software Engineering), cs.DC (Distributed Computing), "
    "cs.DB (Databases), cs.OS (Operating Systems), "
    "cs.PL (Programming Languages), cs.AR (Hardware Architecture), "
    "cs.NI (Networking/Internet), cs.PF (Performance), "
    "cs.DS (Data Structures/Algorithms), cs.CC (Computational Complexity), "
    "cs.LO (Logic), cs.FL (Formal Languages), "
    "cs.DM (Discrete Mathematics), cs.IT (Information Theory), "
    "cs.NA (Numerical Analysis), cs.SC (Symbolic Computation), "
    "cs.GT (Game Theory/Economics), cs.CG (Computational Geometry), "
    "cs.CR (Cryptography/Security), cs.HC (Human-Computer Interaction), "
    "cs.CY (Computers and Society), cs.CE (Computational Engineering), "
    "cs.GR (Graphics), cs.MM (Multimedia), "
    "cs.SD (Sound), cs.SI (Social/Information Networks), "
    "cs.DL (Digital Libraries), cs.ET (Emerging Technologies), "
    "cs.MS (Mathematical Software), cs.GL (General Literature), "
    "cs.OH (Other), cs.SY (Systems and Control)\n"
    "Stats: stat.ML (Machine Learning), stat.ME (Methodology), "
    "stat.TH (Theory), stat.AP (Applications), stat.CO (Computation)\n"
    "EESS: eess.SP (Signal Processing), eess.AS (Audio/Speech), "
    "eess.IV (Image/Video Processing), eess.SY (Systems and Control)\n"
    "Math: math.OC (Optimization/Control), math.ST (Statistics Theory), "
    "math.PR (Probability)\n"
    "Q-Bio: q-bio.QM (Quantitative Methods), q-bio.NC (Neurons and Cognition)\n"
    "Q-Fin: q-fin.ST (Statistical Finance), q-fin.CP (Computational Finance)\n"
    "Physics: physics.data-an (Data Analysis)\n\n"
    "Respond with a JSON object containing:\n"
    '  "categories": ["cs.AI", "cs.CV", ...],\n'
    '  "reasoning": "Brief explanation of why these categories are relevant"\n\n'
    "Select 10-20 most relevant categories. Prioritize recall over precision - "
    "it is better to include a potentially relevant category than to miss one."
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
                "max_tokens": 8192,
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
