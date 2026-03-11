# app/orchestration/planner.py
import json
from typing import Any

from app.llm.client import chat
from app.orchestration.registry import DOC_REGISTRY

ALLOWED_ROUTES = {"single", "multi", "unknown"}


def _registry_view() -> dict[str, Any]:
    """Expose only safe registry info to the planner (no document text)."""
    view = {}
    for key, info in DOC_REGISTRY.items():
        view[key] = {
            "document_id": info["document_id"],
            "aliases": info.get("aliases", []),
            "description": info.get("description", ""),
        }
    return view


def build_planner_messages(question: str) -> list[dict]:
    system = (
    "You are a planning agent.\n"
    "\n"
    "Your task is to analyze the user's question and decide:\n"
    "- whether it concerns a single document,\n"
    "- multiple documents,\n"
    "- or cannot be answered safely.\n"
    "\n"
    "You MUST follow these rules strictly:\n"
    "\n"
    "1. Do NOT answer the question.\n"
    "2. Do NOT use external knowledge.\n"
    "3. Use ONLY the document registry provided.\n"
    "4. Output ONLY valid JSON. No text outside JSON.\n"
    "5. Use ONLY registry KEYS as values in \"targets\".\n"
    "6. If the question explicitly mentions TWO or more different systems,\n"
    "   you MUST:\n"
    "   - set route = \"multi\"\n"
    "   - include ALL relevant registry keys in \"targets\"\n"
    "7. If a concept could apply to multiple systems AND the question does NOT\n"
    "   explicitly name one, you MUST set route = \"unknown\".\n"
    "8. Do NOT guess based on what documents exist in the registry.\n"
    "9. If unsure, choose route = \"unknown\".\n"
    "\n"
    "The output MUST match this schema exactly:\n"
    "{\"route\":\"single|multi|unknown\",\"targets\":[],\"reason\":\"\"}\n"
    "\n"
    "You MUST follow the examples below exactly.\n"
    "\n"
    "Example 1:\n"
    "Question: Why don’t readers block writers in PostgreSQL?\n"
    "Output:\n"
    "{\"route\":\"single\",\"targets\":[\"postgres\"],\"reason\":\"The question explicitly refers to PostgreSQL\"}\n"
    "\n"
    "Example 2:\n"
    "Question: Compare MVCC in PostgreSQL versus snapshot isolation in SQL Server.\n"
    "Output:\n"
    "{\"route\":\"multi\",\"targets\":[\"postgres\",\"sqlserver\"],\"reason\":\"The question explicitly compares two database systems\"}\n"
    "\n"
    "Example 3:\n"
    "Question: How does MVCC work?\n"
    "Output:\n"
    "{\"route\":\"unknown\",\"targets\":[],\"reason\":\"MVCC is a general concept and no specific system was mentioned\"}\n"
)


    user = {
        "registry": _registry_view(),
        "question": question,
    }

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user)},
    ]


def parse_and_validate_plan(raw: str) -> dict:
    """
    Strictly parse and validate.
    If invalid -> return unknown plan.
    """
    try:
        data = json.loads(raw)
    except Exception:
        return {"route": "unknown", "targets": [], "reason": "Invalid JSON from planner"}

    route = data.get("route")
    targets = data.get("targets", [])
    reason = data.get("reason", "")

    if route not in ALLOWED_ROUTES:
        return {"route": "unknown", "targets": [], "reason": "Invalid route"}

    if not isinstance(targets, list) or not all(isinstance(t, str) for t in targets):
        return {"route": "unknown", "targets": [], "reason": "Invalid targets"}

    # Only allow known registry keys
    targets = [t for t in targets if t in DOC_REGISTRY]

    # Enforce shape rules
    if route == "single" and len(targets) > 1:
        targets = targets[:1]
    if route == "multi" and len(targets) < 2:
        return {"route": "unknown", "targets": [], "reason": "Multi requires 2 targets"}

    # Optional: cap multi to 2 for now
    if route == "multi":
        targets = targets[:2]

    if not isinstance(reason, str):
        reason = ""

    return {"route": route, "targets": targets, "reason": reason}


def plan_question(question: str) -> dict:
    messages = build_planner_messages(question)
    raw = chat(messages)  # must return text
    return parse_and_validate_plan(raw)
