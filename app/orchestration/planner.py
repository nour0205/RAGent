# app/orchestration/planner.py
import json
from typing import Any

from app.llm.client import chat
from app.orchestration.registry import DOC_REGISTRY

ALLOWED_ROUTES = {
    "concept_explanation",
    "source_recall",
    "exam_preparation",
    "unknown",
}


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
    "You are a study planning agent.\n"
    "\n"
    "Your task is to analyze the user's question and classify its intent.\n"
    "\n"
    "You MUST choose one of the following routes:\n"
    "- concept_explanation → the user wants to understand a concept\n"
    "- source_recall → the user wants to find where something was explained\n"
    "- exam_preparation → the user wants to know what to revise or what matters most\n"
    "- unknown → if the question is unclear or cannot be safely classified\n"
    "\n"
    "You MUST follow these rules strictly:\n"
    "\n"
    "1. Do NOT answer the question.\n"
    "2. Do NOT use external knowledge.\n"
    "3. Use ONLY the document registry provided.\n"
    "4. Output ONLY valid JSON. No text outside JSON.\n"
    "5. Use ONLY registry KEYS as values in \"targets\".\n"
    "6. Select targets only if the question clearly refers to specific documents.\n"
    "7. If unsure about the intent, choose \"unknown\".\n"
    "\n"
    "The output MUST match this schema exactly:\n"
    "{\"route\":\"concept_explanation|source_recall|exam_preparation|unknown\",\"targets\":[],\"reason\":\"\"}\n"
    "\n"
    "Examples:\n"
    "\n"
    "Question: What is Bayes theorem?\n"
    "Output:\n"
    "{\"route\":\"concept_explanation\",\"targets\":[],\"reason\":\"The user is asking to understand a concept\"}\n"
    "\n"
    "Question: Which document explains Bayes theorem?\n"
    "Output:\n"
    "{\"route\":\"source_recall\",\"targets\":[],\"reason\":\"The user is asking to locate a source\"}\n"
    "\n"
    "Question: What should I revise for the exam?\n"
    "Output:\n"
    "{\"route\":\"exam_preparation\",\"targets\":[],\"reason\":\"The user is asking about revision priorities\"}\n"
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

   
    if not isinstance(reason, str):
        reason = ""

    return {"route": route, "targets": targets, "reason": reason}


def plan_question(question: str) -> dict:
    messages = build_planner_messages(question)
    raw = chat(messages)  # must return text
    return parse_and_validate_plan(raw)
