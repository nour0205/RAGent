def build_answer_prompt(question: str, chunks: list[str], route: str) -> list[dict]:
    context_block = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(chunks)])

    # Intent-aware instruction (simple)
    if route == "concept_explanation":
        instruction = "Explain the concept clearly and simply."
    elif route == "source_recall":
        instruction = "Focus on identifying where the concept is explained."
    elif route == "exam_preparation":
        instruction = "Highlight important topics and what should be revised."
    else:
        instruction = "Answer the question."

    system = (
    "You are a study assistant.\n"
    "\n"
    "Answer the user's question using ONLY the provided context.\n"
    "\n"
    "Rules:\n"
    "- Do NOT use external knowledge.\n"
    "- If the answer is not in the context, say: \"I don't know.\"\n"
    "- Keep explanations clear and simple.\n"
    "-  Do NOT use numbered citations like [1] or [2].\n"
    "- Refer to sources naturally using document names if needed.\n"
    "Structure your answer like this:\n"
    "1. A clear explanation (2–4 sentences)\n"
    "2. (Optional) Key points if helpful\n"
    "3. A short review suggestion at the end\n"
)

    user = (
        f"Instruction:\n{instruction}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with sources and review suggestion):"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]