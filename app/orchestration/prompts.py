def build_answer_prompt(question: str, chunks: list[str], route: str) -> list[dict]:
    context_block = "\n".join([f"[{i+1}] {t}" for i, t in enumerate(chunks)])

    if route == "concept_explanation":
        instruction = (
            "Explain the concept clearly and simply using only the provided notes. "
            "Make it feel like you are teaching from the student's own material. "
            "Define the concept first, then mention the most important related ideas from the notes."
        )
        answer_shape = (
            "Structure:\n"
            "1. A short definition\n"
            "2. A brief explanation based on the notes\n"
            "3. Key ideas to remember\n"
            "4. A short review suggestion\n"
        )

    elif route == "source_recall":
        instruction = (
            "Answer by identifying where the concept appears in the provided notes. "
            "Name the most relevant document(s) first, then briefly say what was explained there. "
            "Be direct and source-focused."
        )
        answer_shape = (
            "Structure:\n"
            "1. State the most relevant document name(s)\n"
            "2. Briefly explain what that source says about the concept\n"
            "3. Add a short review suggestion\n"
        )

    elif route == "exam_preparation":
        instruction = (
            "Create a revision-oriented answer using only the provided notes. "
            "Focus on what the student should review first, what ideas belong together, "
            "and what is most important to understand for studying."
        )
        answer_shape = (
            "Structure:\n"
            "1. Start with: 'From your notes, the main things to review are:'\n"
            "2. List 3 to 5 key topics to revise\n"
            "3. For each topic, give a short explanation tied to the notes\n"
            "4. End with a short study priority suggestion\n"
        )

    else:
        instruction = "Answer the question using only the provided notes."
        answer_shape = (
            "Structure:\n"
            "1. A direct answer\n"
            "2. Key supporting points from the notes\n"
            "3. A short review suggestion\n"
        )

    system = (
        "You are Recall, a study assistant that helps a student learn from their own notes.\n\n"
        "Use ONLY the provided context.\n\n"
        "Rules:\n"
        "- Do NOT use outside knowledge.\n"
        "- If the answer is not supported by the context, say exactly: 'I don't know.'\n"
        "- Do NOT mention concepts that do not appear in the provided context.\n"
        "- Every major point in the answer must be directly supported by the notes.\n"
        "- Stay grounded in the notes.\n"
        "- Prefer phrases like 'From your notes', 'The material explains', or 'These notes describe' when helpful.\n"
        "- Do NOT mention chunk numbers like [1] or [2].\n"
        "- Do NOT invent document names or facts.\n"
        "- Keep the answer concise, clear, and study-friendly.\n"
        "- If multiple sources are relevant, synthesize them carefully.\n"
    )

    user = (
        f"Instruction:\n{instruction}\n\n"
        f"{answer_shape}\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{question}\n\n"
        f"Write the answer now."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]