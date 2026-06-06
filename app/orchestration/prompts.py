def build_answer_prompt(question: str, sources: list, route: str) -> list[dict]:
    context_block = "\n".join([
        f"[S{i+1}] ({sources[i].document_id}) {sources[i].text}"
        for i in range(len(sources))
    ])

    if route == "concept_explanation":
        instruction = (
            "Explain the concept clearly and simply using only the provided notes. "
            "Make it feel like a clean study brief from the student's own material."
        )
        answer_shape = (
            "Use this exact Markdown structure:\n\n"
            "## Definition\n"
            "Give a short definition in 1-2 sentences.\n\n"
            "## Explanation\n"
            "Explain the concept using the notes. Keep it clear and student-friendly.\n\n"
            "## Key Ideas\n"
            "- Write 3 to 5 bullet points.\n"
            "- Each bullet should be directly supported by the notes.\n\n"
            "## Review Tip\n"
            "Give one short practical review suggestion."
        )

    elif route == "source_recall":
        instruction = (
            "Help the student locate where the concept appears in their notes. "
            "Be direct and source-focused."
        )
        answer_shape = (
            "Use this exact Markdown structure:\n\n"
            "## Where It Appears\n"
            "Name the most relevant document or documents.\n\n"
            "## What The Notes Say\n"
            "Briefly explain what those notes say about the concept.\n\n"
            "## Review Tip\n"
            "Give one short practical review suggestion."
        )

    elif route == "exam_preparation":
        instruction = (
            "Create a revision-oriented study brief using only the provided notes. "
            "Focus on what the student should review first."
        )
        answer_shape = (
    "Use this exact Markdown structure:\n\n"

    "## What To Review\n"
    "- List 3 to 5 topics.\n"
    "- For each topic:\n"
    "  - Start with **bold topic name**.\n"
    "  - Give a one-sentence explanation.\n"
    "  - End with a citation such as [S1].\n\n"

    "## Key Connections\n"
    "- Briefly explain how the topics relate to each other.\n"
    "- Use 1 short paragraph only.\n"
    "- Include citations when relevant.\n\n"

    "## Exam Focus\n"
    "- Write 3 short bullet points.\n"
    "- Focus on what is most important to understand or remember.\n"
    "- Include citations where appropriate.\n\n"

    "## Review Tip\n"
    "- Give one short practical revision suggestion."
)

    else:
        instruction = "Answer the question using only the provided notes."
        answer_shape = (
            "Use this exact Markdown structure:\n\n"
            "## Answer\n"
            "Give a direct answer.\n\n"
            "## Supporting Points\n"
            "- Write 2 to 5 bullet points from the notes.\n\n"
            "## Review Tip\n"
            "Give one short practical review suggestion."
        )

    system = (
        "You are Recall, a study assistant that helps students learn from their own notes.\n\n"

        "Your goal is not only to answer questions, but to help students understand, revise, and remember concepts.\n\n"

        "Use ONLY the provided context.\n\n"

        "Grounding Rules:\n"
        "- Never use outside knowledge.\n"
        "- If the answer is not supported by the notes, say exactly: 'I don't know.'\n"
        "- Every important claim must come from the provided notes.\n"
        "- Never invent facts, examples, concepts, or document names.\n"
        "- Only cite sources that appear in the provided context.\n\n"

        "Writing Style:\n"
        "- Write like a high-quality study guide.\n"
        "- Be concise and easy to revise from.\n"
        "- Prefer short paragraphs.\n"
        "- Avoid repetition.\n"
        "- Avoid academic jargon when a simpler explanation is possible.\n"
        "- Do not write long textbook-style explanations.\n"
        "- Focus on understanding rather than completeness.\n\n"

        "Formatting:\n"
        "- Use Markdown.\n"
        "- Use section headings with ##.\n"
        "- Use bullet points for key ideas.\n"
        "- Use **bold** for important concepts.\n"
        "- Keep sections compact and visually scannable.\n\n"

        "Citations:\n"
        "- Add citations using [S1], [S2], [S3].\n"
        "- Cite only the most important factual claims.\n"
        "- Do not cite every sentence.\n"
        "- Do not place multiple citations on every bullet point.\n"
        "- The answer should feel naturally grounded, not overloaded with citations.\n\n"

        "Answer Quality:\n"
        "- Keep explanations between 3 and 6 sentences when possible.\n"
        "- Prefer key ideas over lengthy detail.\n"
        "- Prioritize what a student should remember for revision.\n"
        "- End with a practical review tip when appropriate.\n"
    )

    user = (
        f"Instruction:\n{instruction}\n\n"
        f"{answer_shape}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question:\n{question}\n\n"
        f"Write the answer now as a polished study brief."
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]