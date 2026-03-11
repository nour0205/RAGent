def build_compare_prompt(question: str, ctx_a: list[str], ctx_b: list[str]) -> list[dict]:
    a_block = "\n".join([f"[A{i+1}] {t}" for i, t in enumerate(ctx_a)])
    b_block = "\n".join([f"[B{i+1}] {t}" for i, t in enumerate(ctx_b)])

    system = (
        "You compare two sources using ONLY the provided contexts.\n"
        "Rules:\n"
        "- If the answer is missing from either side, say: \"I don't know.\"\n"
        "- Do not use outside knowledge.\n"
        "- Cite sources like [A1] or [B2].\n"
    )

    user = (
        f"Context A:\n{a_block}\n\n"
        f"Context B:\n{b_block}\n\n"
        f"Question: {question}\nAnswer:"
    )

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
