import re

from app.schemas.retrieval import RetrievedChunk, RankedChunk


STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "why", "how", "what",
    "does", "do", "did", "and", "or", "of", "to", "in", "on", "for",
    "with", "by", "at", "from", "this", "that", "it", "as", "be"
}


def tokenize(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z0-9_]+\b", text.lower())


def keyword_overlap_score(question: str, chunk: str) -> float:
    q_tokens = [t for t in tokenize(question) if t not in STOPWORDS]
    c_tokens = set(tokenize(chunk))

    if not q_tokens:
        return 0.0

    overlap = sum(1 for token in q_tokens if token in c_tokens)

    phrase_bonus = 0.0
    lowered_q = question.lower().strip(" ?.")
    lowered_c = chunk.lower()
    if lowered_q and lowered_q in lowered_c:
        phrase_bonus = 2.0

    return float(overlap) + phrase_bonus


def rerank_items(
    question: str,
    items: list[RetrievedChunk],
    k: int = 4
) -> list[RankedChunk]:
    scored: list[RankedChunk] = []

    for original_rank, item in enumerate(items):
        text = item.text.strip()
        if not text:
            continue

        lexical_score = keyword_overlap_score(question, text)
        rank_bonus = max(0.0, 1.0 - (original_rank * 0.05))
        final_score = lexical_score + rank_bonus

        scored.append(
            RankedChunk(
                **item.model_dump(),
                rerank_score=final_score
            )
        )

    scored.sort(key=lambda x: x.rerank_score, reverse=True)
    return scored[:k]