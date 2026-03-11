def build_rag_prompt(question: str, contexts: list[str]) -> list[dict]:
    # Label contexts so the model can cite them.
    context_block = "\n\n".join(
        [f"[{i+1}] {ctx}" for i, ctx in enumerate(contexts)]
    )

    system = (
        "You are a strict assistant that answers ONLY using the provided context.\n"
        "Rules:\n"
        "- If the answer is not in the context, say: \"I don't know.\"\n"
        "- Do NOT use outside knowledge.\n"
        "- Cite sources using bracket numbers like [1], [2].\n"
    )

    user = f"Context:\n{context_block}\n\nQuestion: {question}\nAnswer:"

    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]
from app.embeddings.embedder import embed_texts

from app.llm.client import chat
from app.rag.pipeline import build_rag_prompt  # (or keep in same file)




import re


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

    # bonus if the full important phrase appears
    phrase_bonus = 0.0
    lowered_q = question.lower().strip(" ?.")
    lowered_c = chunk.lower()
    if lowered_q and lowered_q in lowered_c:
        phrase_bonus = 2.0

    return float(overlap) + phrase_bonus


def rerank_items(question: str, items: list[dict], k: int = 4) -> list[dict]:
    scored = []

    for original_rank, item in enumerate(items):
        text = item["text"]
        lexical_score = keyword_overlap_score(question, text)

        # small bonus for earlier vector retrieval rank
        rank_bonus = max(0.0, 1.0 - (original_rank * 0.05))

        final_score = lexical_score + rank_bonus

        scored.append((final_score, item))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [item for _, item in scored[:k]]

from app.embeddings.embedder import embed_texts
from app.llm.client import chat
from app.vectordb.chroma_store import ChromaStore

def rag_answer_with_store(question: str, store, k: int = 4, where: dict | None = None, retrieve_k: int = 10) -> str:
    query_embedding = embed_texts([question])[0]

    candidates = store.query_full(query_embedding, k=retrieve_k, where=where)
    items = rerank_items(question, candidates, k=k)

    contexts = store.query(query_embedding, k=k,  where=where)

   

    messages = build_rag_prompt(question, contexts)
    return chat(messages)



from app.embeddings.embedder import embed_texts
from app.llm.client import chat



def rag_answer_with_sources(question: str, store, k: int = 4, where: dict | None = None, retrieve_k: int = 10):
    query_embedding = embed_texts([question])[0]


    candidates = store.query_full(query_embedding, k=retrieve_k, where=where)
    items = rerank_items(question, candidates, k=k)

    contexts = [item["text"] for item in items]
    messages = build_rag_prompt(question, contexts)
    answer = chat(messages)

    sources = []
    for item in items:
        meta = item["metadata"] or {}
        sources.append({
            "document_id": meta.get("document_id", "unknown"),
            "chunk_index": meta.get("chunk_index"),
            "text": item["text"]
        })

    return {
        "answer": answer,
        "sources": sources
    }
