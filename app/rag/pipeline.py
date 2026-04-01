from app.llm.client import chat
from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.reranker import rerank_items
from app.schemas.api import SourceItem, RagAnswerResult


def build_rag_prompt(question: str, contexts: list[str]) -> list[dict]:
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


def rag_answer_with_store(
    question: str,
    store,
    k: int = 4,
    where: dict | None = None,
    retrieve_k: int = 10
) -> str:
    candidates = hybrid_retrieve(
        store=store,
        question=question,
        k=retrieve_k,
        where=where
    )

    items = rerank_items(question, candidates, k=k)

    contexts = [item.text for item in items if item.text.strip()]

    if not contexts:
        return "I don't know."

    messages = build_rag_prompt(question, contexts)
    return chat(messages)


def rag_answer_with_sources(
    question: str,
    store,
    k: int = 4,
    where: dict | None = None,
    retrieve_k: int = 10
) -> RagAnswerResult:
    candidates = hybrid_retrieve(
        store=store,
        question=question,
        k=retrieve_k,
        where=where
    )

    items = rerank_items(question, candidates, k=k)

    contexts = [item.text for item in items if item.text.strip()]

    if not contexts:
        return RagAnswerResult(
            answer="I don't know.",
            sources=[]
        )

    messages = build_rag_prompt(question, contexts)
    answer = chat(messages)

    if answer.strip() == "I don't know.":
        return RagAnswerResult(
            answer=answer,
            sources=[]
        )

    sources = [
        SourceItem(
            document_id=item.document_id,
            chunk_index=item.chunk_index,
            text=item.text,
            retrieval_type=item.retrieval_type,
            hybrid_score=item.hybrid_score,
            rerank_score=item.rerank_score,
        )
        for item in items
    ]

    return RagAnswerResult(
        answer=answer,
        sources=sources
    )