from app.rag.hybrid_retriever import hybrid_retrieve
from app.rag.pipeline import rerank_items
from app.orchestration.registry import normalize_document_id

def retrieve_for_document(
    store,
    question: str,
    document_id: str,
    k: int = 4,
    retrieve_k: int = 10
):
    resolved_id = normalize_document_id(document_id)
    where = {"document_id": resolved_id}

    print("\n[DEBUG] retrieve_for_document")
    print("  requested:", document_id)
    print("  resolved :", resolved_id)
    print("  where    :", where)

    candidates = hybrid_retrieve(
        store=store,
        question=question,
        k=retrieve_k,
        where=where
    )

    items = rerank_items(question, candidates, k=k)

    normalized = []
    for item in items:
        item_copy = dict(item)
        item_copy["text"] = item.get("text") or item.get("document") or ""
        normalized.append(item_copy)

    return normalized