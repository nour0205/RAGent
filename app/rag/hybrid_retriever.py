from app.embeddings.embedder import embed_texts
from app.vectordb.whoosh_index import search_whoosh
from app.schemas.retrieval import RetrievedChunk


def _get_chunk_key(item: dict) -> str:
    if "id" in item:
        return str(item["id"])

    meta = item.get("metadata", {}) or {}

    if "chunk_id" in meta:
        return str(meta["chunk_id"])

    document_id = meta.get("document_id", "unknown_doc")
    chunk_index = meta.get("chunk_index", "unknown_chunk")
    return f"{document_id}:{chunk_index}"


def _get_metadata(item: dict) -> dict:
    return item.get("metadata", {}) or {}


def _get_text(item: dict) -> str:
    return item.get("text") or item.get("document") or ""


def _to_retrieved_chunk(item: dict) -> RetrievedChunk:
    meta = _get_metadata(item)

    return RetrievedChunk(
        id=str(item.get("id") or _get_chunk_key(item)),
        document_id=meta.get("document_id", "unknown"),
        chunk_id=meta.get("chunk_id"),
        chunk_index=meta.get("chunk_index"),
        text=_get_text(item),
        retrieval_type=item.get("retrieval_type", "unknown"),
        score=item.get("score"),
        bm25_score=item.get("bm25_score"),
        hybrid_score=item.get("hybrid_score"),
        rank=item.get("rank"),
        found_by_vector=item.get("found_by_vector", False),
        found_by_bm25=item.get("found_by_bm25", False),
    )


def reciprocal_rank_fusion(
    vector_results: list[dict],
    bm25_results: list[dict],
    k: int = 60
) -> list[RetrievedChunk]:
    scores = {}
    merged = {}

    for item in vector_results:
        chunk_id = _get_chunk_key(item)
        rank = item["rank"]

        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

        item_copy = dict(item)
        item_copy["id"] = chunk_id
        item_copy["retrieval_type"] = "vector"
        item_copy["found_by_vector"] = True
        item_copy["found_by_bm25"] = False

        merged[chunk_id] = item_copy

    for item in bm25_results:
        chunk_id = _get_chunk_key(item)
        rank = item["rank"]

        scores[chunk_id] = scores.get(chunk_id, 0) + 1 / (k + rank)

        if chunk_id in merged:
            merged[chunk_id]["found_by_bm25"] = True
            merged[chunk_id]["retrieval_type"] = "hybrid"
            merged[chunk_id]["bm25_score"] = item.get("score")
        else:
            item_copy = dict(item)
            item_copy["id"] = chunk_id
            item_copy["retrieval_type"] = "bm25"
            item_copy["found_by_vector"] = False
            item_copy["found_by_bm25"] = True
            merged[chunk_id] = item_copy

    fused: list[RetrievedChunk] = []

    for chunk_id, item in merged.items():
        new_item = dict(item)
        new_item["hybrid_score"] = scores[chunk_id]
        fused.append(_to_retrieved_chunk(new_item))

    fused.sort(key=lambda x: x.hybrid_score or 0.0, reverse=True)
    return fused


def hybrid_retrieve(
    store,
    question: str,
    k: int = 5,
    where: dict | None = None
) -> list[RetrievedChunk]:
    query_embedding = embed_texts([question])[0]

    vector_results = store.query_with_scores(
        query_embedding=query_embedding,
        k=k,
        where=where
    )

    for rank, item in enumerate(vector_results):
        item["rank"] = rank + 1
        item["retrieval_type"] = "vector"
        item["id"] = _get_chunk_key(item)

    bm25_results = search_whoosh(question, limit=k)

    for rank, item in enumerate(bm25_results):
        item["rank"] = rank + 1
        item["retrieval_type"] = "bm25"
        item["id"] = _get_chunk_key(item)

    if where:
        filtered_bm25 = []
        for item in bm25_results:
            meta = item.get("metadata", {}) or {}
            ok = True
            for key, value in where.items():
                if meta.get(key) != value:
                    ok = False
                    break
            if ok:
                filtered_bm25.append(item)
        bm25_results = filtered_bm25

    fused = reciprocal_rank_fusion(vector_results, bm25_results)

    return fused[:k]