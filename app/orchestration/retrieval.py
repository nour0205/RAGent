from app.embeddings.embedder import embed_texts

def retrieve_for_document(store, question: str, document_id: str, k: int = 4, retrieve_k: int = 10):
    query_embedding = embed_texts([question])[0]
    where = {"document_id": document_id}
    candidates = store.query_full(query_embedding, k=retrieve_k, where=where)
    return store.query_full(query_embedding, k=k, where=where)