import chromadb

class ChromaStore:
    def __init__(self, collection_name: str = "demo", persist_dir: str = ".chroma"):
        # 🔴 THIS IS THE CRITICAL CHANGE
        self.client = chromadb.PersistentClient(path=persist_dir)

        self.collection = self.client.get_or_create_collection(
            name=collection_name
        )

    def add_texts(self, ids: list[str], texts: list[str], embeddings: list[list[float]], metadatas):
        self.collection.add(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas
        )

    def query(self, query_embedding: list[float], k: int = 3, where: dict | None = None) -> list[str]:
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where
        )
        return result["documents"][0]

    def query_with_scores(
        self,
        query_embedding: list[float],
        k: int = 3,
        where: dict | None = None,
    ):
        result = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=k,
            where=where,
            include=["documents", "metadatas", "distances"]
        )

        rows = []
        for i in range(len(result["documents"][0])):
            rows.append({
                "rank": i + 1,
                "distance": result["distances"][0][i],
                "document": result["documents"][0][i],
                "metadata": result["metadatas"][0][i],
            })

        return rows
    
def query_full(self, query_embedding, k: int = 4, where: dict | None = None):
    results = self.collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        where=where,
        include=["documents", "metadatas", "distances"]
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]
    ids = results.get("ids", [[]])[0] if "ids" in results else [None] * len(documents)

    items = []
    for i in range(len(documents)):
        meta = metadatas[i] or {}
        chunk_id = ids[i] or meta.get("chunk_id")

        items.append({
            "id": chunk_id,
            "chunk_id": chunk_id,
            "text": documents[i],
            "metadata": meta,
            "score": float(distances[i]),
            "rank": i + 1,
            "retrieval_type": "vector",
        })

    return items