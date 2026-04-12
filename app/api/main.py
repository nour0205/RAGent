from collections import defaultdict
import uuid

from fastapi import FastAPI

# ---- Core RAG imports ----
from app.ingestion.chunker import base_chunks, apply_overlap
from app.embeddings.embedder import embed_texts
from app.vectordb.chroma_store import ChromaStore
from app.rag.pipeline import rag_answer_with_sources
from app.rag.hybrid_retriever import hybrid_retrieve
from app.utils.hash import hash_text
from app.llm.client import chat
from app.catalog.document_catalog import upsert_document_entry

# ---- Orchestration imports ----
from app.orchestration.planner import plan_question
from app.orchestration.prompts import build_answer_prompt
from app.orchestration.utils import build_study_hint  

# ---- Schema imports ----
from app.schemas.api import (
    AnswerResponse,
    IngestRequest,
    QuestionRequest,
    SourceItem,
)

# ---- Whoosh imports ----
from app.vectordb.whoosh_index import add_chunks_to_whoosh


# -------------------------------------------------------------------
# App + Store
# -------------------------------------------------------------------
app = FastAPI(title="Recall API", version="2.0")
store = ChromaStore(collection_name="api-demo")


# -------------------------------------------------------------------
# ASK (simple RAG)
# -------------------------------------------------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    where = {}

    if req.document_id:
        where["document_id"] = req.document_id
    if req.owner:
        where["owner"] = req.owner

    if not where:
        where = None

    result = rag_answer_with_sources(
        question=req.question,
        store=store,
        k=5,
        where=where
    )

    return AnswerResponse(
        answer=result.answer,
        route="concept_explanation",
        sources=result.sources,
        study_hint=None,
    )


# -------------------------------------------------------------------
# INGEST
# -------------------------------------------------------------------
@app.post("/ingest")
def ingest(req: IngestRequest):
    doc_hash = hash_text(req.text)

    existing = store.collection.get(
        where={"doc_hash": doc_hash},
        limit=1
    )
    if existing["ids"]:
        return {"status": "duplicate"}

    existing_id = store.collection.get(
        where={"document_id": req.document_id},
        limit=1
    )
    if existing_id["ids"]:
        return {"status": "conflict"}

    chunks = base_chunks(req.text)
    chunks = apply_overlap(chunks, overlap=1)

    if not chunks:
        return {"status": "no content"}

    embeddings = embed_texts(chunks)

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = []

    for i in range(len(chunks)):
        meta = {
            "chunk_id": ids[i],
            "document_id": req.document_id,
            "doc_hash": doc_hash,
            "chunk_index": i,
            "document_type": req.document_type,
            "course": req.course,
            "topic_tags": req.topic_tags,
        }
        if req.source:
            meta["source"] = req.source
        if req.owner:
            meta["owner"] = req.owner
        metadatas.append(meta)

    whoosh_chunks = [
        {
            "chunk_id": ids[i],
            "document_id": req.document_id,
            "chunk_index": i,
            "text": chunks[i],
        }
        for i in range(len(chunks))
    ]

    store.add_texts(
        ids=ids,
        texts=chunks,
        embeddings=embeddings,
        metadatas=metadatas
    )

    add_chunks_to_whoosh(whoosh_chunks)

    upsert_document_entry({
        "document_id": req.document_id,
        "title": req.document_id,
        "preview": chunks[0][:300],
        "chunk_count": len(chunks),
    })

    return {"status": "ingested", "chunks_added": len(chunks)}


# -------------------------------------------------------------------
# ASK ROUTED (Recall version)
# -------------------------------------------------------------------
@app.post("/ask_routed", response_model=AnswerResponse)
def ask_routed(req: QuestionRequest):

    # 1. Plan (intent)
    plan = plan_question(req.question)
    route = plan["route"]

    # 2. Retrieve (global, multi-doc)
    where = {}
    if req.owner:
        where["owner"] = req.owner
    if not where:
        where = None

    results = hybrid_retrieve(
        store=store,
        question=req.question,
        k=5,
        where=where
    )

    if not results:
        return AnswerResponse(
            answer="I don't know.",
            route=route,
            sources=[],
            study_hint=None,
        )

    # 3. Build prompt
    chunks = [item.text for item in results]

    messages = build_answer_prompt(
        question=req.question,
        chunks=chunks,
        route=route
    )

    answer = chat(messages)

    # 4. Build sources
    sources = [
        SourceItem(
            document_id=item.document_id,
            chunk_index=item.chunk_index,
            text=item.text,
            retrieval_type=item.retrieval_type,
            hybrid_score=item.hybrid_score,
            rerank_score=getattr(item, "rerank_score", None),
        )
        for item in results
    ]

    # 5. Study hint
    study_hint = build_study_hint(route, sources)

    return AnswerResponse(
        answer=answer,
        route=route,
        sources=sources,
        study_hint=study_hint,
    )


# -------------------------------------------------------------------
# DEBUG
# -------------------------------------------------------------------
@app.post("/debug/retrieve")
def debug_retrieve(req: QuestionRequest):
    results = hybrid_retrieve(
        store=store,
        question=req.question,
        k=5
    )
    return {"results": [r.model_dump() for r in results]}


@app.post("/debug/plan")
def debug_plan(req: QuestionRequest):
    return {"plan": plan_question(req.question)}


# -------------------------------------------------------------------
# DOCUMENT LIST
# -------------------------------------------------------------------
@app.get("/documents")
@app.get("/documents")
def list_documents():
    data = store.collection.get(include=["metadatas", "documents"])

    grouped = defaultdict(lambda: {
        "document_id": None,
        "chunks": 0,
        "preview": None,
    })

    metadatas = data.get("metadatas") or []
    documents = data.get("documents") or []

    for meta, doc in zip(metadatas, documents):
        if meta is None:
            continue
        if doc is None:
            continue

        doc_id = meta.get("document_id")
        if not doc_id:
            continue

        entry = grouped[doc_id]
        entry["document_id"] = doc_id
        entry["chunks"] += 1

        if not entry["preview"]:
            entry["preview"] = doc[:200]

    return {"documents": list(grouped.values())}