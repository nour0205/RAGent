from collections import defaultdict
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
app = FastAPI(title="Recall API", version="3.0")
store = ChromaStore(collection_name="api-demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------
def build_where_clause(req: QuestionRequest):
    where = {}

    if req.document_id:
        where["document_id"] = req.document_id
    if req.owner:
        where["owner"] = req.owner

    return where if where else None


def to_source_items(items):
    return [
        SourceItem(
            document_id=item.document_id,
            chunk_index=item.chunk_index,
            text=item.text,
            retrieval_type=item.retrieval_type,
            hybrid_score=getattr(item, "hybrid_score", None),
            rerank_score=getattr(item, "rerank_score", None),
        )
        for item in items
    ]


# -------------------------------------------------------------------
# ASK (merged Recall v3)
# -------------------------------------------------------------------
@app.post("/ask", response_model=AnswerResponse)
def ask(req: QuestionRequest):
    where = build_where_clause(req)

    # 1. Plan the study intent
    plan = plan_question(req.question)
    route = plan.get("route", "unknown")

    # 2. Try routed retrieval first
    results = hybrid_retrieve(
        store=store,
        question=req.question,
        k=5,
        where=where,
    )

    routed_sources = to_source_items(results[:3]) if results else []

    routed_answer = None
    routed_ok = False

    # 3. Try routed generation if retrieval returned something
    if results:
        messages = build_answer_prompt(
            question=req.question,
            sources=results[:3],
            route=route,
        )

        try:
            routed_answer = chat(messages).strip()
            if routed_answer and routed_answer != "I don't know.":
                routed_ok = True
        except Exception:
            routed_ok = False

    # 4. Return routed answer if successful
    if routed_ok:
        study_hint = build_study_hint(route, routed_sources)
        return AnswerResponse(
            answer=routed_answer,
            route=route,
            sources=routed_sources,
            study_hint=study_hint,
        )

    # 5. Fallback to reliable generic RAG
    fallback = rag_answer_with_sources(
        question=req.question,
        store=store,
        k=5,
        where=where,
    )

    fallback_sources = fallback.sources if fallback.sources else []
    source_items = to_source_items(fallback_sources)

    study_hint = build_study_hint(route, source_items)

    return AnswerResponse(
        answer=fallback.answer,
        route=route,
        sources=source_items,
        study_hint=study_hint,
    )


# -------------------------------------------------------------------
# INGEST
# -------------------------------------------------------------------
@app.post("/ingest")
def ingest(req: IngestRequest):
    doc_hash = hash_text(req.text)

    existing = store.collection.get(
        where={"doc_hash": doc_hash},
        limit=1,
    )
    if existing["ids"]:
        return {"status": "duplicate"}

    existing_id = store.collection.get(
        where={"document_id": req.document_id},
        limit=1,
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
            "topic_tags": ", ".join(req.topic_tags) if req.topic_tags else "",
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
        metadatas=metadatas,
    )

    add_chunks_to_whoosh(whoosh_chunks)

    upsert_document_entry(
        {
            "document_id": req.document_id,
            "title": req.document_id,
            "preview": chunks[0][:300],
            "chunk_count": len(chunks),
        }
    )

    return {"status": "ingested", "chunks_added": len(chunks)}


# -------------------------------------------------------------------
# DEBUG
# -------------------------------------------------------------------
@app.post("/debug/retrieve")
def debug_retrieve(req: QuestionRequest):
    where = build_where_clause(req)

    results = hybrid_retrieve(
        store=store,
        question=req.question,
        k=5,
        where=where,
    )

    return {"results": [r.model_dump() for r in results]}


@app.post("/debug/plan")
def debug_plan(req: QuestionRequest):
    return {"plan": plan_question(req.question)}


# -------------------------------------------------------------------
# DOCUMENT LIST
# -------------------------------------------------------------------
@app.get("/documents")
def list_documents():
    data = store.collection.get(include=["metadatas", "documents"])

    grouped = defaultdict(
        lambda: {
            "document_id": None,
            "chunks": 0,
            "preview": None,
        }
    )

    metadatas = data.get("metadatas") or []
    documents = data.get("documents") or []

    for meta, doc in zip(metadatas, documents):
        if meta is None or doc is None:
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