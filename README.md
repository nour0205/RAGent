# Recall

**Recall** is a retrieval-first study assistant that helps students learn from their own lecture materials.

Instead of generating answers from general knowledge, Recall retrieves relevant content from uploaded notes, lectures, and study documents, then produces grounded explanations with transparent source attribution and personalized study guidance.

The goal is not simply to answer questions, but to help students revisit concepts, locate where they were taught, and identify what to review next.

---

## Why Recall?

Students often struggle with:

* Finding where a concept was originally covered
* Revisiting large collections of lecture notes
* Identifying what topics deserve further revision
* Trusting AI-generated answers that lack sources

Recall addresses these challenges through a retrieval-first architecture that keeps responses grounded in the student's own materials.

---

## Core Features

### Grounded Answers

Every response is generated using retrieved lecture content rather than relying solely on the language model.

### Intent-Aware Learning

Recall identifies the user's study intent and adapts responses accordingly.

Supported intents include:

* Concept Explanation
* Source Recall
* Exam Preparation
* Unknown / Fallback

### Hybrid Retrieval

Combines:

* Semantic search (vector embeddings)
* Keyword search (Whoosh)

to improve recall and retrieval quality.

### Reranking

Retrieved chunks are reranked before answer generation to prioritize the most relevant study material.

### Source Attribution

Each answer includes supporting lecture passages used during generation.

### Study Guidance

Recall provides study hints that suggest what concepts or materials should be reviewed next.

### Knowledge Base Management

Users can ingest, browse, and manage study materials through the frontend interface.

---

## Example Workflow

1. Upload lecture notes
2. Notes are chunked and indexed
3. User asks a question
4. Recall identifies the learning intent
5. Relevant chunks are retrieved using hybrid search
6. Results are reranked
7. Grounded answer is generated
8. Sources and study recommendations are returned

---

## Architecture

```text
User Question
       │
       ▼
Intent Classification
       │
       ▼
Hybrid Retrieval
(Vector + Keyword)
       │
       ▼
Reranking
       │
       ▼
Grounded Generation
       │
       ▼
Answer + Sources + Study Hint
```

### Backend

```text
app/
├── api/
├── embeddings/
├── ingestion/
├── llm/
├── orchestration/
├── rag/
├── schemas/
├── vectordb/
└── utils/
```

### Frontend

```text
recall-frontend/
└── React + TypeScript
```

---

## Tech Stack

### Backend

* FastAPI
* ChromaDB
* Whoosh
* OpenAI-compatible LLM API
* Pydantic

### Frontend

* React
* TypeScript
* Tailwind CSS

### Retrieval

* Semantic embeddings
* Hybrid retrieval
* Reranking pipeline

---

## API Endpoints

### Ask a Question

```http
POST /ask
```

Returns:

```json
{
  "answer": "...",
  "route": "exam_preparation",
  "sources": [],
  "study_hint": "..."
}
```

### Ingest a Document

```http
POST /ingest
```

### List Documents

```http
GET /documents
```

---

## Running the Project

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Start Backend

```bash
python -m uvicorn app.api.main:app --reload
```

Backend:

```text
http://127.0.0.1:8000
```

API Docs:

```text
http://127.0.0.1:8000/docs
```

### Start Frontend

```bash
cd recall-frontend
npm install
npm run dev
```

Frontend:

```text
http://localhost:5173
```

---

## Current Limitations

* Basic intent classification
* Heuristic reranking
* No document update/delete workflow
* Single-user local deployment
* Limited evaluation coverage

---

## Future Improvements

* Cross-encoder reranking
* Semantic document routing
* Query rewriting
* Retrieval evaluation dashboard
* Multi-user support
* Study-path generation
* Streaming responses

---

## License

MIT License
