LexQuery

LexQuery is a legal-domain Retrieval-Augmented Generation (RAG) system for querying a collection of legal and statutory documents.

The project combines LLM-based answer generation, vector embeddings, vector databases, lexical retrieval, retrieval evaluation, and an LLM-based review loop to produce answers grounded in source documents.

Key Concepts

RAG (Retrieval-Augmented Generation): retrieves relevant document passages before generating an answer.

Embeddings: converts document chunks and user queries into numerical vectors representing semantic meaning.

Vector Database: Qdrant stores document embeddings and performs similarity search.

Vectorless / Lexical RAG: SQLite FTS5 provides a keyword-based retrieval path without embeddings.

Chunking: PDF text is split into overlapping, page-aware chunks to provide useful retrieval units while preserving citation metadata.

LLM Review Loop: a separate model evaluates generated answers for relevance, groundedness, and completeness. Failed answers can trigger query rewriting and another retrieval/generation round.

Source Citations: retrieved chunks retain source file and page metadata, allowing the UI to link directly to the cited PDF page.

Retrieval Evaluation: vector and lexical retrieval modes can be compared using latency, source coverage, overlap, and answer-quality metrics.

Fine-Tuning: the repository includes a LoRA/QLoRA training and evaluation scaffold for experimenting with legal-domain model adaptation.

Architecture

                         ┌─────────────────────┐
                         │     User / UI       │
                         │     Vite Frontend   │
                         └──────────┬──────────┘
                                    │
                              POST /rag
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │     FastAPI API     │
                         └──────────┬──────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │                               │
                    ▼                               ▼
          ┌──────────────────┐            ┌──────────────────┐
          │  Vector RAG      │            │  Lexical RAG     │
          │                  │            │                  │
          │ Ollama Embedding │            │ SQLite FTS5      │
          │       ↓          │            │ pageIndex        │
          │     Qdrant       │            │                  │
          └────────┬─────────┘            └────────┬─────────┘
                   │                               │
                   └───────────────┬───────────────┘
                                   ▼
                         ┌─────────────────────┐
                         │ Retrieved Context   │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Local LLM (Qwen)    │
                         │ Answer Generation   │
                         └──────────┬──────────┘
                                    ▼
                         ┌─────────────────────┐
                         │ Optional LLM Judge  │
                         │ Relevance /         │
                         │ Groundedness /      │
                         │ Completeness        │
                         └──────────┬──────────┘
                                    │
                         Failed review → Query
                         rewrite → Retrieve again
                                    │
                                    ▼
                         ┌─────────────────────┐
                         │ Answer + Citations  │
                         └─────────────────────┘

Technology Stack

Layer

Technology

Backend API

Python, FastAPI

Frontend

JavaScript, Vite

PDF processing

pypdf

Embedding model

nomic-embed-text via Ollama

Vector database

Qdrant

Local LLM

Qwen 2.5 7B via Ollama

Lexical retrieval

SQLite FTS5

Review / query rewriting

Gemini 2.5 Flash

Fine-tuning

Hugging Face + LoRA/QLoRA

Evaluation

Python, CSV/JSON metrics, Matplotlib

Document Ingestion

Documents are stored in data/ as PDFs.

The ingestion pipeline:

Extract text from PDFs page by page.

Normalize whitespace and preserve paragraph boundaries.

Split text into approximately 1800-character chunks with overlap.

Preserve metadata such as:

source filename

starting and ending page

chunk index

Generate embeddings using nomic-embed-text.

Store embeddings and metadata in the Qdrant collection btp_docs.

Store the same chunks in a SQLite FTS5 index for lexical/pageIndex retrieval.

This creates two retrieval representations of the same document corpus:

PDF
 │
 ├──► Text chunks ──► Embeddings ──► Qdrant
 │
 └──► Text chunks ──► SQLite FTS5 ──► Lexical / pageIndex retrieval

Retrieval

LexQuery supports two retrieval modes.

1. Vector Retrieval

The query is embedded using the same embedding model used during document ingestion.

User Query
    ↓
Embedding Model
    ↓
Query Vector
    ↓
Qdrant Similarity Search
    ↓
Top-K Relevant Chunks

Qdrant uses cosine similarity for vector comparison.

This approach is useful when the relevant document passage uses different wording from the user's query but has similar semantic meaning.

2. Lexical / pageIndex Retrieval

The project also maintains a SQLite FTS5 index.

User Query
    ↓
Tokenization
    ↓
SQLite FTS5 Search
    ↓
BM25 Ranking
    ↓
Top-K Chunks

This provides a vector-free retrieval path and makes it possible to experimentally compare semantic retrieval against traditional lexical search.

RAG Generation

After retrieval, the selected chunks are assembled into a bounded context window.

The local Qwen model then receives:

the user's question

retrieved document context

source/chunk identifiers

The model is instructed to ground its answer in the retrieved context and cite supporting context blocks.

The final API response includes the generated answer together with source metadata.

LLM Review Loop

LexQuery optionally evaluates the generated answer using Gemini.

The reviewer scores:

Relevance

Groundedness

Completeness

If the answer does not meet the review threshold, the system can ask the model to rewrite the retrieval query and perform another retrieval + generation cycle.

Question
   ↓
Retrieve
   ↓
Generate Answer
   ↓
LLM Judge
   │
   ├── Pass ───────────────► Return Answer
   │
   └── Retry
         ↓
    Rewrite Query
         ↓
      Retrieve
         ↓
      Generate

This introduces a lightweight retrieval feedback loop rather than relying on a single retrieval pass.

Citations

Retrieved chunks retain page-level metadata.

The backend exposes source documents through the API and generates URLs containing the cited page number. The frontend displays these as clickable source cards.

This allows a user to move from:

Generated Answer
      ↓
Citation
      ↓
Source PDF
      ↓
Cited Page

Evaluation

The evaluate/ directory contains experiments for comparing retrieval approaches.

Current evaluation outputs include:

retrieval latency

source coverage

context size

pairwise document overlap

Jaccard similarity

answer relevance

answer groundedness

answer completeness

Example:

python3 evaluate/compare_rag_modes.py     --topk 5     --with-answers     --outdir evaluate/rag_mode_outputs

python3 evaluate/visualize_rag_modes.py     --outdir evaluate/rag_mode_outputs     --topk 5

The generated results are stored as JSON/CSV files and visualized as plots.

Fine-Tuning

The fine_tuning/ directory provides an experimental pipeline for adapting a causal language model to legal-domain data.

It includes:

SFT dataset preparation

LoRA/QLoRA training scaffold

baseline vs candidate model evaluation

result visualization

The current repository contains the fine-tuning workflow scaffold; a completed fine-tuning run should only be claimed after an actual training and evaluation run has been performed.

Project Structure

LexQuery/
├── data/                         # Legal and statutory PDFs
├── create_embeddings.py         # PDF processing + embeddings + Qdrant ingestion
├── build_lexical_index.py       # Build lexical/pageIndex index
├── lexical_index.py              # SQLite FTS5 retrieval
├── make_call_to_fine_tuned_llm.py# FastAPI RAG backend
├── docker-compose.yaml           # Qdrant service
├── qdrant_storage/               # Local Qdrant storage
├── lexical_chunks.db             # SQLite FTS5 index
├── evaluate/                     # Retrieval evaluation and plots
├── fine_tuning/                  # LoRA/QLoRA experimentation
├── ui/legal/                     # Vite frontend
└── IMPLEMENTATION_ROADMAP.md     # Implementation status and roadmap

Local Setup

Prerequisites

Install:

Python 3.10+

Node.js / npm

Docker

Ollama

Pull the required local models:

ollama pull qwen2.5:7b
ollama pull nomic-embed-text

1. Install Python Dependencies

pip install -r requirements.txt

2. Start Qdrant

docker compose up -d qdrant

Qdrant will be available at:

http://127.0.0.1:6333

3. Build the Knowledge Base

Generate embeddings and populate Qdrant:

python3 create_embeddings.py

Build/rebuild the lexical index:

python3 build_lexical_index.py

4. Start the Backend

uvicorn make_call_to_fine_tuned_llm:app     --host 127.0.0.1     --port 8000     --reload

The API runs at:

http://127.0.0.1:8000

If review/query-rewrite functionality is enabled, configure GOOGLE_API_KEY in the environment or .env.

5. Start the Frontend

cd ui/legal
npm install
npm run dev

The Vite development server runs on:

http://127.0.0.1:5173

API

The primary endpoint is:

POST /rag

Example request:

{
  "query": "What does Article 14 of the Constitution provide?",
  "top_k": 5,
  "retrieval_mode": "vector",
  "enable_review": true,
  "max_review_rounds": 2
}

Supported retrieval modes:

vector

page_index

The response contains:

generated answer

citations

retrieval mode used

review metadata

final retrieval query

Project Status

Implemented components include:

PDF ingestion and page-aware chunking

embedding generation

Qdrant vector storage

SQLite FTS5 lexical retrieval

FastAPI RAG backend

local Ollama-based answer generation

clickable PDF citations

optional LLM answer review

query rewriting and retry

retrieval-mode evaluation

fine-tuning workflow scaffold

The project is intended as an experimental legal RAG system and research/academic project rather than a substitute for professional legal advice.
