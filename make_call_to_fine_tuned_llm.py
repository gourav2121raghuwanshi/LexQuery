"""
FastAPI RAG server with:
- vector retrieval via Qdrant
- vectorless pageIndex retrieval via SQLite FTS5
- LLM answer review loop with query rewrite retries
- clickable PDF citations

Run:
  uvicorn make_call_to_fine_tuned_llm:app --host 127.0.0.1 --port 8000 --reload
"""

from __future__ import annotations

import json
import logging
import os
import re
import threading
from datetime import datetime, timezone
from typing import Any, Dict, List, Literal, Optional
from urllib.parse import quote
from uuid import uuid4

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from lexical_index import connect_index, initialize_index, search_lexical


# -----------------------------
# Config
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
LEXICAL_DB_PATH = os.path.join(BASE_DIR, "lexical_chunks.db")
ENV_PATH = os.path.join(BASE_DIR, ".env")
QUERY_HISTORY_PATH = os.path.join(BASE_DIR, "query_history.json")


def load_env_file(path: str) -> None:
    """Load KEY=VALUE pairs from a local .env file if present."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value


load_env_file(ENV_PATH)

OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
GEMINI_API_URL = "https://generativelanguage.googleapis.com/v1beta/models"

LLM_MODEL = "qwen2.5:7b"
JUDGE_MODEL = "gemini-2.5-flash"
REWRITE_MODEL = "gemini-2.5-flash"
EMBED_MODEL = "nomic-embed-text"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "").strip()

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "btp_docs"

DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT = 250
REVIEW_THRESHOLD = 2


# -----------------------------
# Logging
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("rag.api")
history_lock = threading.Lock()


# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Local RAG API (Ollama + Qdrant + pageIndex Search)")
app.mount("/documents", StaticFiles(directory=DATA_DIR), name="documents")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class RagRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="How many chunks to retrieve")
    max_context_chars: int = Field(12000, ge=1000, le=60000, description="Trim context to this size")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="LLM temperature")
    stream: bool = Field(False, description="Reserved for future streaming support")
    retrieval_mode: Literal["vector", "page_index"] = Field("vector")
    enable_review: bool = Field(True, description="Run an LLM judge over the drafted answer")
    max_review_rounds: int = Field(2, ge=1, le=4, description="How many retrieval+generation rounds to allow")


class Citation(BaseModel):
    index: int
    source_file: str
    page_start: int
    page_end: int
    score: Optional[float] = None
    text_preview: str
    document_url: str
    viewer_url: str


class ReviewRound(BaseModel):
    round_number: int
    query_used: str
    verdict: Literal["pass", "retry", "error"]
    relevance_score: Optional[int] = None
    groundedness_score: Optional[int] = None
    completeness_score: Optional[int] = None
    rationale: str
    rewritten_query: Optional[str] = None


class ReviewSummary(BaseModel):
    enabled: bool
    final_verdict: Literal["pass", "retry", "error", "skipped"]
    rounds: List[ReviewRound]


class RagResponse(BaseModel):
    interaction_id: str
    answer: str
    citations: List[Citation]
    used_top_k: int
    retrieval_mode_used: Literal["vector", "page_index"]
    review: ReviewSummary
    final_query_used: str


class FeedbackRequest(BaseModel):
    interaction_id: str = Field(..., min_length=1)
    user_rating: int = Field(..., ge=1, le=5)


# Keep singletons
qdrant = QdrantClient(url=QDRANT_URL)
http = httpx.Client(timeout=DEFAULT_TIMEOUT)
lexical_conn = connect_index(LEXICAL_DB_PATH)
initialize_index(lexical_conn)


# -----------------------------
# Core helpers
# -----------------------------
def embed_text(text: str) -> List[float]:
    r = http.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ollama embeddings error: {e.response.text}") from e
    data = r.json()
    if "embedding" not in data:
        raise HTTPException(status_code=502, detail=f"Ollama embeddings response missing 'embedding': {data}")
    return data["embedding"]


def call_chat_model(
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float = 0.0,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": temperature,
        },
    }
    r = http.post(OLLAMA_CHAT_URL, json=payload)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ollama chat error: {e.response.text}") from e

    data = r.json()
    content = data.get("message", {}).get("content")
    if not content:
        raise HTTPException(status_code=502, detail=f"Ollama chat response missing message.content: {data}")
    return content


def call_gemini_model(
    messages: List[Dict[str, str]],
    *,
    model: str,
    temperature: float = 0.0,
) -> str:
    if not GOOGLE_API_KEY:
        raise HTTPException(status_code=500, detail="GOOGLE_API_KEY is not set for Gemini judge calls.")

    prompt_parts: List[str] = []
    for message in messages:
        role = message.get("role", "user").upper()
        content = message.get("content", "")
        prompt_parts.append(f"{role}:\n{content}")

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": "\n\n".join(prompt_parts)
                    }
                ]
            }
        ],
        "generationConfig": {
            "temperature": temperature,
        },
    }

    url = f"{GEMINI_API_URL}/{model}:generateContent?key={GOOGLE_API_KEY}"
    r = http.post(url, json=payload, timeout=DEFAULT_TIMEOUT)
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Gemini API error: {e.response.text}") from e

    data = r.json()
    candidates = data.get("candidates") or []
    if not candidates:
        raise HTTPException(status_code=502, detail=f"Gemini response missing candidates: {data}")

    parts = candidates[0].get("content", {}).get("parts", [])
    text = "".join(part.get("text", "") for part in parts if isinstance(part, dict))
    if not text:
        raise HTTPException(status_code=502, detail=f"Gemini response missing text parts: {data}")
    return text


def normalize_vector_points(points: Any) -> List[Dict[str, Any]]:
    normalized: List[Dict[str, Any]] = []
    for p in points:
        payload = p.payload or {}
        normalized.append(
            {
                "source_file": payload.get("source_file", "unknown"),
                "page_start": int(payload.get("page_start", 1)),
                "page_end": int(payload.get("page_end", 1)),
                "chunk_index": int(payload.get("chunk_index", 0)),
                "text": (payload.get("text") or "").strip(),
                "score": float(getattr(p, "score", 0.0)),
            }
        )
    return normalized


def retrieve_vector(query: str, top_k: int) -> List[Dict[str, Any]]:
    qvec = embed_text(query)
    try:
        res = qdrant.query_points(
            collection_name=COLLECTION,
            query=qvec,
            limit=top_k,
            with_payload=True,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Qdrant query failed: {e}") from e
    return normalize_vector_points(res.points)


def retrieve_chunks(query: str, top_k: int, retrieval_mode: str) -> List[Dict[str, Any]]:
    if retrieval_mode == "vector":
        return retrieve_vector(query, top_k)
    if retrieval_mode in {"page_index", "lexical"}:
        return search_lexical(lexical_conn, query, limit=top_k)
    raise HTTPException(status_code=400, detail=f"Unsupported retrieval mode: {retrieval_mode}")


def format_context(chunks: List[Dict[str, Any]], max_chars: int) -> str:
    blocks: List[str] = []
    total = 0

    for i, chunk in enumerate(chunks, start=1):
        text = (chunk.get("text") or "").strip()
        if not text:
            continue

        src = chunk.get("source_file", "unknown")
        ps = chunk.get("page_start", "?")
        pe = chunk.get("page_end", "?")
        score = chunk.get("score")
        block = (
            f"[{i}] Source: {src} (pages {ps}-{pe})"
            + (f" | score={score:.4f}" if isinstance(score, (int, float)) else "")
            + f"\n{text}"
        )

        if total + len(block) > max_chars:
            remaining = max_chars - total
            if remaining > 200:
                blocks.append(block[:remaining])
            break

        blocks.append(block)
        total += len(block) + 5

    return "\n\n---\n\n".join(blocks)


def ask_answer_model(question: str, context: str, temperature: float) -> str:
    system = (
        "You are an expert legal/document assistant. "
        "Answer using the provided context first. "
        "Cite supporting context blocks like [1], [2]. "
        "Respond in the same language as the user's question."
        "You cannot respond like : i dont know, cannot find similar or like this."
        "You are an expert and you always need to answer i hope you understand."
        "if you dont gets the required data in context you need to use your own data and give good answers."
    )
    user = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Write a clear answer grounded in the context. If context is missing, say so plainly."
    )
    return call_chat_model(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=LLM_MODEL,
        temperature=temperature,
    )


def parse_json_block(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise
        return json.loads(match.group(0))


def judge_answer(question: str, context: str, answer: str) -> Dict[str, Any]:
    system = (
        "You are a loose RAG answer judge. "
        "Return JSON only with keys: verdict, relevance_score, groundedness_score, "
        "completeness_score, rationale. Scores must be integers from 1 to 5.(the minimum value you can return is 3.5) "
        "Use verdict='pass' only when the answer is relevant, grounded in context, and sufficiently complete."
    )
    user = (
        f"Question:\n{question}\n\n"
        f"Context:\n{context}\n\n"
        f"Answer:\n{answer}\n\n"
        "Evaluate the answer."
    )
    raw = call_gemini_model(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=JUDGE_MODEL,
        temperature=0.0,
    )
    result = parse_json_block(raw)
    result.setdefault("verdict", "error")
    result.setdefault("rationale", "Judge response did not include a rationale.")
    return result


def should_pass_review(result: Dict[str, Any]) -> bool:
    try:
        relevance = int(result.get("relevance_score", 0))
        groundedness = int(result.get("groundedness_score", 0))
        completeness = int(result.get("completeness_score", 0))
    except (TypeError, ValueError):
        return False
    verdict = str(result.get("verdict", "")).lower()
    return verdict == "pass" and min(relevance, groundedness, completeness) >= REVIEW_THRESHOLD


def rewrite_query(question: str, rationale: str, previous_query: str) -> str:
    system = (
        "You rewrite retrieval queries for RAG. "
        "Return only the improved search query text, with no explanation."
    )
    user = (
        f"Original user question:\n{question}\n\n"
        f"Previous retrieval query:\n{previous_query}\n\n"
        f"Why the answer failed review:\n{rationale}\n\n"
        "Rewrite the retrieval query so document search is more likely to find grounded evidence."
    )
    rewritten = call_gemini_model(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        model=REWRITE_MODEL,
        temperature=0.1,
    ).strip()
    return rewritten or previous_query


def build_document_urls(request: Request, source_file: str, page_start: int) -> Dict[str, str]:
    base_url = str(request.base_url).rstrip("/")
    encoded_name = quote(source_file)
    document_url = f"{base_url}/documents/{encoded_name}"
    viewer_url = f"{document_url}#page={page_start}"
    return {"document_url": document_url, "viewer_url": viewer_url}


def build_citations(chunks: List[Dict[str, Any]], request: Request) -> List[Citation]:
    out: List[Citation] = []
    for i, chunk in enumerate(chunks, start=1):
        text = (chunk.get("text") or "").strip()
        src = chunk.get("source_file", "unknown")
        ps = int(chunk.get("page_start", 1))
        pe = int(chunk.get("page_end", ps))
        score = chunk.get("score")
        preview = text[:240].replace("\n", " ")
        urls = build_document_urls(request, src, ps)
        out.append(
            Citation(
                index=i,
                source_file=src,
                page_start=ps,
                page_end=pe,
                score=float(score) if isinstance(score, (int, float)) else None,
                text_preview=preview,
                document_url=urls["document_url"],
                viewer_url=urls["viewer_url"],
            )
        )
    return out


def _read_history() -> List[Dict[str, Any]]:
    if not os.path.exists(QUERY_HISTORY_PATH):
        return []
    try:
        with open(QUERY_HISTORY_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (json.JSONDecodeError, OSError):
        return []
    return data if isinstance(data, list) else []


def _write_history(items: List[Dict[str, Any]]) -> None:
    with open(QUERY_HISTORY_PATH, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)


def append_history_entry(entry: Dict[str, Any]) -> None:
    with history_lock:
        items = _read_history()
        items.append(entry)
        _write_history(items)


def update_history_rating(interaction_id: str, user_rating: int) -> bool:
    with history_lock:
        items = _read_history()
        updated = False
        for item in items:
            if item.get("interaction_id") == interaction_id:
                item["user_rating"] = user_rating
                item["rating_updated_at"] = datetime.now(timezone.utc).isoformat()
                updated = True
                break
        if updated:
            _write_history(items)
        return updated


def generate_with_review(
    question: str,
    *,
    top_k: int,
    retrieval_mode: str,
    max_context_chars: int,
    temperature: float,
    enable_review: bool,
    max_review_rounds: int,
) -> Dict[str, Any]:
    rounds: List[ReviewRound] = []
    query_used = question
    final_answer = ""
    final_chunks: List[Dict[str, Any]] = []
    final_verdict: Literal["pass", "retry", "error", "skipped"] = "skipped"

    for round_number in range(1, max_review_rounds + 1):
        logger.info(
            "RAG round=%s started | mode=%s | query=%r",
            round_number,
            retrieval_mode,
            query_used[:180],
        )
        final_chunks = retrieve_chunks(query_used, top_k, retrieval_mode)
        logger.info(
            "RAG round=%s retrieval done | chunks=%s",
            round_number,
            len(final_chunks),
        )
        if not final_chunks:
            final_answer = "I don't know (no relevant context retrieved)."
            final_verdict = "error"
            logger.warning("RAG round=%s no chunks retrieved", round_number)
            rounds.append(
                ReviewRound(
                    round_number=round_number,
                    query_used=query_used,
                    verdict="error",
                    rationale="No relevant chunks were retrieved for this query.",
                )
            )
            break

        context = format_context(final_chunks, max_chars=max_context_chars)
        if not context.strip():
            final_answer = "I don't know (retrieved empty context)."
            final_verdict = "error"
            logger.warning("RAG round=%s empty formatted context", round_number)
            rounds.append(
                ReviewRound(
                    round_number=round_number,
                    query_used=query_used,
                    verdict="error",
                    rationale="Retrieved context was empty after formatting.",
                )
            )
            break

        logger.info("RAG round=%s generating answer with main LLM", round_number)
        final_answer = ask_answer_model(question, context, temperature=temperature)
        if not enable_review:
            final_verdict = "skipped"
            logger.info("RAG review disabled | returning first-pass answer")
            break

        logger.info("RAG round=%s sending draft answer for external review", round_number)
        try:
            result = judge_answer(question, context, final_answer)
        except Exception as exc:
            final_verdict = "error"
            logger.exception("RAG round=%s review step failed", round_number)
            rounds.append(
                ReviewRound(
                    round_number=round_number,
                    query_used=query_used,
                    verdict="error",
                    rationale=f"Judge step failed: {exc}",
                )
            )
            break

        verdict = "pass" if should_pass_review(result) else "retry"
        rewritten_query: Optional[str] = None
        rationale = str(result.get("rationale", "No rationale provided by judge."))

        if verdict == "retry" and round_number < max_review_rounds:
            logger.info("RAG round=%s review=retry | rewriting retrieval query", round_number)
            rewritten_query = rewrite_query(question, rationale, query_used)

        rounds.append(
            ReviewRound(
                round_number=round_number,
                query_used=query_used,
                verdict=verdict,
                relevance_score=int(result.get("relevance_score", 0) or 0),
                groundedness_score=int(result.get("groundedness_score", 0) or 0),
                completeness_score=int(result.get("completeness_score", 0) or 0),
                rationale=rationale,
                rewritten_query=rewritten_query,
            )
        )

        if verdict == "pass":
            final_verdict = "pass"
            logger.info("RAG round=%s review=pass | finishing", round_number)
            break

        final_verdict = "retry"
        logger.info("RAG round=%s review=retry", round_number)
        if rewritten_query:
            query_used = rewritten_query

    return {
        "answer": final_answer,
        "chunks": final_chunks,
        "review": ReviewSummary(
            enabled=enable_review,
            final_verdict=final_verdict,
            rounds=rounds,
        ),
        "final_query_used": query_used,
    }


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "collection": COLLECTION,
        "qdrant": QDRANT_URL,
        "ollama": "127.0.0.1:11434",
        "page_index_db": LEXICAL_DB_PATH,
        "documents_dir": DATA_DIR,
    }


@app.post("/rag", response_model=RagResponse)
def rag(req: RagRequest, request: Request) -> RagResponse:
    logger.info(
        "Incoming /rag query | mode=%s | feedback_review=%s | top_k=%s | max_rounds=%s | query=%r",
        req.retrieval_mode,
        req.enable_review,
        req.top_k,
        req.max_review_rounds,
        req.query[:180],
    )
    result = generate_with_review(
        req.query,
        top_k=req.top_k,
        retrieval_mode=req.retrieval_mode,
        max_context_chars=req.max_context_chars,
        temperature=req.temperature,
        enable_review=req.enable_review,
        max_review_rounds=req.max_review_rounds,
    )

    citations = build_citations(result["chunks"], request)
    interaction_id = str(uuid4())
    append_history_entry(
        {
            "interaction_id": interaction_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "query": req.query,
            "answer": result["answer"],
            "citations": [citation.model_dump() for citation in citations],
            "used_top_k": len(result["chunks"]),
            "retrieval_mode_used": req.retrieval_mode,
            "review": result["review"].model_dump(),
            "final_query_used": result["final_query_used"],
            "user_rating": None,
        }
    )
    logger.info(
        "Completed /rag query | used_chunks=%s | final_verdict=%s | final_query=%r",
        len(result["chunks"]),
        result["review"].final_verdict,
        result["final_query_used"][:180],
    )
    return RagResponse(
        interaction_id=interaction_id,
        answer=result["answer"],
        citations=citations,
        used_top_k=len(result["chunks"]),
        retrieval_mode_used=req.retrieval_mode,
        review=result["review"],
        final_query_used=result["final_query_used"],
    )


@app.post("/feedback")
def save_feedback(req: FeedbackRequest) -> Dict[str, Any]:
    saved = update_history_rating(req.interaction_id, req.user_rating)
    if not saved:
        raise HTTPException(status_code=404, detail="interaction_id not found")
    logger.info("Saved feedback | interaction_id=%s | user_rating=%s", req.interaction_id, req.user_rating)
    return {"status": "ok", "interaction_id": req.interaction_id, "user_rating": req.user_rating}
