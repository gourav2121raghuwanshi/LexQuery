"""
FastAPI RAG server:
- Embeds query via Ollama (nomic-embed-text)
- Retrieves top-k from Qdrant
- Sends context + query to Ollama chat (qwen2.5:7b)
- Returns answer + citations

http://127.0.0.1:8000/docs
http://127.0.0.1:8000/health
Prereqs:
  pip install fastapi uvicorn httpx qdrant-client

Run:
  uvicorn make_call_to_fine_tuned_llm:app --host 127.0.0.1 --port 8000 --reload

Test:
  curl -X POST http://127.0.0.1:8000/rag \
    -H "Content-Type: application/json" \
    -d '{"query":"What are Fundamental Rights in the Indian Constitution?","top_k":5}'
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
import json, re
from fastapi.middleware.cors import CORSMiddleware

# -----------------------------
# Config
# -----------------------------
OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"

LLM_MODEL = "qwen2.5:7b"
EMBED_MODEL = "nomic-embed-text"

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "btp_docs"

DEFAULT_TOP_K = 5
DEFAULT_TIMEOUT = 250



# -----------------------------
# FastAPI app
# -----------------------------
app = FastAPI(title="Local RAG API (Ollama + Qdrant)")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# class RagRequest(BaseModel):
#     query: str = Field(..., min_length=1, description="User question")
#     top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="How many chunks to retrieve")
#     max_context_chars: int = Field(12000, ge=1000, le=60000, description="Trim context to this size")
#     temperature: float = Field(0.2, ge=0.0, le=2.0, description="LLM temperature")
#     stream: bool = Field(False, description="If True, Ollama streams tokens (this endpoint returns non-streamed output)")


class Citation(BaseModel):
    index: int
    source_file: str
    page_start: Any
    page_end: Any
    score: Optional[float] = None
    text_preview: str


class RagResponse(BaseModel):
    answer: str
    citations: List[Citation]
    used_top_k: int

class ChatTurn(BaseModel):
    role: str = Field(..., pattern="^(user|assistant)$")
    content: str = Field(..., min_length=1, max_length=4000)

class RagRequest(BaseModel):
    query: str = Field(..., min_length=1)
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20)
    max_context_chars: int = Field(12000, ge=1000, le=60000)
    temperature: float = Field(0.2, ge=0.0, le=2.0)
    stream: bool = False
    history: List[ChatTurn] = Field(default_factory=list, description="Last N chat turns")

# Keep singletons
qdrant = QdrantClient(url=QDRANT_URL)
http = httpx.Client(timeout=DEFAULT_TIMEOUT)



OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
OLLAMA_EMBED_LEGACY_URL = "http://127.0.0.1:11434/api/embeddings"
CANDIDATE_K = 40
RERANK_MODEL = "bge-reranker-large"   # or "bge-reranker-v2-m3" if you have it
OLLAMA_RERANK_URL = "http://127.0.0.1:11434/api/chat"  # you can also rerank with LLM, but best is a real reranker



def parse_json_object(s: str) -> Optional[Dict[str, Any]]:
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()

    # direct parse
    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    # fallback: extract first {...} block
    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            return None
    return None


def embed_text(text: str) -> List[float]:
    text = text.strip()
    if len(text) > 5000:
        text = text[:2500] + "\n...\n" + text[-2500:]

    # try /api/embed
    r = http.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": text})
    if r.status_code == 200:
        data = r.json()
        return data["embeddings"][0]

    # fallback /api/embeddings
    r = http.post(OLLAMA_EMBED_LEGACY_URL, json={"model": EMBED_MODEL, "prompt": text})
    try:
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        raise HTTPException(status_code=502, detail=f"Ollama embeddings error: {e.response.text}") from e
    return r.json()["embedding"]

def rewrite_query(query: str, failure_reasons: List[str]) -> str:
    prompt = f"""
Rewrite the query to improve retrieval from the provided legal documents.
Make it more specific. Add jurisdiction if missing (India / Indian Constitution).
Return ONLY the rewritten query.

Query: {query}
Failure reasons: {failure_reasons}
"""
    r = http.post(OLLAMA_CHAT_URL, json={
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2}
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()

def make_standalone_query(query: str, history: List[Dict[str, str]]) -> str:
    if not history:
        return query

    # keep only last 6 messages max
    history = history[-6:]

    convo = "\n".join([f"{h['role'].upper()}: {h['content']}" for h in history])

    prompt = f"""
Rewrite the user's last question into a standalone legal query.
Use conversation context if needed (India / Indian law / Constitution).
Return ONLY the rewritten query.

Conversation:
{convo}

User question: {query}
"""
    r = http.post(OLLAMA_CHAT_URL, json={
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.0}
    })
    r.raise_for_status()
    return r.json()["message"]["content"].strip()
def evaluate_answer(query: str, context: str, answer: str) -> Dict[str, Any]:
    # Speed trick: shrink context for the judge (same evaluation strategy, fewer tokens)
    # Keep only first few blocks and limit each block length.
    MAX_BLOCKS = 6
    MAX_CHARS_PER_BLOCK = 500

    blocks = context.split("\n\n---\n\n")
    blocks = blocks[:MAX_BLOCKS]
    compact_blocks = []
    for b in blocks:
        b = b.strip()
        if not b:
            continue
        # preserve citation header like "[1] Source: ..."
        if "\n" in b:
            header, body = b.split("\n", 1)
            body = body.strip()[:MAX_CHARS_PER_BLOCK]
            compact_blocks.append(f"{header}\n{body}")
        else:
            compact_blocks.append(b[:MAX_CHARS_PER_BLOCK])

    compact_context = "\n\n---\n\n".join(compact_blocks)

    prompt = f"""
Return ONLY valid JSON with:
answer_relevance (0-1),
context_relevance (0-1),
groundedness (0-1),
failure_reasons (array of strings),
verdict ("PASS" or "FAIL").

Query: {query}

Evidence (top retrieved excerpts):
{compact_context}

Answer:
{answer}

Rules:
- Groundedness is low if any claim lacks support in Evidence.
- FAIL if groundedness < 0.7 OR answer_relevance < 0.7.
"""

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {
            "temperature": 0.0,
            # extra speed knobs (optional): reduce sampling overhead
            "num_predict": 200,   # judge output is short JSON; cap generation
        },
    }

    r = http.post(OLLAMA_CHAT_URL, json=payload)
    r.raise_for_status()

    raw = r.json().get("message", {}).get("content", "")
    obj = parse_json_object(raw)

    if not obj:
        return {
            "answer_relevance": 0.0,
            "context_relevance": 0.0,
            "groundedness": 0.0,
            "failure_reasons": ["Evaluator did not return valid JSON"],
            "verdict": "FAIL",
            "raw": raw[:400],
        }
    return obj



def parse_rank_list(s: str) -> Optional[List[int]]:
    if not s:
        return None
    s = s.strip()
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.IGNORECASE).strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, list):
            return [int(x) for x in obj]
        if isinstance(obj, dict):
            for k in ("ranked", "indices", "order"):
                if k in obj and isinstance(obj[k], list):
                    return [int(x) for x in obj[k]]
    except Exception:
        pass

    m = re.search(r"\[[\s\d,]+\]", s)
    if m:
        try:
            return [int(x) for x in json.loads(m.group(0))]
        except Exception:
            return None
    return None

def rerank_with_llm(query: str, points, keep_k: int) -> List[Any]:
    items = []
    for i, p in enumerate(points):
        txt = (p.payload or {}).get("text", "")
        items.append({"i": i, "text": txt[:800]})

    prompt = (
        "You are a reranker. Given a query and passages, return JSON list of indices "
        "sorted by relevance high to low. Only output JSON.\n\n"
        f"Query: {query}\n\n"
        f"Passages: {items}\n"
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "options": {"temperature": 0.2},
    }

    r = http.post(OLLAMA_CHAT_URL, json=payload)
    r.raise_for_status()

    order = r.json()["message"]["content"]

    ranked = parse_rank_list(order)
    if not ranked:
        return points[:keep_k]

    reranked = [points[i] for i in ranked if 0 <= i < len(points)]
    return reranked[:keep_k]

def retrieve_top_k(query: str, top_k: int):
    qvec = embed_text(query)

    res = qdrant.query_points(
        collection_name=COLLECTION,
        query=qvec,
        limit=CANDIDATE_K,       # retrieve more
        with_payload=True,
    )
    points = res.points or []
    points = rerank_with_llm(query, points, keep_k=top_k)
    return points


def format_context(points, max_chars: int) -> str:
    blocks: List[str] = []
    total = 0

    for i, p in enumerate(points, start=1):
        payload = p.payload or {}
        text = (payload.get("text") or "").strip()
        if not text:
            continue

        src = payload.get("source_file", "unknown")
        ps = payload.get("page_start", "?")
        pe = payload.get("page_end", "?")
        score = getattr(p, "score", None)

        block = (
            f"[{i}] Source: {src} (pages {ps}-{pe})"
            + (f" | score={score:.4f}" if score is not None else "")
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


def ask_llm(question: str, context: str, temperature: float) -> str:
    # system = (
    #     "You are an expert legal/document assistant. Use are not limited to the provided context to answer. "
    #     "If the answer is not present in the context, you need to use your existing memory(do not hallucinate ). "
    #     "When you use facts, cite sources like [1], [2] based on the context blocks."
    #     "You are equivalent to an lawyer who has years of practice"
    #     "Always try to use the outside knowledge or case studies you have in your memory to answer the question, do not rely solely on the provided context. "
    # )
    system = (
  "You are a legal/document assistant.\n"
  "You MUST answer using ONLY the provided context.\n"
  "If the answer is not present in the context, reply exactly: "
  "'Not found in the provided documents.'\n"
  "Every factual claim MUST have citations like [1], [2].\n"
  "Do NOT use outside knowledge.\n"
)
    user = (
        f"Context:\n{context}\n\n"
        f"Question:\n{question}\n\n"
        "Answer clearly and include citations like [1], [2]."
    )

    payload = {
        "model": LLM_MODEL,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
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
    msg = data.get("message", {})
    content = msg.get("content")
    if not content:
        raise HTTPException(status_code=502, detail=f"Ollama chat response missing message.content: {data}")
    return content


def build_citations(points) -> List[Citation]:
    out: List[Citation] = []
    for i, p in enumerate(points, start=1):
        payload = p.payload or {}
        text = (payload.get("text") or "").strip()
        src = payload.get("source_file", "unknown")
        ps = payload.get("page_start", "?")
        pe = payload.get("page_end", "?")
        score = getattr(p, "score", None)

        preview = text[:240].replace("\n", " ")
        out.append(
            Citation(
                index=i,
                source_file=src,
                page_start=ps,
                page_end=pe,
                score=score,
                text_preview=preview,
            )
        )
    return out


# -----------------------------
# Endpoints
# -----------------------------
@app.get("/health")
def health():
    # quick checks (lightweight)
    return {"status": "ok", "collection": COLLECTION, "qdrant": QDRANT_URL, "ollama": "127.0.0.1:11434"}


@app.post("/rag", response_model=RagResponse)
def rag(req: RagRequest):
    max_retries = 2
    hist = [{"role": t.role, "content": t.content} for t in req.history]
    query = make_standalone_query(req.query, hist) if hist else req.query
    query = query.strip() or req.query
    best = None

    for attempt in range(max_retries + 1):
        points = retrieve_top_k(query, req.top_k)
        # print(f"Attempt {attempt+1} retrieved {points} points")
        for i in points:
            print(i)
            print("----")
            print("----")
            print("----")
            print("----")
        if not points:
            break
        context = format_context(points, req.max_context_chars)
        answer = ask_llm(query, context, req.temperature)
        print(f"Attempt {attempt+1} answer:\n{answer}\n")
        ev = evaluate_answer(query, context, answer)
        best = (answer, points, ev)

        if ev.get("verdict") == "PASS":
            return RagResponse(answer=answer, citations=build_citations(points), used_top_k=len(points))

        query = rewrite_query(query, ev.get("failure_reasons", []))

    # fallback
    if best:
        answer, points, ev = best
        return RagResponse(answer=answer, citations=build_citations(points), used_top_k=len(points))
    return RagResponse(answer="Not found in the provided documents.", citations=[], used_top_k=0)
 