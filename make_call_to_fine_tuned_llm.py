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
class RagRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User question")
    top_k: int = Field(DEFAULT_TOP_K, ge=1, le=20, description="How many chunks to retrieve")
    max_context_chars: int = Field(12000, ge=1000, le=60000, description="Trim context to this size")
    temperature: float = Field(0.2, ge=0.0, le=2.0, description="LLM temperature")
    stream: bool = Field(False, description="If True, Ollama streams tokens (this endpoint returns non-streamed output)")


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


# Keep singletons
qdrant = QdrantClient(url=QDRANT_URL)
http = httpx.Client(timeout=DEFAULT_TIMEOUT)


# -----------------------------
# Core functions
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


def retrieve_top_k(query: str, top_k: int):
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

    return res.points


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
    system = (
        "You are an expert legal/document assistant. Use are not limited to the provided context to answer. "
        "If the answer is not present in the context, you need to use your existing memory(do not hallucinate ). "
        "When you use facts, cite sources like [1], [2] based on the context blocks."
        "You are equivalent to an lawyer who has years of practice"
        "You need to answer the question in same language the query/question(for which you need to detect the language) was made, do not change the language of the answer. "
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
    points = retrieve_top_k(req.query, req.top_k)
    for i in points:
        print("THE top matchings are : ",i)
        print("")
        print("")
    print("")
    print("")

    if not points:
        return RagResponse(answer="I don't know (no relevant context retrieved).", citations=[], used_top_k=0)

    context = format_context(points, max_chars=req.max_context_chars)
    if not context.strip():
        return RagResponse(answer="I don't know (retrieved empty context).", citations=build_citations(points), used_top_k=len(points))

    answer = ask_llm(req.query, context, temperature=req.temperature)
    print(f"LLM Answer:\n{answer}\n")  # log answer for debugging
    return RagResponse(
        answer=answer,
        citations=build_citations(points),
        used_top_k=len(points),
    )

# ********************** WITHOUT_FASTAPI *************************
# from __future__ import annotations

# import httpx
# from qdrant_client import QdrantClient


# # ---- Config ----
# OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"
# OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"

# LLM_MODEL = "qwen2.5:7b"
# EMBED_MODEL = "nomic-embed-text"

# QDRANT_URL = "http://127.0.0.1:6333"
# COLLECTION = "btp_docs"

# TOP_K = 8  # change to 3/5/8 as you like


# def embed_text(text: str) -> list[float]:
#     """Create an embedding using local Ollama embedding model."""
#     payload = {"model": EMBED_MODEL, "prompt": text}
#     r = httpx.post(OLLAMA_EMBED_URL, json=payload, timeout=120)
#     r.raise_for_status()
#     return r.json()["embedding"]


# def retrieve_top_k(qdrant: QdrantClient, query: str, top_k: int = TOP_K):
#     """Search Qdrant for top-k similar chunks."""
#     qvec = embed_text(query)
#     res = qdrant.query_points(
#         collection_name=COLLECTION,
#         query=qvec,
#         limit=top_k,
#         with_payload=True,
#     )
#     return res.points


# def build_context(points) -> str:
#     """
#     Format retrieved chunks into a context block.
#     We include citations: filename + page range.
#     """
#     blocks = []
#     for i, p in enumerate(points, start=1):
#         payload = p.payload or {}
#         text = (payload.get("text") or "").strip()
#         src = payload.get("source_file", "unknown")
#         ps = payload.get("page_start", "?")
#         pe = payload.get("page_end", "?")
#         score = getattr(p, "score", None)

#         if not text:
#             continue

#         blocks.append(
#             f"[{i}] Source: {src} (pages {ps}-{pe})"
#             + (f" | score={score:.4f}" if score is not None else "")
#             + f"\n{text}"
#         )

#     return "\n\n---\n\n".join(blocks)


# def ask_llm(question: str, context: str) -> str:
#     """
#     Call Qwen with retrieved context.
#     We instruct it to answer using only the context, and cite sources [1], [2], etc.
#     """
#     system = (
#         "You are a legal/document assistant. Use ONLY the provided context to answer. "
#         "If the answer is not present in the context, say you don't know. "
#         "When you use facts, cite sources like [1], [2] based on the context blocks."
#     )

#     user = (
#         f"Context:\n{context}\n\n"
#         f"Question:\n{question}\n\n"
#         "Answer clearly and include citations like [1], [2]."
#     )

#     payload = {
#         "model": LLM_MODEL,
#         "messages": [
#             {"role": "system", "content": system},
#             {"role": "user", "content": user},
#         ],
#         "stream": False,
#     }

#     r = httpx.post(OLLAMA_CHAT_URL, json=payload, timeout=180)
#     r.raise_for_status()
#     return r.json()["message"]["content"]


# def main():
#     qdrant = QdrantClient(url=QDRANT_URL)

#     # Change this question to whatever you want
#     question = "What does the Indian Constitution say about fundamental rights?"

#     points = retrieve_top_k(qdrant, question, top_k=TOP_K)

#     if not points:
#         print("No relevant chunks found in Qdrant.")
#         return

#     context = build_context(points)

#     print("\n--- Retrieved Context (top-k) ---\n")
#     print(context[:4000])  # preview (avoid dumping huge text)

#     answer = ask_llm(question, context)

#     print("\n\n--- Final Answer (Qwen) ---\n")
#     print(answer)


# if __name__ == "__main__":
#     main()



#  ************************** INITIAL_MODEL ************************************
# import httpx

# OLLAMA_URL = "http://127.0.0.1:11434/api/chat"

# payload = {
#     "model": "qwen2.5:7b",
#     "messages": [
#         {"role": "system", "content": "You are a helpful assistant."},
#         {"role": "user", "content": "Explain TCP vs UDP in simple words."}
#     ],
#     "stream": False
# }

# try:
#     response = httpx.post(OLLAMA_URL, json=payload, timeout=120)
#     response.raise_for_status()

#     data = response.json()
#     print("\nModel Response:\n")
#     print(data["message"]["content"])

# except httpx.RequestError as e:
#     print(f"Request failed: {e}")