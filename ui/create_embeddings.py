# from __future__ import annotations

# import os
# import re
# import glob
# import hashlib
# from dataclasses import dataclass
# from typing import List, Dict, Any, Iterable, Tuple

# import httpx
# from pypdf import PdfReader

# from qdrant_client import QdrantClient
# from qdrant_client.http import models as qm


# # -----------------------------
# # Config
# # -----------------------------
# DATA_DIR = "./data"
# PDF_GLOB = os.path.join(DATA_DIR, "*.pdf")

# OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
# EMBED_MODEL = "nomic-embed-text"

# QDRANT_URL = "http://127.0.0.1:6333"
# COLLECTION = "btp_docs"

# # Chunking tuned for legal PDFs
# CHUNK_MAX_CHARS = 1800
# CHUNK_OVERLAP_CHARS = 250

# # Upsert batching
# UPSERT_BATCH_SIZE = 64


# # -----------------------------
# # Helpers
# # -----------------------------
# def stable_int_id(s: str) -> int:
#     """Stable 64-bit-ish int id from sha1."""
#     h = hashlib.sha1(s.encode("utf-8")).hexdigest()
#     return int(h[:15], 16)


# def clean_text(t: str) -> str:
#     """Normalize whitespace without destroying paragraph boundaries."""
#     t = t.replace("\r\n", "\n").replace("\r", "\n")
#     # collapse spaces/tabs
#     t = re.sub(r"[ \t]+", " ", t)
#     # keep paragraph breaks, reduce too many newlines
#     t = re.sub(r"\n{3,}", "\n\n", t)
#     return t.strip()


# def extract_pdf_pages(pdf_path: str) -> List[str]:
#     """Extract text per page (best-effort)."""
#     reader = PdfReader(pdf_path)
#     pages = []
#     for i, page in enumerate(reader.pages):
#         txt = page.extract_text() or ""
#         txt = clean_text(txt)
#         pages.append(txt)
#     return pages


# def split_into_chunks(text: str, max_chars: int, overlap_chars: int) -> List[str]:
#     """
#     Paragraph-aware chunking with overlap.
#     - Split by paragraphs primarily.
#     - Accumulate paragraphs until max_chars.
#     - Overlap uses tail chars of previous chunk.
#     """
#     if not text:
#         return []

#     paras = [p.strip() for p in text.split("\n\n") if p.strip()]
#     chunks: List[str] = []
#     buf: List[str] = []
#     buf_len = 0

#     def flush():
#         nonlocal buf, buf_len
#         if not buf:
#             return
#         chunk = "\n\n".join(buf).strip()
#         if chunk:
#             chunks.append(chunk)
#         buf, buf_len = [], 0

#     for p in paras:
#         # If a single paragraph is enormous, hard-split it.
#         if len(p) > max_chars:
#             flush()
#             start = 0
#             while start < len(p):
#                 end = min(start + max_chars, len(p))
#                 part = p[start:end].strip()
#                 if part:
#                     chunks.append(part)
#                 start = end - overlap_chars if end < len(p) else end
#             continue

#         # Normal accumulation
#         add_len = len(p) + (2 if buf else 0)  # account for "\n\n"
#         if buf_len + add_len <= max_chars:
#             buf.append(p)
#             buf_len += add_len
#         else:
#             flush()
#             buf.append(p)
#             buf_len = len(p)

#     flush()

#     # Apply overlap between produced chunks (character tail overlap)
#     if overlap_chars > 0 and len(chunks) > 1:
#         overlapped: List[str] = [chunks[0]]
#         for i in range(1, len(chunks)):
#             prev = overlapped[-1]
#             tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
#             merged = (tail + "\n\n" + chunks[i]).strip()
#             # Keep merged from growing too large
#             if len(merged) > max_chars + overlap_chars:
#                 merged = merged[-(max_chars + overlap_chars):]
#             overlapped.append(merged)
#         chunks = overlapped

#     return chunks


# def ollama_embed(client: httpx.Client, text: str) -> List[float]:
#     payload = {"model": EMBED_MODEL, "prompt": text}
#     r = client.post(OLLAMA_EMBED_URL, json=payload)
#     r.raise_for_status()
#     return r.json()["embedding"]


# def ensure_collection(qdrant: QdrantClient, vector_size: int) -> None:
#     existing = {c.name for c in qdrant.get_collections().collections}
#     if COLLECTION in existing:
#         return
#     qdrant.create_collection(
#         collection_name=COLLECTION,
#         vectors_config=qm.VectorParams(
#             size=vector_size,
#             distance=qm.Distance.COSINE,
#         ),
#     )


# @dataclass
# class ChunkRecord:
#     source_file: str
#     page_start: int
#     page_end: int
#     chunk_index: int
#     text: str


# def build_chunks_for_pdf(pdf_path: str) -> List[ChunkRecord]:
#     pages = extract_pdf_pages(pdf_path)
#     fname = os.path.basename(pdf_path)

#     records: List[ChunkRecord] = []
#     chunk_idx = 0

#     # Page-aware: chunk each page, but allow merging small pages into a single chunk block
#     # to reduce tiny chunks (common in scanned-ish or sparse pages).
#     buffer_text = ""
#     buffer_page_start = 1
#     buffer_page_end = 1

#     def flush_buffer():
#         nonlocal buffer_text, buffer_page_start, buffer_page_end, chunk_idx
#         if not buffer_text.strip():
#             buffer_text = ""
#             return
#         chunks = split_into_chunks(buffer_text, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
#         for c in chunks:
#             records.append(
#                 ChunkRecord(
#                     source_file=fname,
#                     page_start=buffer_page_start,
#                     page_end=buffer_page_end,
#                     chunk_index=chunk_idx,
#                     text=c,
#                 )
#             )
#             chunk_idx += 1
#         buffer_text = ""

#     for i, page_text in enumerate(pages, start=1):
#         if not page_text:
#             continue

#         # If buffer empty, start range
#         if not buffer_text:
#             buffer_page_start = i
#             buffer_page_end = i
#             buffer_text = page_text
#         else:
#             # Try to merge next page if it still stays reasonably sized
#             tentative = buffer_text + "\n\n" + page_text
#             if len(tentative) <= CHUNK_MAX_CHARS * 2:
#                 buffer_text = tentative
#                 buffer_page_end = i
#             else:
#                 flush_buffer()
#                 buffer_page_start = i
#                 buffer_page_end = i
#                 buffer_text = page_text

#     flush_buffer()
#     return records


# def upsert_records(qdrant: QdrantClient, records: List[ChunkRecord]) -> None:
#     if not records:
#         return

#     with httpx.Client(timeout=180) as http:
#         # Create collection using first embedding to get dims
#         first_vec = ollama_embed(http, records[0].text)
#         ensure_collection(qdrant, vector_size=len(first_vec))

#         points: List[qm.PointStruct] = []

#         def push_batch():
#             nonlocal points
#             if points:
#                 qdrant.upsert(collection_name=COLLECTION, points=points)
#                 points = []

#         # add first point
#         rid = stable_int_id(f"{records[0].source_file}|{records[0].page_start}-{records[0].page_end}|{records[0].chunk_index}")
#         points.append(
#             qm.PointStruct(
#                 id=rid,
#                 vector=first_vec,
#                 payload={
#                     "text": records[0].text,
#                     "source_file": records[0].source_file,
#                     "page_start": records[0].page_start,
#                     "page_end": records[0].page_end,
#                     "chunk_index": records[0].chunk_index,
#                 },
#             )
#         )

#         for rec in records[1:]:
#             vec = ollama_embed(http, rec.text)
#             rid = stable_int_id(f"{rec.source_file}|{rec.page_start}-{rec.page_end}|{rec.chunk_index}")
#             points.append(
#                 qm.PointStruct(
#                     id=rid,
#                     vector=vec,
#                     payload={
#                         "text": rec.text,
#                         "source_file": rec.source_file,
#                         "page_start": rec.page_start,
#                         "page_end": rec.page_end,
#                         "chunk_index": rec.chunk_index,
#                     },
#                 )
#             )

#             if len(points) >= UPSERT_BATCH_SIZE:
#                 push_batch()

#         push_batch()


# def main():
#     pdfs = sorted(glob.glob(PDF_GLOB))
#     if not pdfs:
#         print(f"❌ No PDFs found in {DATA_DIR}")
#         return

#     qdrant = QdrantClient(url=QDRANT_URL)

#     total_chunks = 0
#     for pdf in pdfs:
#         print(f"\n📄 Processing: {pdf}")
#         records = build_chunks_for_pdf(pdf)
#         print(f"   → extracted chunks: {len(records)}")
#         upsert_records(qdrant, records)
#         total_chunks += len(records)

#     print(f"\n✅ Done. Total chunks upserted into '{COLLECTION}': {total_chunks}")
#     print("   Qdrant UI: http://localhost:6333/dashboard")


# if __name__ == "__main__":
#     main()









# *****************************
from __future__ import annotations

import os
import re
import glob
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Tuple

import httpx
from pypdf import PdfReader

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# -----------------------------
# Config
# -----------------------------
DATA_DIR = "./data"
PDF_GLOB = os.path.join(DATA_DIR, "*.pdf")

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embed"
OLLAMA_EMBED_LEGACY_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "btp_docs_new"

# Make chunks smaller to avoid context-length errors
TARGET_TOKENS = 250          # was 500
OVERLAP_TOKENS = 60          # was 120

# Batch sizes
EMBED_BATCH_SIZE = 8
UPSERT_BATCH_SIZE = 64

MIN_CHUNK_CHARS = 200

# VERY IMPORTANT: strict limit (word-based) to guarantee safe embedding
MAX_EMBED_TOKENS = 350       # hard cap before embedding


# -----------------------------
# Hashing & IDs
# -----------------------------
def sha1_hex(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()


def stable_int_id(s: str) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:15], 16)


# -----------------------------
# Text cleanup
# -----------------------------
def normalize_whitespace(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def fix_hyphenation(t: str) -> str:
    return re.sub(r"(\w)-\n(\w)", r"\1\2", t)


def extract_pdf_pages_raw(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    pages: List[str] = []
    for page in reader.pages:
        pages.append((page.extract_text() or "").strip())
    return pages


def detect_repeated_lines(pages: List[str], min_len: int = 8) -> Tuple[set, set]:
    if len(pages) < 3:
        return set(), set()

    top_counts: Dict[str, int] = {}
    bot_counts: Dict[str, int] = {}

    for p in pages:
        lines = [ln.strip() for ln in p.splitlines() if ln.strip()]
        if not lines:
            continue
        for ln in lines[:3]:
            if len(ln) >= min_len:
                top_counts[ln] = top_counts.get(ln, 0) + 1
        for ln in lines[-3:]:
            if len(ln) >= min_len:
                bot_counts[ln] = bot_counts.get(ln, 0) + 1

    threshold = max(2, int(0.6 * len(pages)))
    headers = {ln for ln, c in top_counts.items() if c >= threshold}
    footers = {ln for ln, c in bot_counts.items() if c >= threshold}
    return headers, footers


def clean_page_text(page_text: str, header_lines: set, footer_lines: set) -> str:
    lines = [ln.rstrip() for ln in page_text.splitlines()]
    out: List[str] = []
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append("")
            continue
        if s in header_lines or s in footer_lines:
            continue
        if re.fullmatch(r"\d{1,4}", s):
            continue
        out.append(s)

    t = "\n".join(out)
    t = fix_hyphenation(t)
    t = normalize_whitespace(t)
    return t


def approx_tokens(text: str) -> int:
    return len(text.split())


# -----------------------------
# Structure blocks + chunking
# -----------------------------
def split_into_struct_blocks(text: str) -> List[str]:
    if not text:
        return []

    heading_pat = re.compile(
        r"(?im)^(?:\s*)("
        r"(?:CHAPTER|Chapter)\s+[IVXLC0-9]+.*|"
        r"(?:SECTION|Section)\s+\d+[A-Z]?\b.*|"
        r"(?:ARTICLE|Article)\s+\d+\b.*|"
        r"(?:RULE|Rule)\s+\d+\b.*|"
        r"§\s*\d+.*|"
        r"(?:SCHEDULE|Schedule)\b.*"
        r")\s*$"
    )

    lines = text.splitlines()
    blocks: List[str] = []
    buf: List[str] = []

    def flush():
        nonlocal buf
        if buf:
            b = "\n".join(buf).strip()
            if b:
                blocks.append(b)
        buf = []

    for ln in lines:
        if heading_pat.match(ln.strip()):
            flush()
            buf.append(ln.strip())
        else:
            buf.append(ln)
    flush()

    return blocks


def hard_split_to_max_tokens(text: str, max_tokens: int) -> List[str]:
    """
    GUARANTEE: return pieces each <= max_tokens (word-based).
    """
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    parts = []
    i = 0
    while i < len(words):
        part_words = words[i:i + max_tokens]
        parts.append(" ".join(part_words))
        i += max_tokens
    return parts


def chunk_blocks_token_aware(blocks: List[str], target_tokens: int, overlap_tokens: int) -> List[str]:
    chunks: List[str] = []
    buf: List[str] = []
    buf_tokens = 0

    def flush():
        nonlocal buf, buf_tokens
        if not buf:
            return
        c = "\n\n".join(buf).strip()
        if c:
            chunks.append(c)
        buf = []
        buf_tokens = 0

    for b in blocks:
        bt = approx_tokens(b)

        if bt > target_tokens * 2:
            flush()
            # split huge block by max tokens directly
            for piece in hard_split_to_max_tokens(b, target_tokens):
                if len(piece) >= MIN_CHUNK_CHARS:
                    chunks.append(piece)
            continue

        if buf_tokens + bt <= target_tokens:
            buf.append(b)
            buf_tokens += bt
        else:
            flush()
            buf.append(b)
            buf_tokens = bt

    flush()

    # overlap
    if overlap_tokens > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = out[-1].split()
            tail = prev[-overlap_tokens:] if len(prev) > overlap_tokens else prev
            out.append((" ".join(tail) + "\n\n" + chunks[i]).strip())
        chunks = out

    # final safety: hard split any chunk > MAX_EMBED_TOKENS
    safe_chunks: List[str] = []
    for c in chunks:
        if approx_tokens(c) > MAX_EMBED_TOKENS:
            safe_chunks.extend(hard_split_to_max_tokens(c, MAX_EMBED_TOKENS))
        else:
            safe_chunks.append(c)

    return [c for c in safe_chunks if len(c) >= MIN_CHUNK_CHARS]

def split_text_mid(text: str) -> tuple[str, str]:
    """
    Split roughly in the middle on a paragraph boundary if possible,
    else on a sentence boundary, else hard split.
    """
    t = text.strip()
    if len(t) < 2:
        return t, ""

    mid = len(t) // 2

    # try paragraph boundary near mid
    for sep in ["\n\n", "\n", ". "]:
        window = 800
        left = t[max(0, mid - window): mid + window]
        idx = left.find(sep)
        if idx != -1:
            cut = max(1, max(0, mid - window) + idx + len(sep))
            return t[:cut].strip(), t[cut:].strip()

    # hard split
    return t[:mid].strip(), t[mid:].strip()


def is_ctx_error(err_text: str) -> bool:
    return "exceeds the context length" in (err_text or "").lower()

MAX_EMBED_CHARS = 8000  # hard safety clamp (not token-based, but prevents huge payloads)


def clamp_chars(text: str, max_chars: int = MAX_EMBED_CHARS) -> str:
    text = text.strip()
    if len(text) <= max_chars:
        return text
    # keep head + tail (better than only head for legal docs)
    half = max_chars // 2
    return (text[:half] + "\n...\n" + text[-half:]).strip()


# -----------------------------
# Ollama embeddings (robust)
# -----------------------------
def ollama_embed_one_legacy(http: httpx.Client, text: str) -> List[float]:
    """
    Legacy single embedding:
      POST /api/embeddings {model, prompt}
    """
    text = clamp_chars(text)

    r = http.post(OLLAMA_EMBED_LEGACY_URL, json={"model": EMBED_MODEL, "prompt": text})
    if r.status_code == 200:
        return r.json()["embedding"]

    # IMPORTANT: show real body if it's not context error
    body = r.text
    if r.status_code >= 400 and not is_ctx_error(body):
        raise RuntimeError(f"Ollama /api/embeddings error {r.status_code}: {body}")

    # If context-length error, split text and average embeddings (or store both)
    # We'll do: embed both halves and average => single vector for the chunk.
    left, right = split_text_mid(text)
    if not right:  # cannot split anymore
        raise RuntimeError(f"Ollama /api/embeddings still too long even after split: {body}")

    v1 = ollama_embed_one_legacy(http, left)
    v2 = ollama_embed_one_legacy(http, right)

    # average the vectors to keep a single vector per chunk (simplest)
    return [(a + b) / 2.0 for a, b in zip(v1, v2)]

def ollama_embed_batch(http: httpx.Client, texts: List[str]) -> List[List[float]]:
    """
    Robust batch embed:
    1) Try /api/embed with list input.
    2) If fails, split the batch.
    3) Base case: embed one text via legacy with recursive splitting on ctx error.
    """
    texts = [clamp_chars(t) for t in texts]

    # Try /api/embed batch first
    try:
        r = http.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "input": texts})
        if r.status_code == 200:
            data = r.json()
            if "embeddings" in data:
                return data["embeddings"]
            # unexpected shape -> fallback
        # if 400/404 -> fallback
    except httpx.HTTPError:
        pass

    # Fallback: split batch until size 1
    if len(texts) > 1:
        mid = len(texts) // 2
        return ollama_embed_batch(http, texts[:mid]) + ollama_embed_batch(http, texts[mid:])

    # Single text: legacy embedding (handles ctx error by splitting)
    return [ollama_embed_one_legacy(http, texts[0])]


# -----------------------------
# Qdrant
# -----------------------------
def ensure_collection(qdrant: QdrantClient, vector_size: int) -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION in existing:
        return
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=vector_size, distance=qm.Distance.COSINE),
    )


def already_indexed(qdrant: QdrantClient, chunk_sha1: str) -> bool:
    res, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=qm.Filter(
            must=[qm.FieldCondition(key="chunk_sha1", match=qm.MatchValue(value=chunk_sha1))]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(res) > 0


@dataclass
class ChunkRecord:
    source_file: str
    page_start: int
    page_end: int
    chunk_index: int
    text: str
    doc_sha1: str
    chunk_sha1: str


# -----------------------------
# Build chunks
# -----------------------------
def build_chunks_for_pdf(pdf_path: str) -> List[ChunkRecord]:
    raw_pages = extract_pdf_pages_raw(pdf_path)
    header_lines, footer_lines = detect_repeated_lines(raw_pages)

    fname = os.path.basename(pdf_path)
    doc_fingerprint = sha1_hex("\n\n".join(raw_pages))

    cleaned_pages = [clean_page_text(p, header_lines, footer_lines) for p in raw_pages]

    # merge pages into a doc text
    doc_text = "\n\n".join([p for p in cleaned_pages if p.strip()])
    blocks = split_into_struct_blocks(doc_text)
    chunks = chunk_blocks_token_aware(blocks, TARGET_TOKENS, OVERLAP_TOKENS)

    records: List[ChunkRecord] = []
    for idx, c in enumerate(chunks):
        records.append(
            ChunkRecord(
                source_file=fname,
                page_start=1,
                page_end=len(cleaned_pages),
                chunk_index=idx,
                text=c,
                doc_sha1=doc_fingerprint,
                chunk_sha1=sha1_hex(c),
            )
        )
    return records


# -----------------------------
# Upsert
# -----------------------------
def upsert_records(qdrant: QdrantClient, records: List[ChunkRecord]) -> None:
    if not records:
        return

    with httpx.Client(timeout=180) as http:
        first_vec = ollama_embed_batch(http, [records[0].text])[0]
        ensure_collection(qdrant, vector_size=len(first_vec))

        points: List[qm.PointStruct] = []

        def push():
            nonlocal points
            if points:
                qdrant.upsert(collection_name=COLLECTION, points=points)
                points = []

        batch_texts: List[str] = []
        batch_recs: List[ChunkRecord] = []

        def flush_embed_batch():
            nonlocal batch_texts, batch_recs, points
            if not batch_recs:
                return
            vecs = ollama_embed_batch(http, batch_texts)

            for rec, vec in zip(batch_recs, vecs):
                pid = stable_int_id(rec.chunk_sha1)
                points.append(
                    qm.PointStruct(
                        id=pid,
                        vector=vec,
                        payload={
                            "text": rec.text,
                            "source_file": rec.source_file,
                            "page_start": rec.page_start,
                            "page_end": rec.page_end,
                            "chunk_index": rec.chunk_index,
                            "doc_sha1": rec.doc_sha1,
                            "chunk_sha1": rec.chunk_sha1,
                        },
                    )
                )
                if len(points) >= UPSERT_BATCH_SIZE:
                    push()

            batch_texts = []
            batch_recs = []

        for rec in records:
            if already_indexed(qdrant, rec.chunk_sha1):
                continue

            batch_recs.append(rec)
            batch_texts.append(rec.text)

            if len(batch_recs) >= EMBED_BATCH_SIZE:
                flush_embed_batch()

        flush_embed_batch()
        push()


def main():
    pdfs = sorted(glob.glob(PDF_GLOB))
    if not pdfs:
        print(f"❌ No PDFs found in {DATA_DIR}")
        return

    qdrant = QdrantClient(url=QDRANT_URL)

    for pdf in pdfs:
        print(f"\n📄 Processing: {pdf}")
        records = build_chunks_for_pdf(pdf)
        print(f"   → extracted chunks: {len(records)} (pre-skip)")
        upsert_records(qdrant, records)
        print(f"   → done: {os.path.basename(pdf)}")

    print(f"\n✅ Done. Qdrant UI: http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()