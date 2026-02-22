from __future__ import annotations

import os
import re
import glob
import hashlib
from dataclasses import dataclass
from typing import List, Dict, Any, Iterable, Tuple

import httpx
from pypdf import PdfReader

from qdrant_client import QdrantClient
from qdrant_client.http import models as qm


# -----------------------------
# Config
# -----------------------------
DATA_DIR = "./data"
PDF_GLOB = os.path.join(DATA_DIR, "*.pdf")

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "btp_docs"

# Chunking tuned for legal PDFs
CHUNK_MAX_CHARS = 1800
CHUNK_OVERLAP_CHARS = 250

# Upsert batching
UPSERT_BATCH_SIZE = 64


# -----------------------------
# Helpers
# -----------------------------
def stable_int_id(s: str) -> int:
    """Stable 64-bit-ish int id from sha1."""
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:15], 16)


def clean_text(t: str) -> str:
    """Normalize whitespace without destroying paragraph boundaries."""
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    # collapse spaces/tabs
    t = re.sub(r"[ \t]+", " ", t)
    # keep paragraph breaks, reduce too many newlines
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def extract_pdf_pages(pdf_path: str) -> List[str]:
    """Extract text per page (best-effort)."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        txt = page.extract_text() or ""
        txt = clean_text(txt)
        pages.append(txt)
    return pages


def split_into_chunks(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Paragraph-aware chunking with overlap.
    - Split by paragraphs primarily.
    - Accumulate paragraphs until max_chars.
    - Overlap uses tail chars of previous chunk.
    """
    if not text:
        return []

    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: List[str] = []
    buf: List[str] = []
    buf_len = 0

    def flush():
        nonlocal buf, buf_len
        if not buf:
            return
        chunk = "\n\n".join(buf).strip()
        if chunk:
            chunks.append(chunk)
        buf, buf_len = [], 0

    for p in paras:
        # If a single paragraph is enormous, hard-split it.
        if len(p) > max_chars:
            flush()
            start = 0
            while start < len(p):
                end = min(start + max_chars, len(p))
                part = p[start:end].strip()
                if part:
                    chunks.append(part)
                start = end - overlap_chars if end < len(p) else end
            continue

        # Normal accumulation
        add_len = len(p) + (2 if buf else 0)  # account for "\n\n"
        if buf_len + add_len <= max_chars:
            buf.append(p)
            buf_len += add_len
        else:
            flush()
            buf.append(p)
            buf_len = len(p)

    flush()

    # Apply overlap between produced chunks (character tail overlap)
    if overlap_chars > 0 and len(chunks) > 1:
        overlapped: List[str] = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = overlapped[-1]
            tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            merged = (tail + "\n\n" + chunks[i]).strip()
            # Keep merged from growing too large
            if len(merged) > max_chars + overlap_chars:
                merged = merged[-(max_chars + overlap_chars):]
            overlapped.append(merged)
        chunks = overlapped

    return chunks


def ollama_embed(client: httpx.Client, text: str) -> List[float]:
    payload = {"model": EMBED_MODEL, "prompt": text}
    r = client.post(OLLAMA_EMBED_URL, json=payload)
    r.raise_for_status()
    return r.json()["embedding"]


def ensure_collection(qdrant: QdrantClient, vector_size: int) -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION in existing:
        return
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(
            size=vector_size,
            distance=qm.Distance.COSINE,
        ),
    )


@dataclass
class ChunkRecord:
    source_file: str
    page_start: int
    page_end: int
    chunk_index: int
    text: str


def build_chunks_for_pdf(pdf_path: str) -> List[ChunkRecord]:
    pages = extract_pdf_pages(pdf_path)
    fname = os.path.basename(pdf_path)

    records: List[ChunkRecord] = []
    chunk_idx = 0

    # Page-aware: chunk each page, but allow merging small pages into a single chunk block
    # to reduce tiny chunks (common in scanned-ish or sparse pages).
    buffer_text = ""
    buffer_page_start = 1
    buffer_page_end = 1

    def flush_buffer():
        nonlocal buffer_text, buffer_page_start, buffer_page_end, chunk_idx
        if not buffer_text.strip():
            buffer_text = ""
            return
        chunks = split_into_chunks(buffer_text, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
        for c in chunks:
            records.append(
                ChunkRecord(
                    source_file=fname,
                    page_start=buffer_page_start,
                    page_end=buffer_page_end,
                    chunk_index=chunk_idx,
                    text=c,
                )
            )
            chunk_idx += 1
        buffer_text = ""

    for i, page_text in enumerate(pages, start=1):
        if not page_text:
            continue

        # If buffer empty, start range
        if not buffer_text:
            buffer_page_start = i
            buffer_page_end = i
            buffer_text = page_text
        else:
            # Try to merge next page if it still stays reasonably sized
            tentative = buffer_text + "\n\n" + page_text
            if len(tentative) <= CHUNK_MAX_CHARS * 2:
                buffer_text = tentative
                buffer_page_end = i
            else:
                flush_buffer()
                buffer_page_start = i
                buffer_page_end = i
                buffer_text = page_text

    flush_buffer()
    return records


def upsert_records(qdrant: QdrantClient, records: List[ChunkRecord]) -> None:
    if not records:
        return

    with httpx.Client(timeout=180) as http:
        # Create collection using first embedding to get dims
        first_vec = ollama_embed(http, records[0].text)
        ensure_collection(qdrant, vector_size=len(first_vec))

        points: List[qm.PointStruct] = []

        def push_batch():
            nonlocal points
            if points:
                qdrant.upsert(collection_name=COLLECTION, points=points)
                points = []

        # add first point
        rid = stable_int_id(f"{records[0].source_file}|{records[0].page_start}-{records[0].page_end}|{records[0].chunk_index}")
        points.append(
            qm.PointStruct(
                id=rid,
                vector=first_vec,
                payload={
                    "text": records[0].text,
                    "source_file": records[0].source_file,
                    "page_start": records[0].page_start,
                    "page_end": records[0].page_end,
                    "chunk_index": records[0].chunk_index,
                },
            )
        )

        for rec in records[1:]:
            vec = ollama_embed(http, rec.text)
            rid = stable_int_id(f"{rec.source_file}|{rec.page_start}-{rec.page_end}|{rec.chunk_index}")
            points.append(
                qm.PointStruct(
                    id=rid,
                    vector=vec,
                    payload={
                        "text": rec.text,
                        "source_file": rec.source_file,
                        "page_start": rec.page_start,
                        "page_end": rec.page_end,
                        "chunk_index": rec.chunk_index,
                    },
                )
            )

            if len(points) >= UPSERT_BATCH_SIZE:
                push_batch()

        push_batch()


def main():
    pdfs = sorted(glob.glob(PDF_GLOB))
    if not pdfs:
        print(f"❌ No PDFs found in {DATA_DIR}")
        return

    qdrant = QdrantClient(url=QDRANT_URL)

    total_chunks = 0
    for pdf in pdfs:
        print(f"\n📄 Processing: {pdf}")
        records = build_chunks_for_pdf(pdf)
        print(f"   → extracted chunks: {len(records)}")
        upsert_records(qdrant, records)
        total_chunks += len(records)

    print(f"\n✅ Done. Total chunks upserted into '{COLLECTION}': {total_chunks}")
    print("   Qdrant UI: http://localhost:6333/dashboard")


if __name__ == "__main__":
    main()