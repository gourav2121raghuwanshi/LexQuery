# strategy_euclid_mergedpages.py
from __future__ import annotations

import os, re, glob, hashlib
from dataclasses import dataclass
from typing import List

import httpx
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

DATA_DIR = "data"
PDF_GLOB = os.path.join(DATA_DIR, "*.pdf")

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "btp_docs_euclid_mergedpages"

# larger blocks
CHUNK_MAX_CHARS = 2600
CHUNK_OVERLAP_CHARS = 100
UPSERT_BATCH_SIZE = 64

# Euclidean distance search
QDRANT_DISTANCE = qm.Distance.EUCLID

def stable_int_id(s: str) -> int:
    h = hashlib.sha1(s.encode("utf-8")).hexdigest()
    return int(h[:15], 16)

def clean_text(t: str) -> str:
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

def extract_pdf_pages(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    return [clean_text((p.extract_text() or "")) for p in reader.pages]

def split_hard(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """Hard split (not paragraph aware) to show a different chunking baseline."""
    if not text:
        return []
    out = []
    start = 0
    n = len(text)
    step = max(1, max_chars - overlap_chars)
    while start < n:
        end = min(start + max_chars, n)
        part = text[start:end].strip()
        if part:
            out.append(part)
        if end >= n:
            break
        start += step
    return out

def ollama_embed(client: httpx.Client, text: str) -> List[float]:
    r = client.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]

def ensure_collection(qdrant: QdrantClient, vector_size: int) -> None:
    existing = {c.name for c in qdrant.get_collections().collections}
    if COLLECTION in existing:
        return
    qdrant.create_collection(
        collection_name=COLLECTION,
        vectors_config=qm.VectorParams(size=vector_size, distance=QDRANT_DISTANCE),
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

    # Aggressively merge 3 pages at a time (or until size hits ~2*max)
    buffer = ""
    start_page = 1
    end_page = 1

    def flush():
        nonlocal buffer, chunk_idx, start_page, end_page
        if not buffer.strip():
            buffer = ""
            return
        chunks = split_hard(buffer, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
        for c in chunks:
            records.append(ChunkRecord(fname, start_page, end_page, chunk_idx, c))
            chunk_idx += 1
        buffer = ""

    for i, txt in enumerate(pages, start=1):
        if not txt:
            continue
        if not buffer:
            start_page = i
            end_page = i
            buffer = txt
            continue

        tentative = buffer + "\n\n" + txt
        # allow bigger merged blocks than Strategy 1
        if len(tentative) <= CHUNK_MAX_CHARS * 3:
            buffer = tentative
            end_page = i
        else:
            flush()
            start_page = i
            end_page = i
            buffer = txt

    flush()
    return records

def upsert_records(qdrant: QdrantClient, records: List[ChunkRecord]) -> None:
    if not records:
        return

    with httpx.Client(timeout=180) as http:
        first_vec = ollama_embed(http, records[0].text)
        ensure_collection(qdrant, vector_size=len(first_vec))

        batch: List[qm.PointStruct] = []

        def push():
            nonlocal batch
            if batch:
                qdrant.upsert(collection_name=COLLECTION, points=batch)
                batch = []

        for idx, rec in enumerate(records):
            vec = first_vec if idx == 0 else ollama_embed(http, rec.text)
            rid = stable_int_id(f"{rec.source_file}|{rec.page_start}-{rec.page_end}|{rec.chunk_index}")
            batch.append(
                qm.PointStruct(
                    id=rid,
                    vector=vec,
                    payload={
                        "text": rec.text,
                        "source_file": rec.source_file,
                        "page_start": rec.page_start,
                        "page_end": rec.page_end,
                        "chunk_index": rec.chunk_index,
                        "strategy": "euclid_mergedpages_hardsplit",
                    },
                )
            )
            if len(batch) >= UPSERT_BATCH_SIZE:
                push()

        push()

def main():
    pdfs = sorted(glob.glob(PDF_GLOB))
    if not pdfs:
        print(f"❌ No PDFs found in {DATA_DIR}")
        return

    qdrant = QdrantClient(url=QDRANT_URL)

    total = 0
    for pdf in pdfs:
        print(f"\n📄 Processing: {pdf}")
        recs = build_chunks_for_pdf(pdf)
        print(f"   → chunks: {len(recs)}")
        upsert_records(qdrant, recs)
        total += len(recs)

    print(f"\n✅ Done. Total chunks upserted into '{COLLECTION}': {total}")

if __name__ == "__main__":
    main()