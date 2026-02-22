# strategy_dot_smallchunks_topk.py
from __future__ import annotations

import os, re, glob, hashlib
from dataclasses import dataclass
from typing import List

import httpx
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client.http import models as qm

DATA_DIR = "./data"
PDF_GLOB = os.path.join(DATA_DIR, "*.pdf")

OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

QDRANT_URL = "http://127.0.0.1:6333"
COLLECTION = "btp_docs_dot_smallchunks"

# smaller chunks -> more local signal
CHUNK_MAX_CHARS = 900
CHUNK_OVERLAP_CHARS = 150

UPSERT_BATCH_SIZE = 64
QDRANT_DISTANCE = qm.Distance.DOT

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

def split_into_chunks(text: str, max_chars: int, overlap_chars: int) -> List[str]:
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

        add_len = len(p) + (2 if buf else 0)
        if buf_len + add_len <= max_chars:
            buf.append(p)
            buf_len += add_len
        else:
            flush()
            buf.append(p)
            buf_len = len(p)

    flush()

    if overlap_chars > 0 and len(chunks) > 1:
        out = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = out[-1]
            tail = prev[-overlap_chars:] if len(prev) > overlap_chars else prev
            out.append((tail + "\n\n" + chunks[i]).strip())
        chunks = out

    return chunks

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

    for i, page_text in enumerate(pages, start=1):
        if not page_text:
            continue
        chunks = split_into_chunks(page_text, CHUNK_MAX_CHARS, CHUNK_OVERLAP_CHARS)
        for c in chunks:
            records.append(ChunkRecord(fname, i, i, chunk_idx, c))
            chunk_idx += 1

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
                        "strategy": "dot_smallchunks",
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