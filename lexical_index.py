from __future__ import annotations

import os
import re
import sqlite3
from typing import Any, Dict, Iterable, List


def connect_index(db_path: str) -> sqlite3.Connection:
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    return conn


def initialize_index(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS lexical_chunks
        USING fts5(
            chunk_key UNINDEXED,
            source_file UNINDEXED,
            page_start UNINDEXED,
            page_end UNINDEXED,
            chunk_index UNINDEXED,
            text,
            tokenize = 'unicode61'
        );
        """
    )
    conn.commit()


def replace_source_records(conn: sqlite3.Connection, records: Iterable[Any]) -> int:
    records = list(records)
    if not records:
        return 0

    source_file = records[0].source_file
    conn.execute("DELETE FROM lexical_chunks WHERE source_file = ?", (source_file,))
    conn.executemany(
        """
        INSERT INTO lexical_chunks (
            chunk_key, source_file, page_start, page_end, chunk_index, text
        ) VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                f"{rec.source_file}|{rec.page_start}-{rec.page_end}|{rec.chunk_index}",
                rec.source_file,
                rec.page_start,
                rec.page_end,
                rec.chunk_index,
                rec.text,
            )
            for rec in records
        ],
    )
    conn.commit()
    return len(records)


def _match_query(query: str) -> str:
    tokens = re.findall(r"[A-Za-z0-9]{2,}", query.lower())
    if not tokens:
        return ""
    unique_tokens = list(dict.fromkeys(tokens))
    return " OR ".join(f'"{token}"' for token in unique_tokens[:12])


def search_lexical(conn: sqlite3.Connection, query: str, limit: int = 5) -> List[Dict[str, Any]]:
    match_query = _match_query(query)
    if not match_query:
        return []

    try:
        rows = conn.execute(
            """
            SELECT
                chunk_key,
                source_file,
                page_start,
                page_end,
                chunk_index,
                text,
                bm25(lexical_chunks) AS score
            FROM lexical_chunks
            WHERE lexical_chunks MATCH ?
            ORDER BY score
            LIMIT ?
            """,
            (match_query, limit),
        ).fetchall()
    except sqlite3.OperationalError:
        rows = []

    return [
        {
            "chunk_key": row["chunk_key"],
            "source_file": row["source_file"],
            "page_start": int(row["page_start"]),
            "page_end": int(row["page_end"]),
            "chunk_index": int(row["chunk_index"]),
            "text": row["text"],
            # bm25 is lower-is-better, so convert to higher-is-better for the rest of the app.
            "score": float(-row["score"]),
        }
        for row in rows
    ]
