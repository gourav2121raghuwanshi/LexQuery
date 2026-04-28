from __future__ import annotations

import glob
import os

from create_embeddings import DATA_DIR, PDF_GLOB, build_chunks_for_pdf
from lexical_index import connect_index, initialize_index, replace_source_records


LEXICAL_DB_PATH = "./lexical_chunks.db"


def main() -> None:
    pdfs = sorted(glob.glob(PDF_GLOB))
    if not pdfs:
        print(f"❌ No PDFs found in {DATA_DIR}")
        return

    conn = connect_index(LEXICAL_DB_PATH)
    initialize_index(conn)

    total_records = 0
    for pdf in pdfs:
        print(f"\n📄 Processing: {pdf}")
        records = build_chunks_for_pdf(pdf)
        count = replace_source_records(conn, records)
        total_records += count
        print(f"   → lexical chunks indexed: {count}")

    conn.close()
    print(f"\n✅ Lexical index ready at {LEXICAL_DB_PATH} with {total_records} chunks")


if __name__ == "__main__":
    main()
