from __future__ import annotations

import os, time, json, argparse
from typing import Dict, List, Any, Tuple
import httpx

from qdrant_client import QdrantClient

# --- existing collections ---
STRATEGIES = {
    "cosine_para": "btp_docs_cosine_para",
    "dot_smallchunks": "btp_docs_dot_smallchunks",
    "euclid_mergedpages": "btp_docs_euclid_mergedpages",
}

QDRANT_URL = "http://127.0.0.1:6333"
OLLAMA_EMBED_URL = "http://127.0.0.1:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

BASE_DIR = "/Users/gourav/Developer/python/GenAi/BTP/evaluate"

# -----------------------------
def read_queries(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q or q.startswith("#"):
                continue
            out.append(q)
    return out

def ollama_embed(http: httpx.Client, text: str) -> List[float]:
    r = http.post(OLLAMA_EMBED_URL, json={"model": EMBED_MODEL, "prompt": text})
    r.raise_for_status()
    return r.json()["embedding"]

def doc_key(payload: Dict[str, Any]) -> str:
    sf = payload.get("source_file", "unknown")
    ps = payload.get("page_start", "?")
    pe = payload.get("page_end", "?")
    return f"{sf}|{ps}-{pe}"

def overlap_at_k(a: List[str], b: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(a[:k]) & set(b[:k])) / float(k)

def jaccard_at_k(a: List[str], b: List[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    u = sa | sb
    return (len(sa & sb) / float(len(u))) if u else 0.0

def margin_gap(scores: List[float], k: int) -> Tuple[float, float]:
    # margin = top1-top2, gap = top1-topk
    if not scores:
        return 0.0, 0.0
    margin = (scores[0] - scores[1]) if len(scores) >= 2 else 0.0
    gap = (scores[0] - scores[k-1]) if len(scores) >= k else 0.0
    return margin, gap

# -----------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=os.path.join(BASE_DIR, "queries.txt"))
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--fetch", type=int, default=80, help="fetch top N points then group to topK docs")
    parser.add_argument("--outdir", default=os.path.join(BASE_DIR, "metrics_outputs"))
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    queries = read_queries(args.queries)
    if not queries:
        print(f"❌ No queries found in {args.queries}")
        return

    qdrant = QdrantClient(url=QDRANT_URL)
    existing = {c.name for c in qdrant.get_collections().collections}
    missing = [c for c in STRATEGIES.values() if c not in existing]
    if missing:
        print("❌ Missing collections:", missing)
        return

    per_query_strategy = {}  # query -> strategy -> {"docs":[...], "scores":[...], "latency":...}

    with httpx.Client(timeout=180) as http:
        for qi, q in enumerate(queries, start=1):
            print(f"\n🔎 [{qi}/{len(queries)}] {q}")
            per_query_strategy[q] = {}

            qvec = ollama_embed(http, q)

            for sname, coll in STRATEGIES.items():
                t0 = time.perf_counter()
                res = qdrant.query_points(
                collection_name=coll,
                query=qvec,
                limit=args.fetch,
                with_payload=True,
                with_vectors=False,
            )

                hits = res.points
                latency_ms = int((time.perf_counter() - t0) * 1000)

                # group chunk-level hits -> doc_key using best score
                best: Dict[str, Tuple[float, Dict[str, Any]]] = {}
                for h in hits:
                    payload = h.payload or {}
                    dk = doc_key(payload)
                    sc = float(h.score)
                    if dk not in best or sc > best[dk][0]:
                        best[dk] = (sc, payload)

                ranked = sorted(best.items(), key=lambda kv: kv[1][0], reverse=True)[: args.topk]
                docs = [dk for dk, _ in ranked]
                scores = [kv[1][0] for kv in ranked]

                m, g = margin_gap(scores, args.topk)

                per_query_strategy[q][sname] = {
                    "collection": coll,
                    "latency_ms": latency_ms,
                    "docs": docs,
                    "scores": scores,
                    "margin_top1_top2": m,
                    "gap_top1_topk": g,
                }

                top1 = docs[0] if docs else "NONE"
                top1s = scores[0] if scores else None
                print(f"  {sname:16} latency={latency_ms:4}ms  top1={top1}  score={top1s}")

    # -----------------------------
    # Building numeric comparisons
    # -----------------------------
    # 1) Per-strategy averages
    summary = {"per_strategy": {}, "pairwise": {}, "settings": vars(args), "strategies": STRATEGIES}

    def avg(xs: List[float]) -> float:
        return sum(xs) / len(xs) if xs else 0.0

    for sname in STRATEGIES.keys():
        lat = []
        margins = []
        gaps = []
        for q in queries:
            r = per_query_strategy[q][sname]
            lat.append(r["latency_ms"])
            margins.append(r["margin_top1_top2"])
            gaps.append(r["gap_top1_topk"])
        summary["per_strategy"][sname] = {
            "collection": STRATEGIES[sname],
            "avg_latency_ms": avg([float(x) for x in lat]),
            "avg_margin_top1_top2": avg(margins),
            "avg_gap_top1_topk": avg(gaps),
        }

    # 2) Pairwise overlaps
    names = list(STRATEGIES.keys())
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            overlaps = []
            jaccs = []
            for q in queries:
                da = per_query_strategy[q][a]["docs"]
                db = per_query_strategy[q][b]["docs"]
                overlaps.append(overlap_at_k(da, db, args.topk))
                jaccs.append(jaccard_at_k(da, db, args.topk))
            summary["pairwise"][f"{a} vs {b}"] = {
                "avg_overlap_at_k": avg(overlaps),
                "avg_jaccard_at_k": avg(jaccs),
            }

    # -----------------------------
    # Saveing artifacts
    # -----------------------------
    # A) Raw per-query results (for inspection)
    raw_path = os.path.join(args.outdir, "raw_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(per_query_strategy, f, indent=2)

    # B) Summary metrics
    summary_path = os.path.join(args.outdir, "summary_metrics.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # C) Flat CSV for plotting
    csv_path = os.path.join(args.outdir, "results_flat.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("query,strategy,collection,rank,doc_key,score,latency_ms,margin_top1_top2,gap_top1_topk\n")
        for q in queries:
            for sname in STRATEGIES.keys():
                r = per_query_strategy[q][sname]
                for rank, dk in enumerate(r["docs"], start=1):
                    sc = r["scores"][rank-1] if rank-1 < len(r["scores"]) else ""
                    f.write(
                        f"{json.dumps(q)},{sname},{r['collection']},{rank},{json.dumps(dk)},{sc},"
                        f"{r['latency_ms']},{r['margin_top1_top2']},{r['gap_top1_topk']}\n"
                    )

    # Print final summary
    print("\n" + "=" * 70)
    print("✅ STORED-VECTOR COMPARISON SUMMARY")
    print("=" * 70)
    for sname, s in summary["per_strategy"].items():
        print(f"\n{sname} ({s['collection']})")
        print(f"  avg_latency_ms:        {s['avg_latency_ms']:.2f}")
        print(f"  avg_margin(top1-top2): {s['avg_margin_top1_top2']:.6f}")
        print(f"  avg_gap(top1-topK):    {s['avg_gap_top1_topk']:.6f}")

    print("\nPairwise similarity of retrieved docs (higher = more similar):")
    for k, v in summary["pairwise"].items():
        print(f"  {k}: overlap@{args.topk}={v['avg_overlap_at_k']:.3f}, jaccard@{args.topk}={v['avg_jaccard_at_k']:.3f}")

    print("\n📁 Saved files:")
    print(" -", raw_path)
    print(" -", summary_path)
    print(" -", csv_path)
    print("=" * 70)


if __name__ == "__main__":
    main()