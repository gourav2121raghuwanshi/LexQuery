from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from make_call_to_fine_tuned_llm import (
    ask_answer_model,
    format_context,
    judge_answer,
    retrieve_chunks,
)


BASE_DIR = "/Users/gourav/Developer/python/GenAi/BTP/evaluate"
MODES = [("vector", "vector"), ("page_index", "pageIndex")]


def read_queries(path: str) -> List[str]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            q = line.strip()
            if not q or q.startswith("#"):
                continue
            out.append(q)
    return out


def doc_key(hit: Dict[str, Any]) -> str:
    return f"{hit.get('source_file', 'unknown')}|{hit.get('page_start', '?')}-{hit.get('page_end', '?')}"


def overlap_at_k(a: List[str], b: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(a[:k]) & set(b[:k])) / float(k)


def jaccard_at_k(a: List[str], b: List[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    union = sa | sb
    return (len(sa & sb) / float(len(union))) if union else 0.0


def margin_gap(scores: List[float], k: int) -> Tuple[float, float]:
    if not scores:
        return 0.0, 0.0
    margin = (scores[0] - scores[1]) if len(scores) >= 2 else 0.0
    gap = (scores[0] - scores[k - 1]) if len(scores) >= k else 0.0
    return margin, gap


def avg(xs: List[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def mode_result_for_query(
    query: str,
    *,
    mode: str,
    topk: int,
    max_context_chars: int,
    with_answers: bool,
    temperature: float,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    hits = retrieve_chunks(query, topk, mode)
    latency_ms = int((time.perf_counter() - t0) * 1000)

    docs = [doc_key(hit) for hit in hits]
    scores = [float(hit.get("score", 0.0)) for hit in hits]
    unique_sources = len({hit.get("source_file", "unknown") for hit in hits})
    m, g = margin_gap(scores, topk)

    result: Dict[str, Any] = {
        "latency_ms": latency_ms,
        "docs": docs,
        "scores": scores,
        "unique_sources": unique_sources,
        "margin_top1_top2": m,
        "gap_top1_topk": g,
        "context_chars": 0,
        "answer": "",
        "judge": None,
    }

    context = format_context(hits, max_chars=max_context_chars) if hits else ""
    result["context_chars"] = len(context)

    if with_answers and context.strip():
        answer = ask_answer_model(query, context, temperature=temperature)
        result["answer"] = answer
        try:
            result["judge"] = judge_answer(query, context, answer)
        except Exception as exc:
            result["judge"] = {
                "verdict": "error",
                "relevance_score": 0,
                "groundedness_score": 0,
                "completeness_score": 0,
                "rationale": f"Judge failed: {exc}",
            }

    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default=os.path.join(BASE_DIR, "queries.txt"))
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--outdir", default=os.path.join(BASE_DIR, "rag_mode_outputs"))
    parser.add_argument("--with-answers", action="store_true", help="Generate answers and judge them for each mode")
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    queries = read_queries(args.queries)
    if not queries:
        print(f"❌ No queries found in {args.queries}")
        return

    per_query_mode: Dict[str, Dict[str, Any]] = {}

    for qi, query in enumerate(queries, start=1):
        print(f"\n🔎 [{qi}/{len(queries)}] {query}")
        per_query_mode[query] = {}
        for mode, label in MODES:
            result = mode_result_for_query(
                query,
                mode=mode,
                topk=args.topk,
                max_context_chars=args.max_context_chars,
                with_answers=args.with_answers,
                temperature=args.temperature,
            )
            per_query_mode[query][label] = result
            top1 = result["docs"][0] if result["docs"] else "NONE"
            print(
                f"  {label:8} latency={result['latency_ms']:4}ms "
                f"top1={top1} unique_sources={result['unique_sources']}"
            )

    summary: Dict[str, Any] = {
        "settings": vars(args),
        "modes": [label for _, label in MODES],
        "per_mode": {},
        "pairwise": {},
    }

    for _, label in MODES:
        latencies = []
        margins = []
        gaps = []
        sources = []
        context_chars = []
        relevance = []
        groundedness = []
        completeness = []

        for query in queries:
            result = per_query_mode[query][label]
            latencies.append(float(result["latency_ms"]))
            margins.append(float(result["margin_top1_top2"]))
            gaps.append(float(result["gap_top1_topk"]))
            sources.append(float(result["unique_sources"]))
            context_chars.append(float(result["context_chars"]))
            judge = result.get("judge") or {}
            if judge:
                relevance.append(float(judge.get("relevance_score", 0)))
                groundedness.append(float(judge.get("groundedness_score", 0)))
                completeness.append(float(judge.get("completeness_score", 0)))

        summary["per_mode"][label] = {
            "avg_latency_ms": avg(latencies),
            "avg_margin_top1_top2": avg(margins),
            "avg_gap_top1_topk": avg(gaps),
            "avg_unique_sources": avg(sources),
            "avg_context_chars": avg(context_chars),
            "avg_relevance_score": avg(relevance),
            "avg_groundedness_score": avg(groundedness),
            "avg_completeness_score": avg(completeness),
        }

    labels = [label for _, label in MODES]
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            a, b = labels[i], labels[j]
            overlaps = []
            jaccs = []
            for query in queries:
                da = per_query_mode[query][a]["docs"]
                db = per_query_mode[query][b]["docs"]
                overlaps.append(overlap_at_k(da, db, args.topk))
                jaccs.append(jaccard_at_k(da, db, args.topk))
            summary["pairwise"][f"{a} vs {b}"] = {
                "avg_overlap_at_k": avg(overlaps),
                "avg_jaccard_at_k": avg(jaccs),
            }

    raw_path = os.path.join(args.outdir, "raw_mode_results.json")
    summary_path = os.path.join(args.outdir, "summary_mode_metrics.json")
    csv_path = os.path.join(args.outdir, "mode_results_flat.csv")

    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(per_query_mode, f, indent=2)

    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "query,mode,rank,doc_key,score,latency_ms,unique_sources,margin_top1_top2,"
            "gap_top1_topk,context_chars,relevance_score,groundedness_score,completeness_score\n"
        )
        for query in queries:
            for _, label in MODES:
                result = per_query_mode[query][label]
                judge = result.get("judge") or {}
                for rank, dk in enumerate(result["docs"], start=1):
                    score = result["scores"][rank - 1] if rank - 1 < len(result["scores"]) else ""
                    f.write(
                        f"{json.dumps(query)},{label},{rank},{json.dumps(dk)},{score},"
                        f"{result['latency_ms']},{result['unique_sources']},"
                        f"{result['margin_top1_top2']},{result['gap_top1_topk']},"
                        f"{result['context_chars']},{judge.get('relevance_score', '')},"
                        f"{judge.get('groundedness_score', '')},{judge.get('completeness_score', '')}\n"
                    )

    print("\n" + "=" * 70)
    print("✅ RAG MODE COMPARISON SUMMARY")
    print("=" * 70)
    for label, metrics in summary["per_mode"].items():
        print(f"\n{label}")
        print(f"  avg_latency_ms:          {metrics['avg_latency_ms']:.2f}")
        print(f"  avg_margin(top1-top2):   {metrics['avg_margin_top1_top2']:.6f}")
        print(f"  avg_gap(top1-topK):      {metrics['avg_gap_top1_topk']:.6f}")
        print(f"  avg_unique_sources:      {metrics['avg_unique_sources']:.2f}")
        print(f"  avg_context_chars:       {metrics['avg_context_chars']:.2f}")
        if args.with_answers:
            print(f"  avg_relevance_score:     {metrics['avg_relevance_score']:.2f}")
            print(f"  avg_groundedness_score:  {metrics['avg_groundedness_score']:.2f}")
            print(f"  avg_completeness_score:  {metrics['avg_completeness_score']:.2f}")

    print("\nPairwise retrieval similarity:")
    for key, value in summary["pairwise"].items():
        print(
            f"  {key}: overlap@{args.topk}={value['avg_overlap_at_k']:.3f}, "
            f"jaccard@{args.topk}={value['avg_jaccard_at_k']:.3f}"
        )

    print("\n📁 Saved files:")
    print(" -", raw_path)
    print(" -", summary_path)
    print(" -", csv_path)
    print("=" * 70)


if __name__ == "__main__":
    main()
