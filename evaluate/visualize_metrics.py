from __future__ import annotations

import os
import json
import argparse
from typing import Dict, Any, List, Tuple

import numpy as np
import matplotlib.pyplot as plt


BASE_DIR = "/Users/gourav/Developer/python/GenAi/BTP/evaluate"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def savefig(outdir: str, name: str) -> str:
    path = os.path.join(outdir, name)
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()
    return path


def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def overlap_at_k(a: List[str], b: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return len(set(a[:k]) & set(b[:k])) / float(k)


def jaccard_at_k(a: List[str], b: List[str], k: int) -> float:
    sa, sb = set(a[:k]), set(b[:k])
    u = sa | sb
    return (len(sa & sb) / float(len(u))) if u else 0.0


def extract_per_query_arrays(raw: Dict[str, Any], strategies: List[str]) -> Dict[str, Dict[str, List[float]]]:
    """
    Returns dict[strategy] = {"latency": [...], "margin": [...], "gap": [...]}
    Each list is per query.
    """
    out: Dict[str, Dict[str, List[float]]] = {s: {"latency": [], "margin": [], "gap": []} for s in strategies}

    for q, smap in raw.items():
        for s in strategies:
            r = smap.get(s)
            if not r:
                continue
            out[s]["latency"].append(float(r.get("latency_ms", 0.0)))
            out[s]["margin"].append(float(r.get("margin_top1_top2", 0.0)))
            out[s]["gap"].append(float(r.get("gap_top1_topk", 0.0)))

    return out


def plot_boxplots(per: Dict[str, Dict[str, List[float]]], outdir: str) -> List[str]:
    strategies = sorted(per.keys())
    outputs = []

    # Latency
    plt.figure()
    data = [per[s]["latency"] for s in strategies]
    plt.boxplot(data, labels=strategies, showfliers=True)
    plt.ylabel("Latency (ms)")
    plt.title("Query Latency Distribution by Strategy")
    plt.xticks(rotation=20, ha="right")
    outputs.append(savefig(outdir, "01_latency_boxplot.png"))

    # Margin
    plt.figure()
    data = [per[s]["margin"] for s in strategies]
    plt.boxplot(data, labels=strategies, showfliers=True)
    plt.ylabel("Score margin (top1 - top2)")
    plt.title("Top-1 vs Top-2 Score Margin by Strategy")
    plt.xticks(rotation=20, ha="right")
    outputs.append(savefig(outdir, "02_margin_boxplot.png"))

    # Gap
    plt.figure()
    data = [per[s]["gap"] for s in strategies]
    plt.boxplot(data, labels=strategies, showfliers=True)
    plt.ylabel("Score gap (top1 - topK)")
    plt.title("Top-1 vs Top-K Score Gap by Strategy")
    plt.xticks(rotation=20, ha="right")
    outputs.append(savefig(outdir, "03_gap_boxplot.png"))

    return outputs


def plot_pairwise_heatmaps(raw: Dict[str, Any], strategies: List[str], outdir: str, k: int) -> List[str]:
    outputs = []
    S = sorted(strategies)
    n = len(S)

    # Build average overlap and jaccard matrices
    overlap_mat = np.zeros((n, n), dtype=float)
    jacc_mat = np.zeros((n, n), dtype=float)

    queries = list(raw.keys())

    for i in range(n):
        for j in range(n):
            if i == j:
                overlap_mat[i, j] = 1.0
                jacc_mat[i, j] = 1.0
                continue

            vals_o = []
            vals_j = []
            for q in queries:
                A = raw[q][S[i]]["docs"]
                B = raw[q][S[j]]["docs"]
                vals_o.append(overlap_at_k(A, B, k))
                vals_j.append(jaccard_at_k(A, B, k))
            overlap_mat[i, j] = float(np.mean(vals_o)) if vals_o else 0.0
            jacc_mat[i, j] = float(np.mean(vals_j)) if vals_j else 0.0

    # Overlap heatmap
    plt.figure()
    im = plt.imshow(overlap_mat, aspect="auto")
    plt.colorbar(im, label=f"Avg Overlap@{k}")
    plt.xticks(range(n), S, rotation=20, ha="right")
    plt.yticks(range(n), S)
    plt.title(f"Average Pairwise Top-{k} Overlap (doc_key)")

    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{overlap_mat[i, j]:.2f}", ha="center", va="center")

    outputs.append(savefig(outdir, f"04_overlap_heatmap_at{k}.png"))

    # Jaccard heatmap
    plt.figure()
    im = plt.imshow(jacc_mat, aspect="auto")
    plt.colorbar(im, label=f"Avg Jaccard@{k}")
    plt.xticks(range(n), S, rotation=20, ha="right")
    plt.yticks(range(n), S)
    plt.title(f"Average Pairwise Top-{k} Jaccard (doc_key)")

    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{jacc_mat[i, j]:.2f}", ha="center", va="center")

    outputs.append(savefig(outdir, f"05_jaccard_heatmap_at{k}.png"))
    return outputs


def plot_top1_agreement(raw: Dict[str, Any], strategies: List[str], outdir: str) -> str:
    """
    For each strategy: fraction of queries where its top-1 doc_key equals the majority top-1 among all strategies.
    """
    S = sorted(strategies)
    queries = list(raw.keys())

    agree = {s: 0 for s in S}
    for q in queries:
        top1s = {s: (raw[q][s]["docs"][0] if raw[q][s]["docs"] else "") for s in S}
        picks = [v for v in top1s.values() if v]
        if not picks:
            continue

        freq: Dict[str, int] = {}
        for p in picks:
            freq[p] = freq.get(p, 0) + 1
        majority = max(freq.items(), key=lambda kv: kv[1])[0]

        for s in S:
            if top1s[s] == majority:
                agree[s] += 1

    total = len(queries) if queries else 1
    y = [agree[s] / total for s in S]

    plt.figure()
    plt.bar(S, y)
    plt.ylim(0, 1)
    plt.ylabel("Fraction of queries")
    plt.title("Top-1 Agreement with Majority (proxy stability)")
    plt.xticks(rotation=20, ha="right")
    return savefig(outdir, "06_top1_agreement.png")


def plot_summary_bars(summary: Dict[str, Any], outdir: str) -> List[str]:
    """
    Simple bar charts from summary_metrics.json: avg latency, avg margin, avg gap.
    """
    per = summary.get("per_strategy", {})
    strategies = sorted(per.keys())

    lat = [float(per[s].get("avg_latency_ms", 0.0)) for s in strategies]
    mar = [float(per[s].get("avg_margin_top1_top2", 0.0)) for s in strategies]
    gap = [float(per[s].get("avg_gap_top1_topk", 0.0)) for s in strategies]

    outputs = []

    plt.figure()
    plt.bar(strategies, lat)
    plt.ylabel("Average latency (ms)")
    plt.title("Average Latency by Strategy")
    plt.xticks(rotation=20, ha="right")
    outputs.append(savefig(outdir, "07_avg_latency_bar.png"))

    plt.figure()
    plt.bar(strategies, mar)
    plt.ylabel("Average margin (top1 - top2)")
    plt.title("Average Top-1 vs Top-2 Margin by Strategy")
    plt.xticks(rotation=20, ha="right")
    outputs.append(savefig(outdir, "08_avg_margin_bar.png"))

    plt.figure()
    plt.bar(strategies, gap)
    plt.ylabel("Average gap (top1 - topK)")
    plt.title("Average Top-1 vs Top-K Gap by Strategy")
    plt.xticks(rotation=20, ha="right")
    outputs.append(savefig(outdir, "09_avg_gap_bar.png"))

    return outputs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=os.path.join(BASE_DIR, "metrics_outputs"))
    parser.add_argument("--topk", type=int, default=10)
    args = parser.parse_args()

    raw_path = os.path.join(args.outdir, "raw_results.json")
    summary_path = os.path.join(args.outdir, "summary_metrics.json")
    plot_dir = os.path.join(args.outdir, "plots")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw_results.json not found: {raw_path}")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_metrics.json not found: {summary_path}")

    ensure_dir(plot_dir)

    raw = load_json(raw_path)
    summary = load_json(summary_path)

    strategies = list(summary.get("strategies", {}).keys())
    if not strategies:
        # fallback: infer from raw
        any_q = next(iter(raw.keys()))
        strategies = list(raw[any_q].keys())

    per = extract_per_query_arrays(raw, strategies)

    outputs: List[str] = []
    outputs += plot_boxplots(per, plot_dir)
    outputs += plot_pairwise_heatmaps(raw, strategies, plot_dir, k=args.topk)
    outputs.append(plot_top1_agreement(raw, strategies, plot_dir))
    outputs += plot_summary_bars(summary, plot_dir)

    print("\n✅ Saved plots:")
    for p in outputs:
        print(" -", p)


if __name__ == "__main__":
    main()