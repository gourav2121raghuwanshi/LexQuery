from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np


BASE_DIR = "/Users/gourav/Developer/python/GenAi/BTP/evaluate"


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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
    union = sa | sb
    return (len(sa & sb) / float(len(union))) if union else 0.0


def extract_per_query_arrays(raw: Dict[str, Any], modes: List[str]) -> Dict[str, Dict[str, List[float]]]:
    out: Dict[str, Dict[str, List[float]]] = {
        mode: {
            "latency": [],
            "margin": [],
            "gap": [],
            "sources": [],
            "context_chars": [],
            "relevance": [],
            "groundedness": [],
            "completeness": [],
        }
        for mode in modes
    }

    for query, mode_map in raw.items():
        for mode in modes:
            result = mode_map.get(mode)
            if not result:
                continue
            out[mode]["latency"].append(float(result.get("latency_ms", 0.0)))
            out[mode]["margin"].append(float(result.get("margin_top1_top2", 0.0)))
            out[mode]["gap"].append(float(result.get("gap_top1_topk", 0.0)))
            out[mode]["sources"].append(float(result.get("unique_sources", 0.0)))
            out[mode]["context_chars"].append(float(result.get("context_chars", 0.0)))
            judge = result.get("judge") or {}
            if judge:
                out[mode]["relevance"].append(float(judge.get("relevance_score", 0.0)))
                out[mode]["groundedness"].append(float(judge.get("groundedness_score", 0.0)))
                out[mode]["completeness"].append(float(judge.get("completeness_score", 0.0)))

    return out


def plot_boxplots(per: Dict[str, Dict[str, List[float]]], outdir: str) -> List[str]:
    modes = sorted(per.keys())
    outputs = []

    plt.figure()
    plt.boxplot([per[mode]["latency"] for mode in modes], labels=modes, showfliers=True)
    plt.ylabel("Latency (ms)")
    plt.title("Query Latency Distribution by RAG Mode")
    outputs.append(savefig(outdir, "01_latency_boxplot.png"))

    plt.figure()
    plt.boxplot([per[mode]["margin"] for mode in modes], labels=modes, showfliers=True)
    plt.ylabel("Score margin (top1 - top2)")
    plt.title("Top-1 vs Top-2 Score Margin by RAG Mode")
    outputs.append(savefig(outdir, "02_margin_boxplot.png"))

    plt.figure()
    plt.boxplot([per[mode]["gap"] for mode in modes], labels=modes, showfliers=True)
    plt.ylabel("Score gap (top1 - topK)")
    plt.title("Top-1 vs Top-K Score Gap by RAG Mode")
    outputs.append(savefig(outdir, "03_gap_boxplot.png"))

    plt.figure()
    plt.boxplot([per[mode]["sources"] for mode in modes], labels=modes, showfliers=True)
    plt.ylabel("Unique source files")
    plt.title("Unique Source Coverage by RAG Mode")
    outputs.append(savefig(outdir, "04_unique_sources_boxplot.png"))

    return outputs


def plot_pairwise_heatmaps(raw: Dict[str, Any], modes: List[str], outdir: str, k: int) -> List[str]:
    outputs = []
    labels = sorted(modes)
    n = len(labels)
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
            for query in queries:
                a = raw[query][labels[i]]["docs"]
                b = raw[query][labels[j]]["docs"]
                vals_o.append(overlap_at_k(a, b, k))
                vals_j.append(jaccard_at_k(a, b, k))
            overlap_mat[i, j] = float(np.mean(vals_o)) if vals_o else 0.0
            jacc_mat[i, j] = float(np.mean(vals_j)) if vals_j else 0.0

    plt.figure()
    im = plt.imshow(overlap_mat, aspect="auto")
    plt.colorbar(im, label=f"Avg Overlap@{k}")
    plt.xticks(range(n), labels)
    plt.yticks(range(n), labels)
    plt.title(f"Average Pairwise Top-{k} Overlap by RAG Mode")
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{overlap_mat[i, j]:.2f}", ha="center", va="center")
    outputs.append(savefig(outdir, f"05_overlap_heatmap_at{k}.png"))

    plt.figure()
    im = plt.imshow(jacc_mat, aspect="auto")
    plt.colorbar(im, label=f"Avg Jaccard@{k}")
    plt.xticks(range(n), labels)
    plt.yticks(range(n), labels)
    plt.title(f"Average Pairwise Top-{k} Jaccard by RAG Mode")
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{jacc_mat[i, j]:.2f}", ha="center", va="center")
    outputs.append(savefig(outdir, f"06_jaccard_heatmap_at{k}.png"))
    return outputs


def plot_summary_bars(summary: Dict[str, Any], outdir: str) -> List[str]:
    per_mode = summary.get("per_mode", {})
    modes = sorted(per_mode.keys())
    outputs = []

    def metric_values(metric: str) -> List[float]:
        return [float(per_mode[mode].get(metric, 0.0)) for mode in modes]

    plots = [
        ("07_avg_latency_bar.png", "avg_latency_ms", "Average latency (ms)", "Average Latency by RAG Mode"),
        ("08_avg_unique_sources_bar.png", "avg_unique_sources", "Average unique sources", "Average Source Coverage by RAG Mode"),
        ("09_avg_context_chars_bar.png", "avg_context_chars", "Average context chars", "Average Context Size by RAG Mode"),
        ("10_avg_relevance_bar.png", "avg_relevance_score", "Average relevance score", "Average Relevance by RAG Mode"),
        ("11_avg_groundedness_bar.png", "avg_groundedness_score", "Average groundedness score", "Average Groundedness by RAG Mode"),
        ("12_avg_completeness_bar.png", "avg_completeness_score", "Average completeness score", "Average Completeness by RAG Mode"),
    ]

    for filename, metric, ylabel, title in plots:
        values = metric_values(metric)
        if not any(values):
            continue
        plt.figure()
        plt.bar(modes, values)
        plt.ylabel(ylabel)
        plt.title(title)
        outputs.append(savefig(outdir, filename))

    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", default=os.path.join(BASE_DIR, "rag_mode_outputs"))
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    raw_path = os.path.join(args.outdir, "raw_mode_results.json")
    summary_path = os.path.join(args.outdir, "summary_mode_metrics.json")
    plot_dir = os.path.join(args.outdir, "plots")

    if not os.path.exists(raw_path):
        raise FileNotFoundError(f"raw_mode_results.json not found: {raw_path}")
    if not os.path.exists(summary_path):
        raise FileNotFoundError(f"summary_mode_metrics.json not found: {summary_path}")

    ensure_dir(plot_dir)
    raw = load_json(raw_path)
    summary = load_json(summary_path)
    modes = list(summary.get("modes", [])) or ["vector", "pageIndex"]

    per = extract_per_query_arrays(raw, modes)
    outputs: List[str] = []
    outputs += plot_boxplots(per, plot_dir)
    outputs += plot_pairwise_heatmaps(raw, modes, plot_dir, k=args.topk)
    outputs += plot_summary_bars(summary, plot_dir)

    print("\n✅ Saved RAG mode plots:")
    for path in outputs:
        print(" -", path)


if __name__ == "__main__":
    main()
