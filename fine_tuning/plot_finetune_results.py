from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", required=True, help="Output JSON from evaluate_models.py")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    baseline = report["baseline_metrics"]
    candidate = report["candidate_metrics"]
    labels = ["relevance", "faithfulness", "correctness"]
    baseline_vals = [baseline[label] for label in labels]
    candidate_vals = [candidate[label] for label in labels]
    improvements = [candidate[label] - baseline[label] for label in labels]

    plt.figure()
    x = range(len(labels))
    width = 0.35
    plt.bar([i - width / 2 for i in x], baseline_vals, width=width, label="baseline")
    plt.bar([i + width / 2 for i in x], candidate_vals, width=width, label="candidate")
    plt.xticks(list(x), labels)
    plt.ylim(0, 5)
    plt.ylabel("Average judge score")
    plt.title("Baseline vs Candidate Model")
    plt.legend()
    plt.tight_layout()
    plt.savefig(outdir / "baseline_vs_candidate_scores.png", dpi=220)
    plt.close()

    plt.figure()
    plt.bar(labels, improvements)
    plt.axhline(0, color="black", linewidth=1)
    plt.ylabel("Candidate - Baseline")
    plt.title("Fine-Tuning Improvement by Metric")
    plt.tight_layout()
    plt.savefig(outdir / "score_improvements.png", dpi=220)
    plt.close()

    summary_md = outdir / "summary.md"
    summary_md.write_text(
        "\n".join(
            [
                "# Fine-Tuning Evaluation Summary",
                "",
                f"- Baseline model: `{report['baseline_model']}`",
                f"- Candidate model: `{report['candidate_model']}`",
                "",
                "| Metric | Baseline | Candidate | Improvement |",
                "| --- | ---: | ---: | ---: |",
                *[
                    f"| {label} | {baseline[label]:.3f} | {candidate[label]:.3f} | {improvements[idx]:.3f} |"
                    for idx, label in enumerate(labels)
                ],
            ]
        ),
        encoding="utf-8",
    )
    print(f"Saved plots and summary to {outdir}")


if __name__ == "__main__":
    main()
