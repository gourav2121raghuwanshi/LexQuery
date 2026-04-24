from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List

import httpx


OLLAMA_CHAT_URL = "http://127.0.0.1:11434/api/chat"


def call_model(client: httpx.Client, model: str, question: str, context: str) -> str:
    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "Answer the question using the given context. Be concise and grounded.",
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = client.post(OLLAMA_CHAT_URL, json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"]


def judge_answer(client: httpx.Client, judge_model: str, question: str, reference: str, answer: str) -> Dict[str, Any]:
    payload = {
        "model": judge_model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Return JSON only with keys relevance, faithfulness, correctness. "
                    "Use integer scores from 1 to 5."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{question}\n\nReference answer:\n{reference}\n\nCandidate answer:\n{answer}\n"
                ),
            },
        ],
        "stream": False,
        "options": {"temperature": 0.0},
    }
    response = client.post(OLLAMA_CHAT_URL, json=payload)
    response.raise_for_status()
    content = response.json()["message"]["content"]
    start = content.find("{")
    end = content.rfind("}")
    return json.loads(content[start : end + 1])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", required=True, help="JSONL with question/context/answer")
    parser.add_argument("--baseline-model", required=True)
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument("--judge-model", default="qwen2.5:7b")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    eval_rows = [json.loads(line) for line in Path(args.eval_file).read_text(encoding="utf-8").splitlines() if line.strip()]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    per_example: List[Dict[str, Any]] = []
    with httpx.Client(timeout=300) as client:
        for row in eval_rows:
            baseline_answer = call_model(client, args.baseline_model, row["question"], row["context"])
            candidate_answer = call_model(client, args.candidate_model, row["question"], row["context"])
            baseline_scores = judge_answer(client, args.judge_model, row["question"], row["answer"], baseline_answer)
            candidate_scores = judge_answer(client, args.judge_model, row["question"], row["answer"], candidate_answer)
            per_example.append(
                {
                    "question": row["question"],
                    "baseline_answer": baseline_answer,
                    "candidate_answer": candidate_answer,
                    "baseline_scores": baseline_scores,
                    "candidate_scores": candidate_scores,
                }
            )

    def aggregate(prefix: str) -> Dict[str, float]:
        return {
            "relevance": mean(float(item[f"{prefix}_scores"]["relevance"]) for item in per_example),
            "faithfulness": mean(float(item[f"{prefix}_scores"]["faithfulness"]) for item in per_example),
            "correctness": mean(float(item[f"{prefix}_scores"]["correctness"]) for item in per_example),
        }

    summary = {
        "baseline_model": args.baseline_model,
        "candidate_model": args.candidate_model,
        "baseline_metrics": aggregate("baseline"),
        "candidate_metrics": aggregate("candidate"),
        "examples": per_example,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Saved evaluation report to {output_path}")


if __name__ == "__main__":
    main()
