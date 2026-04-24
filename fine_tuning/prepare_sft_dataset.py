from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path
from typing import Iterable, List, Tuple

from datasets import load_dataset


def build_prompt(question: str, context: str = "") -> str:
    prompt = (
        "You are a grounded Indian legal assistant.\n"
        "Answer carefully, avoid inventing facts, and ask for missing details when needed.\n\n"
    )
    if context.strip():
        prompt += f"Context:\n{context.strip()}\n\n"
    prompt += f"Question:\n{question.strip()}\n"
    return prompt


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def parse_gemma_style(text: str) -> List[Tuple[str, str]]:
    pattern = re.compile(
        r"<start_of_turn>(user|model)\s*(.*?)\s*<end_of_turn>",
        flags=re.DOTALL | re.IGNORECASE,
    )
    turns = [(role.lower(), normalize_space(content)) for role, content in pattern.findall(text or "")]
    pairs: List[Tuple[str, str]] = []
    current_user: str | None = None

    for role, content in turns:
        if not content:
            continue
        if role == "user":
            current_user = content
        elif role == "model" and current_user:
            pairs.append((current_user, content))
            current_user = None

    return pairs


def parse_inst_style(text: str) -> List[Tuple[str, str]]:
    cleaned = (text or "").replace("</s>", " ")
    cleaned = re.sub(r"<s>\s*\[INST\]\s*", " [INST] ", cleaned)
    segments = [segment.strip() for segment in cleaned.split("[INST]") if segment.strip()]
    pairs: List[Tuple[str, str]] = []

    for segment in segments:
        if "]" not in segment:
            continue
        assistant, rest = segment.split("]", 1)
        assistant = normalize_space(assistant)
        user = normalize_space(rest)
        if assistant and user:
            pairs.append((user, assistant))

    return pairs


def row_to_examples(row: dict, text_field: str) -> List[dict]:
    raw_text = str(row.get(text_field, "") or "")
    pairs = parse_gemma_style(raw_text)
    if not pairs:
        pairs = parse_inst_style(raw_text)

    examples = []
    for user_text, assistant_text in pairs:
        examples.append(
            {
                "prompt": build_prompt(user_text),
                "completion": assistant_text,
            }
        )
    return examples


def write_jsonl(path: Path, rows: Iterable[dict]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as dst:
        for row in rows:
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def convert_local_jsonl(input_path: Path) -> List[dict]:
    out = []
    with input_path.open("r", encoding="utf-8") as src:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            question = str(row["question"]).strip()
            context = str(row.get("context", "")).strip()
            answer = str(row["answer"]).strip()
            out.append(
                {
                    "prompt": build_prompt(question, context),
                    "completion": answer,
                }
            )
    return out


def load_hf_examples(dataset_name: str, split: str, text_field: str) -> List[dict]:
    dataset = load_dataset(dataset_name, split=split)
    out: List[dict] = []
    for row in dataset:
        out.extend(row_to_examples(row, text_field=text_field))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Local JSONL with question/context/answer")
    parser.add_argument("--output-dir", required=True, help="Directory to write train/eval JSONL")
    parser.add_argument("--hf-dataset", help="Hugging Face dataset name, e.g. shadow228825/Legal_Advisior_Conversation_With_Client_India")
    parser.add_argument("--hf-split", default="train", help="Dataset split to use when loading from Hugging Face")
    parser.add_argument("--text-field", default="text", help="Text field name for Hugging Face chat datasets")
    parser.add_argument("--eval-ratio", type=float, default=0.1, help="Fraction of examples to reserve for eval")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not args.input and not args.hf_dataset:
        raise ValueError("Provide either --input or --hf-dataset")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.hf_dataset:
        rows = load_hf_examples(args.hf_dataset, split=args.hf_split, text_field=args.text_field)
    else:
        rows = convert_local_jsonl(Path(args.input))

    if not rows:
        raise ValueError("No training examples were produced from the input source")

    rng = random.Random(args.seed)
    rng.shuffle(rows)

    eval_count = max(1, int(len(rows) * args.eval_ratio))
    eval_rows = rows[:eval_count]
    train_rows = rows[eval_count:]
    if not train_rows:
        raise ValueError("Not enough rows left for training after eval split")

    train_path = output_dir / "train_sft.jsonl"
    eval_path = output_dir / "eval_sft.jsonl"

    train_count = write_jsonl(train_path, train_rows)
    eval_count = write_jsonl(eval_path, eval_rows)

    print(f"Prepared {train_count} train rows at {train_path}")
    print(f"Prepared {eval_count} eval rows at {eval_path}")


if __name__ == "__main__":
    main()
