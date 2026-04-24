# Fine-Tuning Workspace

This folder is the training/evaluation scaffold for moving from the current local Ollama model to a legal-domain tuned model.

## What is included

- `prepare_sft_dataset.py`
  Converts curated Q/A pairs or supported Hugging Face chat datasets into instruction-tuning JSONL.
- `train_lora.py`
  LoRA/QLoRA training scaffold for a Hugging Face causal LM.
- `evaluate_models.py`
  Runs a baseline model and a candidate model on the same eval set and scores them.
- `plot_finetune_results.py`
  Produces graphs and a summary table from the evaluation output.
- `requirements-finetune.txt`
  Optional dependencies for the training/evaluation workflow.

## Honest status

- The current repo does not yet contain a labeled SFT dataset.
- No fine-tuning run has been executed by this scaffold yet.
- Improvement charts become meaningful only after both baseline and candidate evaluations are run.

## Suggested flow

1. Prepare `train_sft.jsonl` and `eval_sft.jsonl`.
2. Run `train_lora.py` on a GPU machine.
3. Export or serve the tuned model.
4. Run `evaluate_models.py` on baseline vs tuned model.
5. Run `plot_finetune_results.py` to produce graphs and summary metrics.

## Example dataset row

```json
{"question":"What is Article 14?","context":"Article 14 of the Constitution guarantees equality before law...","answer":"Article 14 guarantees equality before the law and equal protection of the laws."}
```

## Direct Hugging Face dataset usage

The prep script now supports chat-style Hugging Face datasets directly.

Larger candidate:

```bash
python /Users/gourav/Developer/python/GenAi/BTP/fine_tuning/prepare_sft_dataset.py \
  --hf-dataset shadow228825/Legal_Advisior_Conversation_With_Client_India \
  --output-dir /Users/gourav/Developer/python/GenAi/BTP/fine_tuning/data/shadow_legal
```

Smaller pilot run:

```bash
python /Users/gourav/Developer/python/GenAi/BTP/fine_tuning/prepare_sft_dataset.py \
  --hf-dataset jaimadhukar/legal-advisor-gemma-chat \
  --output-dir /Users/gourav/Developer/python/GenAi/BTP/fine_tuning/data/gemma_legal
```

Those commands create:

- `train_sft.jsonl`
- `eval_sft.jsonl`

inside the chosen output directory.
