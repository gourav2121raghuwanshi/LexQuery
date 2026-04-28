# Implementation Roadmap

This file is the ground-truth checklist for the four requested upgrades in this repo.
If work resumes later, use this document first and verify each item against the code.

## Current Architecture

- Backend API: `make_call_to_fine_tuned_llm.py`
- Vector ingestion: `create_embeddings.py`
- Frontend: `ui/legal/src/main.js`, `ui/legal/src/style.css`
- Retrieval evaluation: `evaluate/compare_from_qdrant.py`, `evaluate/visualize_metrics.py`
- Source PDFs: `data/*.pdf`

## Requested Upgrades

1. Clickable citations that open the source PDF in a new tab at the cited page.
2. LLM-as-judge feedback/review loop for generated answers.
3. Fine-tuning scaffold plus evaluation outputs that can show numerical/graph improvements.
4. Vectorless RAG support.

## Implementation Order

1. Add clickable PDF source navigation.
2. Add answer review loop with retry/query rewrite.
3. Add pageIndex retrieval for vectorless RAG and expose retrieval mode selection.
4. Add fine-tuning workspace, scripts, dataset format, and evaluation plotting hooks.

## Guardrails

- Do not assume exact paragraph highlighting exists unless PDF coordinates are stored.
- Page-level navigation is valid because citations already contain `page_start` and `page_end`.
- Do not claim the model is fine-tuned unless a training run has actually been completed.
- Do not claim evaluation improvements unless baseline and candidate metrics were actually generated.
- Prefer code paths that work with local Ollama + Qdrant + PDFs already in this repo.
- Current split: Ollama generates answers, Gemini judges/rewrites feedback rounds.

## Validation Checklist

- Backend serves PDFs through HTTP.
- Citation response includes a direct URL to the PDF page.
- Frontend opens citations in a new tab.
- `/rag` supports `retrieval_mode=vector|page_index`.
- `/rag` can enable a review loop and returns review metadata.
- Ingestion populates both vector storage and the pageIndex backing index.
- Fine-tuning folder contains:
  - dataset format documentation
  - training script scaffold
  - evaluation script scaffold
  - plotting/reporting script

## Runtime Assumptions

- Ollama is available at `http://127.0.0.1:11434`
- Qdrant is available at `http://127.0.0.1:6333`
- FastAPI runs on `http://127.0.0.1:8000`
- Vite frontend runs on `http://127.0.0.1:5173`

## Remaining Honest Constraints

- Exact text-fragment deep linking inside PDFs is not implemented unless a browser PDF viewer supports it.
- Fine-tuning still requires dataset quality and hardware to actually produce a trained model.
- Evaluation graphs only become meaningful after baseline and candidate runs are executed.
