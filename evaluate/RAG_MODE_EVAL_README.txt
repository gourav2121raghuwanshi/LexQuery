RAG MODE EVALUATION
===================

This evaluation compares:

- vector RAG
- pageIndex RAG

Files:

- compare_rag_modes.py
  Runs the comparisons and stores numeric outputs

- visualize_rag_modes.py
  Generates plots from those outputs


RUN STEPS
---------

1. Make sure the backend dependencies/services are available:
- Ollama
- Qdrant
- lexical_chunks.db already built

2. Run the numeric comparison:

python3 evaluate/compare_rag_modes.py --topk 5 --outdir evaluate/rag_mode_outputs

3. If you also want answer-quality scoring using the judge model:

python3 evaluate/compare_rag_modes.py --topk 5 --with-answers --outdir evaluate/rag_mode_outputs

4. Generate the plots:

python3 evaluate/visualize_rag_modes.py --outdir evaluate/rag_mode_outputs --topk 5


OUTPUTS
-------

Numeric files:

- evaluate/rag_mode_outputs/raw_mode_results.json
- evaluate/rag_mode_outputs/summary_mode_metrics.json
- evaluate/rag_mode_outputs/mode_results_flat.csv

Plot files:

- evaluate/rag_mode_outputs/plots/*.png


WHAT YOU CAN PRESENT
--------------------

- Average latency by mode
- Average source coverage by mode
- Average context size by mode
- Pairwise overlap/Jaccard between retrieved docs
- If --with-answers is enabled:
  - average relevance by mode
- average groundedness by mode
- average completeness by mode
