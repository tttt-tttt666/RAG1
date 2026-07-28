---
name: rag-bilingual-evaluator
description: Evaluate a local bilingual RAG application with paired Chinese and English questions, detailed-answer intent checks, retrieved-source relevance, and Top-3 cross-language consistency. Use after RAG code, prompts, chunks, embeddings, ranking, or answer logic changes, and when establishing or comparing regression baselines.
---

# RAG Bilingual Evaluator

Run a deterministic 10-pair/20-query regression suite and report one composite score.

## Workflow

1. Confirm the target project contains `app.py`, `index/ankle_sprain/chunks.jsonl`,
   and `index/ankle_sprain/embeddings/`.
2. Read `references/test_cases.json` only when reviewing or changing the golden set.
3. Run:

   ```bash
   python scripts/evaluate_rag.py \
     --project /absolute/path/to/project \
     --output /absolute/path/to/project/evaluation
   ```

4. Inspect both `latest_report.md` and `latest_results.json`.
5. Report:
   - answer adaptation score;
   - source relevance score;
   - bilingual Top-3 consistency score;
   - overall score;
   - failed cases and concrete retrieved chunk IDs.
6. Treat the suite as a regression guard, not a medical-validity certification.
   Require clinician review for factual correctness and safety.

## Scoring

- Answer adaptation (40%): expected answer intent is present, no unrelated
  extra intent is emitted, and required answer concepts are covered.
- Source relevance (35%): each Top-3 passage contains at least one
  topic-specific evidence term.
- Bilingual consistency (25%): average set overlap of Chinese/English Top-3,
  with an additional exact-order indicator shown separately.
- Overall score: weighted mean of the three dimensions.

Keep scoring deterministic and offline after the embedding model is cached.
Do not translate source documents.

## GitHub Automation

Use `.github/workflows/rag-evaluation.yml` in the target repository to run the
suite after every push. Upload JSON and Markdown reports as workflow artifacts
and write the summary to the GitHub Actions job summary. Do not auto-commit
reports from the workflow because that can create a push loop.
