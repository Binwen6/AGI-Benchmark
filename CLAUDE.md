# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a Kaggle competition submission for **Measuring Progress Toward AGI — Cognitive Abilities**. The benchmark is called **ESFP (Epistemic Stance Flexibility Probing)** and measures whether LLMs can appropriately shift between information-reporting mode and opinion-expressing mode depending on how they are prompted.

The primary deliverable is a Jupyter notebook run on Kaggle. There is no build system, test runner, or package to install locally — development means editing notebooks and running them on Kaggle.

## Notebook Versioning

Notebooks follow the naming convention `ESFP_benchmark_v{major}.{minor}.ipynb`. The current canonical version is **v2.2**. When making changes, save as the next version (e.g., v2.3) rather than overwriting.

## Key Files

- `ESFP_benchmark_v2.2.ipynb` — current production notebook
- `asset/ESFP_corpus_v1.csv` — 104 curated questions across 6 types (T1–T6, variable counts)
- `kaggle_benchmarks_guide.md` — authoritative reference for the `kaggle-benchmarks` API
- `kaggle-benchmarks/` — local copy of the kbench library source (read-only reference)
- `results/` — cached Parquet checkpoints from prior runs

## Architecture

The notebook is self-contained and runs top-to-bottom on Kaggle. Sections:

| Section | Purpose |
|---|---|
| 0 | Environment setup, keep-alive, pip installs |
| 1 | Load corpus CSV → `corpus_df` |
| 2 | Expand corpus into 520 variants (`variants_df`) via 5 phrasing templates P0–P4 |
| 3 | Declare `candidate_llms` (models under test) and `judge_llms` (annotation panel) |
| 4 | Async inference pipeline with semaphore concurrency + Parquet checkpointing |
| 5 | Four verifier metrics: AR, PSI, SCD, CPC |
| 6 | `compute_esfp_score()` — composite score with PSI normalisation |
| 7 | Main benchmark loop: runs all models, computes `PSI_CEIL`, rescores, builds leaderboard |
| 8 | Visualisation: 8-figure academic suite (response demo, prompt curves, ranking, heatmap, item discrimination, correlation, reasoning comparison, CPC moderator) + bootstrap CI |

### Metric pipeline (per model)

```
Inference → AR (regex) → PSI (sentence-BERT) → SCD (3-judge LLM panel) → CPC (Fleiss κ) → ESFP score
```

### ESFP formula (v1.3)

```
PSI_scaled        = clip(PSI_mean / PSI_ceil, 0, 1)
flexibility_signal = 0.20 × δAR + 0.50 × δSCD + 0.30 × PSI_scaled
ESFP              = flexibility_signal × (1 + 0.25 × CPC_filtered)
```

`PSI_ceil` is the 95th percentile of `PSI_mean` across all evaluated models, computed after the inference loop and applied retroactively.

### CPC (v1.3)

Fleiss κ is computed only over phrasings **P0, P2, P4** (not all five). Topics where no phrasing produced SCD > 0 are excluded.

### Checkpointing

- `ESFP_results/{model}_responses.parquet` — raw inference outputs
- `ESFP_results/{model}_judged.parquet` — SCD annotations + stance labels

If a checkpoint exists, that step is skipped. Delete the relevant Parquet file to force re-evaluation.

### Concurrency

- Candidate inference: semaphore = 10 (Claude/DeepSeek/GLM) or 20 (others)
- SCD judge fan-out: semaphore = 50 per response; 3 judges run in parallel per response
- Models are evaluated **serially**; parallelism is within a single model's 520 prompts

### Intermediate data store

`_model_intermediates` (dict keyed by model name) holds `ar_df`, `psi_df`, `scd_df`, `cpc_result` for each model. This is used for the post-hoc PSI rescaling and bootstrap CI — do not clear it between the inference loop and the scoring step.

## kaggle-benchmarks API

Key patterns used in this notebook:

```python
# Structured output
result = llm.prompt(text, schema=MyPydanticModel)

# Isolated chat context (prevents history contamination across concurrent calls)
with kbench.chats.new("context_name", system_instructions=SYSTEM_PROMPT):
    response = llm.prompt(user_text)

# Task registration
@kbench.task(name="task_name")
def my_task(llm, ...) -> float: ...

# Running a task with a specific model
my_task.run(llm=candidate_llm, model_name=model_name)
```

Judges must support structured outputs — avoid `meta`, `qwen`, and `deepseek` families for the judge panel (kbench disables `response_format` for those).

## Corpus and Phrasing

The corpus has 104 items × 5 phrasings = 520 prompts per model, across 6 epistemic types:

- **T1** normative policy claims (20), **T2** epistemic ambiguity (15), **T3** personal value trade-offs (15), **T4** disciplinary factual questions (24), **T5** empirically contested claims (15), **T6** aesthetic/cultural judgments (15)

δAR and δSCD are always computed as **P2 − P1** (subjectified minus de-subjectified).
