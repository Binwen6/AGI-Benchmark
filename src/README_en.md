# ESFP Benchmark

The ESFP Benchmark is designed to measure the **Epistemic Stance Flexibility** of Large Language Models (LLMs) — that is, whether a model can appropriately shift its epistemic register between "objective information-reporting" and "subjective opinion-expressing" depending on the contextual framing of the prompt.

This project has been modularized and decoupled from Kaggle's execution environment using `litellm`. This allows you to seamlessly plug in standard API keys (such as OpenAI, Anthropic, Google, Zhipu, DeepSeek, or private models) and test them locally or on your own servers.

## Project Structure
```text
src/
├── esfp_benchmark/
│   ├── __init__.py
│   ├── config.py         # Core system Prompts and concurrency parameters
│   ├── client.py         # LLM client wrapper based on litellm, with auto-retry and System Prompt fallback strategies
│   ├── generator.py      # Loads the corpus and expands it into 5 distinct Phrasing variants (P0-P4)
│   ├── metrics.py        # Core scoring logic: Calculates AR, PSI (via SentenceTransformers), SCD, CPC Kappa
│   ├── evaluator.py      # Evaluation pipeline: Async inference, Checkpointing, and 3-Judge logic
│   ├── visualize.py      # Core plotting library generating Matplotlib/Seaborn charts (e.g. Fig 1 - Fig 4)
│   └── main.py           # Command-Line Interface (CLI) Master Orchestrator
├── .env.example          # Example template for API Environment Variables
├── requirements.txt      # Python Dependencies list
├── README.md             # Chinese documentation
└── README_EN.md          # This documentation
```

## Setup & Initialization

1. **Navigate to the workspace**  
Switch to this project source directory:
```bash
cd "/Users/binwen6/project/DeepMind/AGI Benchmark/src"
```

2. **Install Dependencies (Python 3.10+)**  
Install the required packages for running evaluations and generating visualizations:
```bash
pip install -r requirements.txt
```

3. **Configure API Credentials**   
Create your local `.env` file and input your relevant API keys.  
Thanks to the broad compatibility of `litellm`, the pipeline seamlessly authenticates to the respective providers without needing codebase modifications.
```bash
cp .env.example .env
```

## How to Run

You can run the end-to-end benchmark directly via the `esfp_benchmark.main` entry point:

```bash
python -m esfp_benchmark.main \
  --models "openai/gpt-4o" "anthropic/claude-3-haiku-20240307" \
  --judges "gemini/gemini-1.5-flash" "openai/gpt-3.5-turbo"
```

### CLI Arguments
* `--models`: (Required) A space-separated list of candidate models to evaluate. Format uses standard `litellm` strings (e.g., `openai/gpt-4o`). To use custom/proxy URLs, configure your environments according to the `litellm` docs.
* `--judges`: (Optional) The panel of models acting as evaluators for the `SCD` (sentence classification) and `CPC` (stance extraction) phases. A multiple-judge setup improves robustness. The default is a two-judge ensemble (`gemini/gemini-2.0-flash-lite`, `qwen/qwen-turbo`) which generally balances accuracy and cost.
* `--corpus`: (Optional) Path to the test questions corpus. By default, it will look up the directory tree to use `../asset/ESFP_corpus_v1.csv`.

### Crash Recovery & Checkpoints
**Never worry about transient API failures!** The pipeline implements high-frequency Checkpoint state-saving (using Parquet files). 
If you are evaluating 3 models and a network failure crashes the script at prompt 400 on the second model, simply **re-run the exact same command**. The system will scan your local drive for `.parquet` checkpoints and instantly resume from where it left off, saving you both time and API quotas.

## Output & Results
Upon reaching the 100% mark across the progress bars, the program will organize reports and charts under the `RESULTS_ESFP` directory:

* `RESULTS_ESFP/final_scores.csv` — Contains the aggregated master metrics table for all evaluated candidate models (ESFP composite score, δAR, δSCD, etc).
* `RESULTS_ESFP/ESFP_bootstrap_ci.csv` — Provides the 95% Confidence Intervals mapped via 1,000 item-level bootstrap resamples.
* `RESULTS_ESFP/figures/` — Contains high-resolution evaluation figures and analytic charts in both `.pdf` and `.png` format.
