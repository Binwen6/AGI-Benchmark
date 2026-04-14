import argparse
import asyncio
import pandas as pd
from typing import List, Dict, Any
from pathlib import Path
from dotenv import load_dotenv

# Automatically load environment variables from .env
# This includes OPENAI_API_KEY, ANTHROPIC_API_KEY, GEMINI_API_KEY, etc.
load_dotenv()

from esfp_benchmark.generator import load_corpus, generate_variants
from esfp_benchmark.client import LLMClient
from esfp_benchmark.metrics import (
    compute_ar_vectors,
    compute_psi_scores,
    compute_esfp_score,
    compute_cpc,
    run_bootstrap_all_models,
)
from esfp_benchmark.evaluator import (
    load_model_results,
    save_model_results,
    load_judged_results,
    save_judged_results,
    run_inference_for_model,
    compute_scd_for_model,
    extract_stances_for_model,
)
from esfp_benchmark.visualize import generate_all_figures
from esfp_benchmark.config import RESULTS_DIR


async def benchmark_model(
    model_name: str,
    judge_names: List[str],
    corpus_path: str = None
) -> Dict[str, Any]:
    print(f"\n{'='*65}\n  Evaluating candidate: {model_name}\n{'='*65}")

    # Prepare data and clients
    corpus_df = load_corpus(corpus_path)
    variants_df = generate_variants(corpus_df)

    candidate_llm = LLMClient(model_name)
    judge_llms = [LLMClient(j) for j in judge_names]

    # ── Step 1: Inference ─────────────────────────────────────────
    responses_df = load_model_results(model_name)
    if responses_df is None:
        print("  [1/5] Running inference ...")
        responses_df = await run_inference_for_model(model_name, candidate_llm, variants_df)
        save_model_results(model_name, responses_df)
    else:
        print("  [1/5] Inference: loaded from checkpoint.")

    # ── Step 2: Attribution Rate (AR) ─────────────────────────────
    print("  [2/5] Computing AR vectors ...")
    ar_df = compute_ar_vectors(responses_df)

    # ── Step 3: Phrasing Sensitivity Index (PSI) ──────────────────
    print("  [3/5] Computing PSI scores ...")
    psi_df = compute_psi_scores(responses_df)

    # ── Step 4: Judge Caching (SCD & CPC) ─────────────────────────
    judged_df = load_judged_results(model_name)
    if judged_df is None:
        print(f"  [4a/5] Running SCD annotation ({len(judge_llms)}-judge panel) ...")
        scd_df = await compute_scd_for_model(responses_df, judge_llms)

        print("  [4b/5] Extracting stances for CPC ...")
        stance_df = await extract_stances_for_model(scd_df, judge_llms)
        
        save_judged_results(model_name, stance_df)
    else:
        print("  [4/5] SCD & Stance extraction: loaded from checkpoint.")
        scd_df = judged_df
        stance_df = judged_df

    cpc_result = compute_cpc(stance_df, scd_df)

    # Note: final scoring relies on PSI_ceil which is global. We will return intermediates
    # and compute the final score in the orchestrator.
    return {
        "model_name": model_name,
        "ar_df": ar_df,
        "psi_df": psi_df,
        "scd_df": scd_df,
        "cpc_result": cpc_result
    }

async def async_main():
    parser = argparse.ArgumentParser(description="ESFP Benchmark Runner (LiteLLM version)")
    parser.add_argument("--models", nargs='+', required=True, help="Litellm strings for candidate models to evaluate (e.g. openai/gpt-4o anthropic/claude-3-haiku)")
    parser.add_argument("--judges", nargs='+', default=["gemini/gemini-2.0-flash-lite", "qwen/qwen-turbo"], help="Litellm strings for judge models (default uses fast/cheap models)")
    parser.add_argument("--corpus", type=str, default="/Users/binwen6/project/DeepMind/AGI Benchmark/asset/ESFP_corpus_v1.csv", help="Path to ESFP_corpus_v1.csv")

    args = parser.parse_args()
    
    print(f"Candidate Models: {args.models}")
    print(f"Judge Models: {args.judges}")

    model_intermediates: Dict[str, Dict] = {}

    for model in args.models:
        intermediates = await benchmark_model(model, args.judges, args.corpus)
        model_intermediates[model] = intermediates

    print("\n" + "="*65)
    print("  Aggregating ESFP Scores")
    print("="*65)

    # Compute global PSI Ceil (95th percentile of PSI_mean across models)
    all_psi_means = []
    for inter in model_intermediates.values():
        val = inter["psi_df"]["PSI"].dropna().mean()
        if pd.notna(val):
            all_psi_means.append(val)
    
    psi_ceil = float(np.percentile(all_psi_means, 95)) if len(all_psi_means) > 0 else 1.0

    all_scores = []
    for model_name, inter in model_intermediates.items():
        score = compute_esfp_score(
            model_name,
            inter["ar_df"],
            inter["psi_df"],
            inter["scd_df"],
            inter["cpc_result"],
            psi_ceil=psi_ceil
        )
        all_scores.append(score)

        print(
            f"  {model_name}:\n"
            f"    ESFP = {score['ESFP']:.5f}  |  "
            f"flex_signal = {score['flexibility_signal']:.4f}  |  "
            f"dAR = {score['delta_AR']:.3f}  "
            f"PSI = {score['PSI_mean']:.3f}  "
            f"dSCD = {score['delta_SCD']:.3f}  "
            f"CPC_kappa = {score['CPC_kappa']:.3f}"
        )

    scores_df = pd.DataFrame(all_scores)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    scores_df.to_csv(RESULTS_DIR / "final_scores.csv", index=False)
    
    # Run Bootstrap CI
    # In earlier versions run_bootstrap_all_models returns the boot_df
    # We will pass this to the graph generation
    from esfp_benchmark.metrics import bootstrap_esfp_ci
    print("\nBootstrap resampling (1000 iterations) ...")
    boot_results = []
    for model_name, inter in model_intermediates.items():
        ci = bootstrap_esfp_ci(
            model_name,
            inter["ar_df"],
            inter["psi_df"],
            inter["scd_df"],
            inter["cpc_result"],
            psi_ceil=psi_ceil,
            n_boot=1000
        )
        boot_results.append(ci)
    
    boot_df = pd.DataFrame(boot_results)
    boot_df.to_csv(RESULTS_DIR / "ESFP_bootstrap_ci.csv", index=False)

    # Generate Figures
    generate_all_figures(scores_df, boot_df, model_intermediates)
    
    print("\n  [DONE] Results and figures populated in:", RESULTS_DIR)

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
