import asyncio
import re
import pandas as pd
from typing import List, Dict, Any, Optional, Literal
from pathlib import Path
from pydantic import BaseModel, Field
from collections import Counter
import tqdm.asyncio

from esfp_benchmark.client import LLMClient
from esfp_benchmark.config import (
    RESPONSE_SYSTEM_PROMPT,
    INFERENCE_SEMAPHORE_LIMIT,
    STRICT_SEMAPHORE_LIMIT,
    SCD_SEMAPHORE_LIMIT,
    SCD_MAX_SENTENCES,
    SCD_JUDGE_SYSTEM_PROMPT,
    STANCE_EXTRACTION_SYSTEM_PROMPT,
    RESULTS_DIR
)

class SentenceLabels(BaseModel):
    labels: List[Literal["A", "B", "C"]] = Field(
        description=(
            "Ordered list of sentence-level epistemic classifications: "
            "A (fact/third-party), B (model stance), or C (filler). "
            "One entry per sentence, in input order."
        )
    )

class StanceLabel(BaseModel):
    stance: Literal["positive", "negative", "neutral", "no_stance"] = Field(
        description="Overall evaluative stance of the text toward its topic."
    )


def _ckpt_path(model_name: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_name)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    return RESULTS_DIR / f"{safe}_responses.parquet"

def save_model_results(model_name: str, df: pd.DataFrame) -> None:
    path = _ckpt_path(model_name)
    df.to_parquet(path, index=False)
    print(f"  [checkpoint saved] {path.name}")

def load_model_results(model_name: str) -> Optional[pd.DataFrame]:
    path = _ckpt_path(model_name)
    if path.exists():
        print(f"  [checkpoint found] Resuming from {path.name}")
        return pd.read_parquet(path)
    return None

def _judge_ckpt_path(model_name: str) -> Path:
    safe = re.sub(r"[^a-zA-Z0-9_\-]", "_", model_name)
    RESULTS_DIR.mkdir(exist_ok=True, parents=True)
    return RESULTS_DIR / f"{safe}_judged.parquet"

def save_judged_results(model_name: str, df: pd.DataFrame) -> None:
    path = _judge_ckpt_path(model_name)
    df.to_parquet(path, index=False)
    print(f"  [judge checkpoint saved] {path.name}")

def load_judged_results(model_name: str) -> Optional[pd.DataFrame]:
    path = _judge_ckpt_path(model_name)
    if path.exists():
        print(f"  [judge checkpoint found] Resuming SCD & Stances from {path.name}")
        return pd.read_parquet(path)
    return None


async def _inference_task(llm: LLMClient, prompt: str, sem: asyncio.Semaphore) -> str:
    async with sem:
        return await llm.async_prompt(prompt, RESPONSE_SYSTEM_PROMPT)

async def run_inference_for_model(
    model_name: str,
    llm: LLMClient,
    variants_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run every variant prompt for one candidate model with a progress bar.
    """
    if any(k in model_name.lower() for k in ["claude", "deepseek", "glm"]):
        limit = STRICT_SEMAPHORE_LIMIT
    else:
        limit = INFERENCE_SEMAPHORE_LIMIT
        
    semaphore = asyncio.Semaphore(limit)
    tasks = [
        _inference_task(llm, row["prompt_text"], semaphore)
        for _, row in variants_df.iterrows()
    ]
    
    print(f"  Dispatching {len(tasks)} calls (semaphore limit = {limit}) ...")
    
    responses = await tqdm.asyncio.tqdm.gather(*tasks, desc=f"Inference ({model_name.split('/')[-1]})")
    
    result_df = variants_df.copy()
    result_df["response"] = list(responses)
    result_df["model"] = model_name
    return result_df


# -----------------------------------------------------------------------
# Judge Execution
# -----------------------------------------------------------------------

def split_into_sentences(text: str) -> List[str]:
    """
    Split a response into sentences on '.', '?', '!' boundaries.
    """
    raw = re.split(r"(?<=[.!?])\s+", str(text).strip())
    sentences = [s.strip() for s in raw if len(s.strip()) >= 12]
    return sentences[:SCD_MAX_SENTENCES]

async def _judge_sentences_single(
    judge_llm: LLMClient,
    sentences: List[str],
    semaphore: asyncio.Semaphore,
) -> List[str]:
    numbered = "\n".join(f"{i + 1}. {s}" for i, s in enumerate(sentences))
    prompt_text = (
        f"Classify each of the following {len(sentences)} sentences.\n\n"
        f"{numbered}\n\n"
        f"Return a JSON object with key 'labels' containing exactly "
        f"{len(sentences)} strings (A, B, or C), one per sentence, in input order."
    )
    
    async with semaphore:
        result = await judge_llm.structured_prompt(prompt_text, SCD_JUDGE_SYSTEM_PROMPT, SentenceLabels)

    labels = result.labels if (result and hasattr(result, 'labels')) else []
    return (labels + ["C"] * len(sentences))[: len(sentences)]

async def _annotate_single_response(
    response: str,
    judge_llms: List[LLMClient],
    semaphore: asyncio.Semaphore,
) -> Dict[str, Any]:
    sentences = split_into_sentences(response)
    if not sentences:
        return {"sentences": [], "majority_labels": [], "SCD": float("nan")}

    judge_tasks = [
        _judge_sentences_single(llm, sentences, semaphore)
        for llm in judge_llms
    ]
    all_labels = await asyncio.gather(*judge_tasks)

    majority_labels = []
    for i in range(len(sentences)):
        votes = [
            all_labels[j][i]
            for j in range(len(all_labels))
            if i < len(all_labels[j])
        ]
        if not votes:
            majority_labels.append("C")
            continue
            
        winner, winner_count = Counter(votes).most_common(1)[0]
        majority_labels.append(winner if winner_count >= (len(judge_llms)/2.0) else "C")

    b_count  = majority_labels.count("B")
    ab_count = majority_labels.count("A") + b_count
    scd = b_count / ab_count if ab_count > 0 else float("nan")

    return {"sentences": sentences, "majority_labels": majority_labels, "SCD": scd}


async def compute_scd_for_model(
    responses_df: pd.DataFrame,
    judge_llms: List[LLMClient],
) -> pd.DataFrame:
    semaphore = asyncio.Semaphore(SCD_SEMAPHORE_LIMIT)
    tasks = [
        _annotate_single_response(row["response"], judge_llms, semaphore)
        for _, row in responses_df.iterrows()
    ]
    
    annotations = await tqdm.asyncio.tqdm.gather(
        *tasks, desc=f"SCD Judging ({len(judge_llms)} Judges)", total=len(tasks)
    )
    
    result_df = responses_df.copy()
    result_df["SCD"] = [a["SCD"] for a in annotations]
    result_df["majority_labels"] = [a["majority_labels"] for a in annotations]
    return result_df


async def _extract_stance_single(
    response: str,
    judge_llm: LLMClient,
    semaphore: asyncio.Semaphore,
) -> str:
    prompt_text = f"Analyze the following text and return its overall stance:\n\n{response}"
    async with semaphore:
        result = await judge_llm.structured_prompt(
            prompt_text, 
            STANCE_EXTRACTION_SYSTEM_PROMPT, 
            StanceLabel
        )

    return result.stance if (result and hasattr(result, 'stance')) else "no_stance"

async def extract_stances_for_model(
    responses_df: pd.DataFrame,
    judge_llms: List[LLMClient],
) -> pd.DataFrame:
    primary_llm = judge_llms[0] # The primary judge does CPC
    semaphore = asyncio.Semaphore(SCD_SEMAPHORE_LIMIT)

    tasks = [
        _extract_stance_single(row["response"], primary_llm, semaphore)
        for _, row in responses_df.iterrows()
    ]

    stances = await tqdm.asyncio.tqdm.gather(
        *tasks, desc="CPC Stance Extraction", total=len(tasks)
    )

    result_df = responses_df.copy()
    result_df["stance"] = list(stances)
    return result_df
