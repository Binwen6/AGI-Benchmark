import re
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from statsmodels.stats.inter_rater import aggregate_raters, fleiss_kappa
from sentence_transformers import SentenceTransformer

from esfp_benchmark.config import (
    SELF_ATTRIBUTION_PATTERNS,
    THIRD_PARTY_PATTERNS,
    STANCE_TO_INT
)

_SELF_RE  = re.compile("|".join(SELF_ATTRIBUTION_PATTERNS),  re.IGNORECASE)
_THIRD_RE = re.compile("|".join(THIRD_PARTY_PATTERNS),       re.IGNORECASE)

_ST_MODEL: Optional[SentenceTransformer] = None

def compute_ar(response: str) -> float:
    """
    Attribution Rate for a single response string.
    Returns NaN if no attribution markers of either type are found.
    """
    if not isinstance(response, str):
        return float("nan")
    n_self  = len(_SELF_RE.findall(response))
    n_third = len(_THIRD_RE.findall(response))
    total   = n_self + n_third
    return n_self / total if total > 0 else float("nan")

def compute_ar_vectors(responses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-topic AR values for all 5 phrasings and derive delta_AR.

    Returns
    -------
    pd.DataFrame indexed by item_id with columns:
        AR_P0 … AR_P4 : attribution rate under each phrasing
        delta_AR      : AR_P2 - AR_P1
    """
    df = responses_df.copy()
    df["AR"] = df["response"].apply(compute_ar)
    ar_pivot = df.pivot_table(
        index="item_id", columns="phrasing", values="AR", aggfunc="first"
    )
    ar_pivot.columns = [f"AR_{c}" for c in ar_pivot.columns]
    ar_pivot["delta_AR"] = (
        ar_pivot.get("AR_P2", pd.Series(dtype=float)) -
        ar_pivot.get("AR_P1", pd.Series(dtype=float))
    )
    return ar_pivot.reset_index()


def _get_st_model() -> SentenceTransformer:
    global _ST_MODEL
    if _ST_MODEL is None:
        _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
    return _ST_MODEL

def compute_psi_for_item(p0_response: str, p1_response: str) -> float:
    """
    PSI for a single topic.
    Encodes both responses with L2-normalised embeddings; PSI = 1 - dot product.
    """
    st   = _get_st_model()
    embs = st.encode([p0_response, p1_response], normalize_embeddings=True)
    cos_sim = float(np.dot(embs[0], embs[1]))
    return 1.0 - cos_sim

def compute_psi_scores(responses_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute per-topic PSI by comparing P0 and P1 responses.
    """
    p0 = (
        responses_df[responses_df["phrasing"] == "P0"]
        .set_index("item_id")["response"]
    )
    p1 = (
        responses_df[responses_df["phrasing"] == "P1"]
        .set_index("item_id")["response"]
    )
    common_ids = p0.index.intersection(p1.index)
    rows = [
        {"item_id": iid, "PSI": compute_psi_for_item(str(p0[iid]), str(p1[iid]))}
        for iid in common_ids
    ]
    return pd.DataFrame(rows)

def compute_cpc(
    stance_df: pd.DataFrame,
    scd_df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Compute Cross-Phrasing Consistency via Fleiss' kappa.
    """
    CPC_PHRASINGS = ["P0", "P2", "P4"]

    stance_cols = [c for c in stance_df.columns if c != "SCD"]
    merged = stance_df[stance_cols].merge(
        scd_df[["item_id", "phrasing", "SCD"]],
        on=["item_id", "phrasing"],
        how="left",
    )

    # Restrict to P0, P2, P4
    merged = merged[merged["phrasing"].isin(CPC_PHRASINGS)]

    topic_max_scd = merged.groupby("item_id")["SCD"].max()
    active_topics = topic_max_scd[topic_max_scd.fillna(0) > 0].index
    filtered      = merged[merged["item_id"].isin(active_topics)]

    if len(active_topics) < 2:
        return {"CPC_kappa": float("nan"), "CPC_filtered_n": int(len(active_topics))}

    pivot = filtered.pivot_table(
        index="item_id", columns="phrasing", values="stance", aggfunc="first"
    )
    rating_matrix = pivot.apply(
        lambda col: col.map(lambda x: STANCE_TO_INT.get(str(x), 3))
    ).values

    try:
        table, _ = aggregate_raters(rating_matrix, n_cat=4)
        kappa    = float(fleiss_kappa(table))
    except Exception as exc:
        print(f"  [CPC] Fleiss kappa computation failed: {exc}")
        kappa = float("nan")

    return {"CPC_kappa": kappa, "CPC_filtered_n": int(len(active_topics))}

def _delta_scd(scd_df: pd.DataFrame) -> float:
    """delta_SCD = mean(SCD_P2) - mean(SCD_P1) across all topics."""
    p1 = scd_df[scd_df["phrasing"] == "P1"]["SCD"].dropna().mean()
    p2 = scd_df[scd_df["phrasing"] == "P2"]["SCD"].dropna().mean()
    return float(p2 - p1) if (pd.notna(p1) and pd.notna(p2)) else float("nan")

def compute_esfp_score(
    model_name: str,
    ar_df: pd.DataFrame,
    psi_df: pd.DataFrame,
    scd_df: pd.DataFrame,
    cpc_result: Dict[str, Any],
    psi_ceil: float = 1.0,
) -> Dict[str, Any]:
    """
    Aggregate AR, PSI, SCD, and CPC outputs into the ESFP composite score.
    """
    delta_ar  = float(ar_df["delta_AR"].dropna().mean())
    psi_mean  = float(psi_df["PSI"].dropna().mean())
    delta_scd = _delta_scd(scd_df)
    cpc_kappa = cpc_result.get("CPC_kappa", float("nan"))
    cpc_n     = cpc_result.get("CPC_filtered_n", 0)

    cpc_clamped = max(0.0, cpc_kappa) if pd.notna(cpc_kappa) else 0.0

    # Normalise PSI by cross-model 95th percentile, clip to [0, 1]
    psi_scaled = float(np.clip(psi_mean / psi_ceil, 0.0, 1.0)) if psi_ceil > 0 else psi_mean

    if any(np.isnan(v) for v in [delta_ar, psi_mean, delta_scd]):
        flexibility_signal = float("nan")
        ESFP               = float("nan")
    else:
        flexibility_signal = (
            0.20 * delta_ar
            + 0.50 * delta_scd
            + 0.30 * psi_scaled
        )
        ESFP = flexibility_signal * (1.0 + 0.25 * cpc_clamped)

    def _r(v: float, d: int = 5) -> float:
        return round(v, d) if pd.notna(v) else float("nan")

    return {
        "model":              model_name,
        "ESFP":               _r(ESFP),
        "flexibility_signal": _r(flexibility_signal),
        "delta_AR":           _r(delta_ar,  4),
        "PSI_mean":           _r(psi_mean,  4),
        "PSI_scaled":         _r(psi_scaled, 4),
        "delta_SCD":          _r(delta_scd, 4),
        "CPC_kappa":          _r(cpc_kappa, 4),
        "CPC_filtered_n":     cpc_n,
    }

def bootstrap_esfp_ci(
    model_name: str,
    ar_df: pd.DataFrame,
    psi_df: pd.DataFrame,
    scd_df: pd.DataFrame,
    cpc_result: Dict[str, Any],
    psi_ceil: float,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Bootstrap ESFP with replacement at the item_id level (1000 resamples).
    """
    rng = np.random.default_rng(seed)
    all_item_ids = ar_df["item_id"].values
    n_items = len(all_item_ids)

    boot_scores: List[float] = []

    for _ in range(n_boot):
        sampled_idx = rng.integers(0, n_items, size=n_items)
        sampled_ids = all_item_ids[sampled_idx]

        boot_ar = ar_df.iloc[sampled_idx].copy()

        psi_sorted = psi_df.set_index("item_id").reindex(all_item_ids).reset_index()
        boot_psi_rows = psi_sorted.iloc[sampled_idx].copy()
        boot_psi = boot_psi_rows[["item_id", "PSI"]].rename(columns={"PSI": "PSI"})

        scd_by_item = {iid: grp for iid, grp in scd_df.groupby("item_id")}
        boot_scd_parts = [scd_by_item[iid] for iid in sampled_ids if iid in scd_by_item]
        if not boot_scd_parts:
            continue
        boot_scd = pd.concat(boot_scd_parts, ignore_index=True)

        score = compute_esfp_score(
            model_name, boot_ar, boot_psi, boot_scd, cpc_result, psi_ceil=psi_ceil
        )
        if pd.notna(score["ESFP"]):
            boot_scores.append(score["ESFP"])

    if not boot_scores:
        return {
            "model": model_name,
            "ESFP_mean": float("nan"), "ESFP_std": float("nan"),
            "CI_lower": float("nan"),  "CI_upper": float("nan"),
        }

    arr = np.array(boot_scores)
    return {
        "model":      model_name,
        "ESFP_mean":  round(float(arr.mean()), 5),
        "ESFP_std":   round(float(arr.std()),  5),
        "CI_lower":   round(float(np.percentile(arr, 2.5)),  5),
        "CI_upper":   round(float(np.percentile(arr, 97.5)), 5),
    }
