import textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
import seaborn as sns
from scipy import stats as scipy_stats
from typing import Dict, Any

from esfp_benchmark.config import RESULTS_DIR

FIGURES_DIR = RESULTS_DIR / "figures"
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# ═══════════════════════════════════════════════════════════════════════
# Global Style
# ═══════════════════════════════════════════════════════════════════════

def set_global_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif", "Liberation Serif", "Times New Roman"],
        "font.size": 8, "axes.titlesize": 11, "axes.labelsize": 10,
        "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 9,
        "figure.dpi": 150, "savefig.dpi": 300,
        "axes.linewidth": 0.8, "grid.linewidth": 0.5, "lines.linewidth": 1.2,
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.spines.top": False, "axes.spines.right": False,
    })

_tab10 = plt.cm.tab10.colors

# Dynamic style assigning
assigned_styles = {}

def _ms(m: str) -> Dict[str, Any]:
    if m in assigned_styles:
        return assigned_styles[m]
    idx = len(assigned_styles) % len(_tab10)
    assigned_styles[m] = dict(color=_tab10[idx], marker="o", display_name=m.split("/")[-1], params="")
    return assigned_styles[m]

def _dn(m: str) -> str:
    return _ms(m)["display_name"]
def _dn_p(m: str) -> str:
    s = _ms(m)
    return f'{s["display_name"]} {s["params"]}'.strip() if s["params"] else s["display_name"]
def _save_fig(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIGURES_DIR / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    print(f"  Saved: figures/{name}.pdf & .png")

# ═══════════════════════════════════════════════════════════════════════
# Figure 1 — Response Demo
# ═══════════════════════════════════════════════════════════════════════
def plot_fig1_response_demo(model_intermediates: Dict[str, Dict], scores_df: pd.DataFrame):
    set_global_style()
    if not model_intermediates:
        print("  Fig 1: No data — skipping."); return

    sorted_df = scores_df.sort_values("ESFP")
    demo_model = sorted_df.iloc[len(sorted_df) // 2]["model"] if len(sorted_df) > 0 else list(model_intermediates.keys())[0]
    if demo_model not in model_intermediates:
        demo_model = list(model_intermediates.keys())[0]

    scd_df = model_intermediates[demo_model]["scd_df"]
    pivot = scd_df.pivot_table(index="item_id", columns="phrasing", values="SCD")
    if "P2" in pivot.columns and "P1" in pivot.columns:
        delta = pivot["P2"].fillna(0) - pivot["P1"].fillna(0)
        best_item = int((delta - delta.median()).abs().idxmin())
    else:
        best_item = int(scd_df["item_id"].iloc[0])

    q_text = str(scd_df[scd_df["item_id"] == best_item].iloc[0].get("question", "N/A"))

    fig = plt.figure(figsize=(7, 4.2))
    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.1, top=0.84, bottom=0.12)

    phrasing_meta = [
        ("P0", "Neutral Baseline", "#f0f0f0"),
        ("P2", "Subjectified", "#dbe9f6"),
        ("P4", "Disagree Invite", "#fde8d0"),
    ]
    for col_i, (p_key, label, bg) in enumerate(phrasing_meta):
        ax = fig.add_subplot(gs[0, col_i])
        ax.set_xlim(0, 10); ax.set_ylim(0, 10)
        ax.set_facecolor(bg); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True); spine.set_color("#bbb"); spine.set_linewidth(0.8)
        row = scd_df[(scd_df["item_id"] == best_item) & (scd_df["phrasing"] == p_key)]
        resp = str(row.iloc[0]["response"])[:400] if not row.empty else "(no data)"
        scd_val = float(row.iloc[0]["SCD"]) if not row.empty else float("nan")
        ax.text(5, 9.5, f"{p_key}: {label}", ha="center", va="top", fontsize=9, fontweight="bold")
        ax.text(5, 5.0, textwrap.fill(resp, width=44), ha="center", va="center",
                fontsize=5.2, family="serif", linespacing=1.35)
        ax.text(5, 0.4, f"SCD = {scd_val:.3f}", ha="center", fontsize=7.5, color="#444")

    fig.suptitle(
        f'Figure 1 - Response Demo: {_dn(demo_model)}\n'
        f'Topic: "{textwrap.shorten(q_text, 85, placeholder="...")}"',
        fontsize=9, fontweight="bold", y=0.96)
    fig.text(0.5, 0.02,
             "Responses from the same model to the same topic under three phrasing conditions.\n"
             "SCD (Stance Content Density) increases as the prompt shifts from neutral to subjectified.",
             ha="center", va="bottom", fontsize=7, color="#555", style="italic")
    _save_fig(fig, "fig1_response_demo")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# Figure 4 — Prompt Response Curve
# ═══════════════════════════════════════════════════════════════════════
def plot_fig2_prompt_response_curve(model_intermediates: Dict[str, Dict]):
    set_global_style()
    if not model_intermediates:
        print("  Fig 4: No data — skipping."); return

    phrasings = ["P0", "P1", "P2", "P3", "P4"]
    x = np.arange(len(phrasings))
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(7, 2.8))

    for model_name, inter in model_intermediates.items():
        st = _ms(model_name)
        ar_df, scd_df = inter["ar_df"], inter["scd_df"]
        kw = dict(color=st["color"], marker=st["marker"], markersize=4.5,
                  linewidth=1.2, markeredgecolor="white", markeredgewidth=0.4,
                  label=_dn(model_name))
        
        ar_vals = []
        for p in phrasings:
            ar_col = f"AR_{p}"
            val = ar_df[ar_col].dropna().mean() if ar_col in ar_df.columns else float('nan')
            ar_vals.append(val)
            
        ax_a.plot(x, ar_vals, **kw)
        
        scd_vals = [scd_df[scd_df["phrasing"] == p]["SCD"].dropna().mean() for p in phrasings]
        ax_b.plot(x, scd_vals, **kw)

    for ax, panel, ylabel in [
        (ax_a, "(A)", "AR (mean across items)"),
        (ax_b, "(B)", "SCD (mean across items)"),
    ]:
        ax.set_xticks(x); ax.set_xticklabels(phrasings)
        ax.set_xlabel("Phrasing Level"); ax.set_ylabel(ylabel, fontsize=9)
        ax.set_title(f"{panel}  Attribution Rate" if "A" in panel else f"{panel}  Stance Content Density",
                     fontsize=10, fontweight="bold", loc="left")
        ax.axhline(0, color="#ccc", linewidth=0.6, linestyle="--"); ax.grid(axis="y", alpha=0.25)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=min(5, len(labels)),
               fontsize=6.5, bbox_to_anchor=(0.5, -0.15), frameon=False)
    fig.suptitle("Figure 4 - Prompt Response Curve", fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save_fig(fig, "fig2_prompt_response_curve")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# Figure 2 — ESFP Score Ranking
# ═══════════════════════════════════════════════════════════════════════
def plot_fig3_main_ranking(scores_df: pd.DataFrame, boot_df: pd.DataFrame = None):
    set_global_style()
    df = scores_df.sort_values("ESFP", ascending=True).copy()
    n = len(df)
    fig, ax = plt.subplots(figsize=(7, max(4, n * 0.45)))

    for i, (_, row) in enumerate(df.iterrows()):
        st = _ms(row["model"]); val = row["ESFP"]
        ax.barh(i, val, color=st["color"], edgecolor="white", linewidth=0.6, height=0.7)
        ci_hi_val = None
        if boot_df is not None and not boot_df.empty:
            br = boot_df[boot_df["model"] == row["model"]]
            if not br.empty:
                ci_lo = float(br.iloc[0]["CI_lower"])
                ci_hi_val = float(br.iloc[0]["CI_upper"])
                ax.errorbar(val, i, xerr=[[val - ci_lo], [ci_hi_val - val]],
                            fmt="none", color="#333", capsize=3, linewidth=0.8)
        x_label = (ci_hi_val + 0.012) if ci_hi_val is not None else (max(val, 0) + 0.012)
        ax.text(x_label, i, f"{val:.3f}", va="center", fontsize=7.5, fontweight="bold")

    ax.set_yticks(np.arange(n))
    ax.set_yticklabels([_dn_p(m) for m in df["model"]], fontsize=8)
    ax.set_xlabel("ESFP Score", fontsize=10)
    ax.set_title("Figure 2 - ESFP Score Ranking with 95% Bootstrap CI",
                 fontsize=11, fontweight="bold", loc="left")
    ax.grid(axis="x", alpha=0.3)
    ax.set_xlim(left=min(0, df["ESFP"].min() - 0.01) if not df.empty else 0)
    plt.tight_layout()
    _save_fig(fig, "fig3_main_result_ranking")
    plt.close(fig)

# ═══════════════════════════════════════════════════════════════════════
# Figure 3 — Metric Decomposition Heatmap
# ═══════════════════════════════════════════════════════════════════════
def plot_fig4_metric_heatmap(scores_df: pd.DataFrame):
    set_global_style()
    df = scores_df.sort_values("ESFP", ascending=False).copy()
    df["display"] = df["model"].apply(_dn)

    metrics = ["delta_AR", "delta_SCD", "PSI_scaled", "CPC_kappa", "flexibility_signal"]
    metric_labels = ["dAR", "dSCD", "PSI_scaled", "CPC k", "Flex Signal"]
    data = df.set_index("display")[metrics].copy()
    data_norm = data.copy()
    for col in metrics:
        lo, hi = data[col].min(), data[col].max()
        data_norm[col] = (data[col] - lo) / (hi - lo) if (hi - lo) > 1e-9 else 0.5

    n_rows, n_cols = data.shape
    fig, ax = plt.subplots(figsize=(5.5, max(3.5, n_rows * 0.42)))
    ax.imshow(data_norm.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    for i in range(n_rows):
        for j in range(n_cols):
            v, nv = data.iloc[i, j], data_norm.iloc[i, j]
            ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                    fontsize=8, color="white" if nv > 0.65 else "black")
    ax.axvline(x=3.5, color="black", linewidth=2.5)
    ax.set_xticks(range(n_cols)); ax.set_xticklabels(metric_labels, fontsize=9)
    ax.set_yticks(range(n_rows)); ax.set_yticklabels(data.index, fontsize=8)
    ax.set_title("Figure 3 - Metric Decomposition Heatmap", fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save_fig(fig, "fig4_metric_heatmap")
    plt.close(fig)

def get_print_score_interval_insight(scores_df: pd.DataFrame):
    ESFP = scores_df["ESFP"].dropna()
    if ESFP.empty:
        return

    q1, q2, q3 = ESFP.quantile([0.25, 0.50, 0.75])
    print("\n" + "=" * 60)
    print("  ESFP Score Distribution — Interval Insight")
    print("=" * 60)
    print(f"  Mean              : {ESFP.mean():.4f}")
    print(f"  Median (50th)     : {q2:.4f}")

def generate_all_figures(scores_df: pd.DataFrame, boot_df: pd.DataFrame, model_intermediates: Dict[str, Dict]):
    print("\n" + "=" * 65)
    print("  Generating ESFP Visualization Suite")
    print("=" * 65 + "\n")

    plot_fig1_response_demo(model_intermediates, scores_df)
    plot_fig2_prompt_response_curve(model_intermediates)
    plot_fig3_main_ranking(scores_df, boot_df)
    plot_fig4_metric_heatmap(scores_df)
    get_print_score_interval_insight(scores_df)

    print(f"\n  All figures saved to: {FIGURES_DIR}/")
    print("=" * 65)
