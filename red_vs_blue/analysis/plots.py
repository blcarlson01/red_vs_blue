"""
plots.py

Camera-ready plots for the Red vs. Blue benchmark.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# Plot config (NeurIPS-safe)
# ============================================================

plt.rcParams.update({
    "figure.figsize": (5, 3.5),
    "font.size": 9,
    "axes.titlesize": 10,
    "axes.labelsize": 9,
    "legend.fontsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


# ============================================================
# Helpers
# ============================================================

def load_results(results_dir: Path) -> pd.DataFrame:
    """Load aggregated results from JSONL file."""
    path = results_dir / "aggregated" / "all_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(
            f"Aggregated results not found at {path}\n"
            f"Run aggregate_results.py first:\n"
            f"  python -m red_vs_blue.analysis.aggregate_results {results_dir}"
        )
    
    # Load JSONL file
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    df = pd.DataFrame(records)
    
    # Compute win_rate column if not present
    if "value" in df.columns and "win_rate" not in df.columns:
        df["win_rate"] = df["value"].astype(float)
    
    return df


def save(fig, out_dir: Path, name: str):
    out_dir.mkdir(exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


# ============================================================
# Plots
# ============================================================

def plot_win_rate(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots()

    for model in df.model.unique():
        sub = df[df.model == model]
        ax.plot(
            sub.num_players,
            sub.win_rate,
            marker="o",
            label=model,
        )

    ax.set_xlabel("Number of Players")
    ax.set_ylabel("Win Rate")
    ax.set_title("Blue Win Rate vs Game Size")
    ax.legend()
    ax.grid(alpha=0.3)

    save(fig, out_dir, "win_rate_vs_players")


def plot_belief_alignment(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots()

    for model in df.model.unique():
        sub = df[df.model == model]
        ax.plot(
            sub.num_players,
            sub.avg_belief_alignment,
            marker="o",
            label=model,
        )

    ax.axhline(0.0, linestyle="--", linewidth=1)
    ax.set_xlabel("Number of Players")
    ax.set_ylabel("Belief Alignment")
    ax.set_title("Epistemic Alignment vs Game Size")
    ax.legend()
    ax.grid(alpha=0.3)

    save(fig, out_dir, "belief_alignment_vs_players")


def plot_alignment_vs_win(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots()

    for model in df.model.unique():
        sub = df[df.model == model]
        ax.scatter(
            sub.avg_belief_alignment,
            sub.win_rate,
            label=model,
            s=40,
        )

    ax.set_xlabel("Belief Alignment")
    ax.set_ylabel("Win Rate")
    ax.set_title("Outcome vs Epistemic Quality")
    ax.legend()
    ax.grid(alpha=0.3)

    save(fig, out_dir, "alignment_vs_win")


def plot_deception(df: pd.DataFrame, out_dir: Path):
    fig, ax = plt.subplots()

    for model in df.model.unique():
        sub = df[df.model == model]
        ax.plot(
            sub.num_players,
            sub.apt_leader_deception,
            marker="o",
            label=model,
        )

    ax.set_xlabel("Number of Players")
    ax.set_ylabel("APT Leader Deception Score")
    ax.set_title("Deception Success vs Game Size")
    ax.legend()
    ax.grid(alpha=0.3)

    save(fig, out_dir, "apt_leader_deception_vs_players")

def plot_cost_vs_win(df: pd.DataFrame, out_dir: Path):
    """
    Plot win rate vs computational cost (if data available).
    """
    # Check for various token column names
    token_col = None
    for col_name in ["tokens_used", "avg_tokens_used", "avg_total_tokens"]:
        if col_name in df.columns and df[col_name].notna().any():
            token_col = col_name
            break
    
    if token_col is None:
        print("  (Skipping cost_vs_win - tokens data not available)")
        return
    
    # Filter to rows with tokens data
    df_tokens = df.dropna(subset=[token_col])
    if df_tokens.empty:
        print("  (Skipping cost_vs_win - tokens data not available)")
        return
    
    fig, ax = plt.subplots()

    for model in df_tokens.model.unique():
        sub = df_tokens[df_tokens.model == model]
        ax.scatter(
            sub[token_col],
            sub.win_rate,
            label=model,
            s=50,
        )

    ax.set_xlabel("Tokens per Game")
    ax.set_ylabel("Win Rate")
    ax.set_title("Performance vs Compute")
    ax.legend()
    ax.grid(alpha=0.3)

    save(fig, out_dir, "cost_vs_win")


# ============================================================
# Main
# ============================================================

def main(results_dir: str):
    results_dir = Path(results_dir)
    
    try:
        df = load_results(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)

    if df.empty:
        print("No aggregated results found.")
        return

    out_dir = results_dir / "figures"

    print("\nGenerating plots...")
    plot_win_rate(df, out_dir)
    plot_belief_alignment(df, out_dir)
    plot_alignment_vs_win(df, out_dir)
    plot_deception(df, out_dir)
    plot_cost_vs_win(df, out_dir)

    print(f"\nSaved figures to {out_dir}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python plots.py <results_dir>")
        sys.exit(1)

    main(sys.argv[1])
