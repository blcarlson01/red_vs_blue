"""
advanced_infographics.py

Publication-ready infographics for Red vs. Blue benchmark analysis.
Generates camera-ready figures suitable for research papers and presentations.
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ============================================================
# Plot Configuration (Publication-Ready)
# ============================================================

plt.rcParams.update({
    "figure.figsize": (6, 4.5),
    "figure.dpi": 100,
    "font.size": 9,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "legend.fontsize": 8,
    "lines.linewidth": 1.5,
    "lines.markersize": 6,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# Use a publication-friendly color palette
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]


# ============================================================
# Helpers
# ============================================================

def load_results(results_dir: Path) -> pd.DataFrame:
    """Load aggregated results from JSONL file."""
    path = results_dir / "aggregated" / "all_results.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"Results not found at {path}")
    
    records = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    
    return pd.DataFrame(records)


def load_analysis(results_dir: Path) -> Dict:
    """Load advanced analysis results."""
    path = results_dir / "aggregated" / "advanced_analysis.json"
    if not path.exists():
        raise FileNotFoundError(f"Analysis not found at {path}")
    
    with open(path, "r") as f:
        return json.load(f)


def save_figure(fig, out_dir: Path, name: str):
    """Save figure in both PDF and PNG formats."""
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{name}.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)
    print(f"  ✓ {name}")


# ============================================================
# Infographic Functions
# ============================================================

def infographic_game_length_distribution(df: pd.DataFrame, out_dir: Path):
    """Distribution of game lengths (rounds played)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    if "rounds_played" not in df.columns or df["rounds_played"].isna().all():
        print("  (Skipping game_length_distribution - data not available)")
        return
    
    rounds = df["rounds_played"].dropna()
    
    ax.hist(rounds, bins=range(int(rounds.min()), int(rounds.max()) + 2), 
            color=PALETTE[0], alpha=0.7, edgecolor="black", linewidth=1.2)
    
    ax.axvline(rounds.mean(), color="red", linestyle="--", linewidth=2, label=f"Mean: {rounds.mean():.1f}")
    ax.axvline(rounds.median(), color="green", linestyle="--", linewidth=2, label=f"Median: {rounds.median():.1f}")
    
    ax.set_xlabel("Game Length (rounds)", fontsize=10)
    ax.set_ylabel("Frequency", fontsize=10)
    ax.set_title("Distribution of Game Lengths", fontsize=11, fontweight="bold")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    
    save_figure(fig, out_dir, "01_game_length_distribution")


def infographic_belief_dynamics_progression(df: pd.DataFrame, out_dir: Path):
    """Belief entropy reduction across games."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    if "avg_entropy_reduction" not in df.columns or df["avg_entropy_reduction"].isna().all():
        print("  (Skipping belief_dynamics_progression - data not available)")
        return
    
    entropy = df["avg_entropy_reduction"].dropna().sort_values().reset_index(drop=True)
    game_ids = np.arange(len(entropy)) + 1
    
    ax.plot(game_ids, entropy, marker="o", linewidth=2, markersize=7, 
            color=PALETTE[1], label="Entropy Reduction")
    ax.fill_between(game_ids, entropy, alpha=0.2, color=PALETTE[1])
    
    ax.axhline(entropy.mean(), color="red", linestyle="--", linewidth=1.5, 
               label=f"Mean: {entropy.mean():.4f}", alpha=0.7)
    
    ax.set_xlabel("Game (sorted by entropy reduction)", fontsize=10)
    ax.set_ylabel("Average Entropy Reduction", fontsize=10)
    ax.set_title("Belief Dynamics: Information Gathered Over Games", fontsize=11, fontweight="bold")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    
    save_figure(fig, out_dir, "02_belief_dynamics_progression")


def infographic_deception_effectiveness(df: pd.DataFrame, analysis: Dict, out_dir: Path):
    """Deception score vs red win rate."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    if "apt_leader_deception" not in df.columns or "blues_win" not in df.columns:
        print("  (Skipping deception_effectiveness - data not available)")
        return
    
    valid_df = df[["apt_leader_deception", "blues_win"]].dropna()
    if len(valid_df) == 0:
        print("  (Skipping deception_effectiveness - no valid data)")
        return
    
    # Convert to red win
    red_win = valid_df["blues_win"] == 0.0
    
    # Create scatter plot with jitter for visibility
    x_jitter = valid_df["apt_leader_deception"] + np.random.normal(0, 0.0001, len(valid_df))
    colors = [PALETTE[0] if w else PALETTE[2] for w in red_win]
    
    for i, (x, y) in enumerate(zip(x_jitter, red_win)):
        ax.scatter(x, y, s=100, color=colors[i], alpha=0.6, edgecolors="black", linewidth=0.5)
    
    # Add trend line if enough points
    if len(valid_df) > 1:
        z = np.polyfit(valid_df["apt_leader_deception"], red_win.astype(int), 1)
        p = np.poly1d(z)
        x_line = np.linspace(valid_df["apt_leader_deception"].min(), valid_df["apt_leader_deception"].max(), 100)
        ax.plot(x_line, p(x_line), "r--", linewidth=2, alpha=0.7, label="Trend")
    
    ax.set_xlabel("APT Leader Deception Score", fontsize=10)
    ax.set_ylabel("Outcome (0=Red, 1=Blue)", fontsize=10)
    ax.set_title("Deception Effectiveness vs Game Outcome", fontsize=11, fontweight="bold")
    ax.set_ylim(-0.15, 1.15)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Red Win", "Blue Win"])
    ax.grid(alpha=0.3, axis="x")
    
    if len(valid_df) > 1:
        ax.legend()
    
    save_figure(fig, out_dir, "03_deception_effectiveness")


def infographic_entropy_box_plot(df: pd.DataFrame, out_dir: Path):
    """Entropy reduction by information gathering level."""
    fig, ax = plt.subplots(figsize=(6, 4.5))
    
    if "avg_entropy_reduction" not in df.columns or df["avg_entropy_reduction"].isna().all():
        print("  (Skipping entropy_box_plot - data not available)")
        return
    
    entropy = df["avg_entropy_reduction"].dropna()
    q75 = entropy.quantile(0.75)
    q25 = entropy.quantile(0.25)
    median = entropy.median()
    
    # Categorize games
    high_entropy = entropy[entropy >= q75]
    mid_entropy = entropy[(entropy > q25) & (entropy < q75)]
    low_entropy = entropy[entropy <= q25]
    
    data_to_plot = [low_entropy, mid_entropy, high_entropy]
    labels = [f"Low\n(n={len(low_entropy)})", f"Medium\n(n={len(mid_entropy)})", f"High\n(n={len(high_entropy)})"]
    
    bp = ax.boxplot(data_to_plot, tick_labels=labels, patch_artist=True, 
                     widths=0.6, showmeans=True)
    
    # Color the boxes
    for patch, color in zip(bp['boxes'], [PALETTE[3], PALETTE[4], PALETTE[5]]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel("Entropy Reduction", fontsize=10)
    ax.set_xlabel("Information Gathering Level", fontsize=10)
    ax.set_title("Distribution of Information Gathering Across Games", fontsize=11, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    
    save_figure(fig, out_dir, "04_entropy_box_plot")


def infographic_model_performance_heatmap(df: pd.DataFrame, analysis: Dict, out_dir: Path):
    """Model performance across all metrics."""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    if "model_comparison" not in analysis or isinstance(analysis["model_comparison"], dict) and "note" in analysis["model_comparison"]:
        print("  (Skipping model_performance_heatmap - data not available)")
        return
    
    model_data = analysis["model_comparison"]
    if not model_data or "note" in model_data:
        print("  (Skipping model_performance_heatmap - data not available)")
        return
    
    # Build matrix
    metrics = ["blue_win_rate", "avg_rounds", "avg_entropy", "avg_belief_alignment"]
    models = list(model_data.keys())
    
    matrix = []
    for model in models:
        row = [
            model_data[model].get("blue_win_rate", 0),
            model_data[model].get("avg_rounds", 0) / 15,  # Normalize to 0-1 range
            model_data[model].get("avg_entropy", 0) * 50,  # Scale for visibility
            model_data[model].get("avg_belief_alignment", 0) * 1000,  # Scale for visibility
        ]
        matrix.append(row)
    
    matrix = np.array(matrix)
    
    # Normalize each column to 0-1
    matrix_norm = (matrix - matrix.min(axis=0)) / (matrix.max(axis=0) - matrix.min(axis=0) + 1e-8)
    
    im = ax.imshow(matrix_norm, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)
    
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_yticks(np.arange(len(models)))
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_yticklabels(models)
    
    # Add text annotations
    for i in range(len(models)):
        for j in range(len(metrics)):
            text = ax.text(j, i, f"{matrix[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=9)
    
    ax.set_title("Model Performance Comparison", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Normalized Score")
    fig.tight_layout()
    
    save_figure(fig, out_dir, "05_model_performance_heatmap")


def infographic_game_outcome_summary(df: pd.DataFrame, out_dir: Path):
    """Pie chart of game outcomes."""
    fig, ax = plt.subplots(figsize=(6, 5))
    
    if "blues_win" not in df.columns:
        print("  (Skipping game_outcome_summary - data not available)")
        return
    
    outcomes = df["blues_win"].dropna()
    if len(outcomes) == 0:
        print("  (Skipping game_outcome_summary - no valid data)")
        return
    
    blue_wins = int((outcomes == 1.0).sum())
    red_wins = int((outcomes == 0.0).sum())
    
    sizes = [blue_wins, red_wins]
    labels = [f"Blue Wins\n({blue_wins})", f"Red Wins\n({red_wins})"]
    colors = [PALETTE[2], PALETTE[0]]
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%",
                                        explode=explode, startangle=90, textprops={"fontsize": 10})
    
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_fontweight("bold")
        autotext.set_fontsize(11)
    
    ax.set_title("Game Outcome Distribution", fontsize=11, fontweight="bold")
    
    save_figure(fig, out_dir, "06_game_outcome_summary")


def infographic_early_game_correlations(df: pd.DataFrame, out_dir: Path):
    """Correlation heatmap of early-game metrics."""
    fig, ax = plt.subplots(figsize=(7, 5))
    
    cols_to_check = ["avg_entropy_reduction", "rounds_played", "avg_belief_alignment"]
    available_cols = [c for c in cols_to_check if c in df.columns]
    
    if len(available_cols) < 2:
        print("  (Skipping early_game_correlations - insufficient data)")
        return
    
    # Compute correlation matrix
    corr_data = df[available_cols].dropna()
    if len(corr_data) < 2:
        print("  (Skipping early_game_correlations - insufficient valid data)")
        return
    
    corr_matrix = corr_data.corr()
    
    # Create heatmap using matplotlib imshow
    im = ax.imshow(corr_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    
    # Set ticks and labels
    ax.set_xticks(np.arange(len(corr_matrix.columns)))
    ax.set_yticks(np.arange(len(corr_matrix.columns)))
    ax.set_xticklabels(corr_matrix.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr_matrix.columns, fontsize=9)
    
    # Add correlation values as text annotations
    for i in range(len(corr_matrix)):
        for j in range(len(corr_matrix)):
            text = ax.text(j, i, f"{corr_matrix.iloc[i, j]:.2f}",
                          ha="center", va="center", color="black", fontsize=9, fontweight="bold")
    
    # Add gridlines
    ax.set_xticks(np.arange(len(corr_matrix.columns)) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(corr_matrix.columns)) - 0.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle="-", linewidth=1)
    
    ax.set_title("Correlation Matrix: Early-Game Predictors", fontsize=11, fontweight="bold")
    plt.colorbar(im, ax=ax, label="Correlation")
    fig.tight_layout()
    
    save_figure(fig, out_dir, "07_early_game_correlations")


def infographic_metrics_summary_panel(df: pd.DataFrame, analysis: Dict, out_dir: Path):
    """Summary panel with key metrics."""
    fig = plt.figure(figsize=(10, 6))
    gs = fig.add_gridspec(2, 2, hspace=0.4, wspace=0.3)
    
    # Panel 1: Games Summary
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.axis("off")
    summary_text = f"""
    GAMES SUMMARY
    ──────────────────
    Total Games: {len(df)}
    Avg Rounds: {df['rounds_played'].mean():.1f}
    Min/Max: {df['rounds_played'].min():.0f}–{df['rounds_played'].max():.0f}
    """
    ax1.text(0.1, 0.5, summary_text, fontsize=9, family="monospace", 
             verticalalignment="center", bbox=dict(boxstyle="round", facecolor=PALETTE[0], alpha=0.3))
    
    # Panel 2: Win Rates
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.axis("off")
    if "blues_win" in df.columns:
        blue_wins = (df["blues_win"] == 1.0).sum()
        red_wins = (df["blues_win"] == 0.0).sum()
        total = blue_wins + red_wins
        win_text = f"""
        WIN RATES
        ──────────────────
        Blue: {blue_wins}/{total} ({100*blue_wins/total:.1f}%)
        Red: {red_wins}/{total} ({100*red_wins/total:.1f}%)
        """
    else:
        win_text = "WIN RATES\n──────────────────\nData unavailable"
    
    ax2.text(0.1, 0.5, win_text, fontsize=9, family="monospace",
             verticalalignment="center", bbox=dict(boxstyle="round", facecolor=PALETTE[1], alpha=0.3))
    
    # Panel 3: Beliefs & Deception
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.axis("off")
    belief_text = f"""
    EPISTEMIC METRICS
    ──────────────────
    Avg Entropy Δ: {df['avg_entropy_reduction'].mean():.4f}
    Avg Alignment: {df['avg_belief_alignment'].mean():.4f}
    Avg Deception: {df['apt_leader_deception'].mean():.4f}
    """
    ax3.text(0.1, 0.5, belief_text, fontsize=9, family="monospace",
             verticalalignment="center", bbox=dict(boxstyle="round", facecolor=PALETTE[2], alpha=0.3))
    
    # Panel 4: Computational Cost
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.axis("off")
    token_col = None
    for col in ["tokens_used", "input_tokens", "output_tokens"]:
        if col in df.columns and df[col].notna().any():
            token_col = col
            break
    
    if token_col:
        cost_text = f"""
        COMPUTATIONAL COST
        ──────────────────
        Avg {token_col}: {df[token_col].mean():.0f}
        Total: {df[token_col].sum():.0f}
        Per game: {df[token_col].mean():.0f}
        """
    else:
        cost_text = "COMPUTATIONAL COST\n──────────────────\nData unavailable"
    
    ax4.text(0.1, 0.5, cost_text, fontsize=9, family="monospace",
             verticalalignment="center", bbox=dict(boxstyle="round", facecolor=PALETTE[3], alpha=0.3))
    
    fig.suptitle("Red vs. Blue Benchmark: Summary Metrics Panel", fontsize=13, fontweight="bold", y=0.98)
    
    save_figure(fig, out_dir, "08_metrics_summary_panel")


# ============================================================
# Main
# ============================================================

def main(results_dir: str):
    """Generate all infographics."""
    results_dir = Path(results_dir)
    
    try:
        df = load_results(results_dir)
        analysis = load_analysis(results_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    if df.empty:
        print("No results to visualize.")
        return
    
    out_dir = results_dir / "figures"
    
    print("\n" + "="*80)
    print("GENERATING PUBLICATION-READY INFOGRAPHICS")
    print("="*80)
    
    # Generate all infographics
    infographic_game_length_distribution(df, out_dir)
    infographic_belief_dynamics_progression(df, out_dir)
    infographic_deception_effectiveness(df, analysis, out_dir)
    infographic_entropy_box_plot(df, out_dir)
    infographic_model_performance_heatmap(df, analysis, out_dir)
    infographic_game_outcome_summary(df, out_dir)
    infographic_early_game_correlations(df, out_dir)
    infographic_metrics_summary_panel(df, analysis, out_dir)
    
    print("\n" + "="*80)
    print(f"✓ All infographics saved to {out_dir}")
    print("="*80 + "\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python advanced_infographics.py <results_dir>")
        sys.exit(1)
    
    main(sys.argv[1])
