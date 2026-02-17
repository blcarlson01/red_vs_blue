from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd
from inspect_ai.model import GenerateConfig, get_model

ROOT = Path(__file__).resolve().parent.parent
ANALYSIS_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from aggregate_results import aggregate_role_policy_rewards
from plots import plot_role_policy_rewards


def _load_summary(summary_path: Path) -> dict:
    with summary_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _results_dataframe(summary: dict) -> pd.DataFrame:
    rows = []
    model_label = summary.get("model") or "polarix_sh"

    for r in summary.get("results", []):
        rows.append(
            {
                "model": model_label,
                "rollout": r.get("rollout"),
                "winner": r.get("winner"),
                "value": r.get("total_reward", 0.0),
                "blue_role_reward": r.get("blue_role_reward", 0.0),
                "red_role_reward": r.get("red_role_reward", 0.0),
            }
        )

    return pd.DataFrame(rows)


def _save_agent_ratings(summary: dict, out_dir: Path) -> Path:
    polarix = summary.get("polarix", {})
    ratings = polarix.get("agent_ratings", {})
    probs = polarix.get("agent_equilibrium_prob", {})

    rows = []
    for agent, rating in ratings.items():
        rows.append(
            {
                "agent": agent,
                "rating": rating,
                "equilibrium_prob": probs.get(agent),
            }
        )

    df = pd.DataFrame(rows).sort_values("rating", ascending=False)
    out_path = out_dir / "polarix_agent_ratings.csv"
    df.to_csv(out_path, index=False)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(df["agent"], df["rating"], color="#5B8FF9", alpha=0.8)
    ax1.set_ylabel("Polarix Rating")
    ax1.set_title("Polarix Agent Ratings")

    ax2 = ax1.twinx()
    ax2.plot(df["agent"], df["equilibrium_prob"], color="#F08C2B", marker="o")
    ax2.set_ylabel("Equilibrium Probability")

    fig.tight_layout()
    fig_path = out_dir / "polarix_agent_ratings.png"
    fig.savefig(fig_path)
    plt.close(fig)

    return out_path


def _winner_counts(df_results: pd.DataFrame) -> pd.DataFrame:
    return (
        df_results["winner"]
        .fillna("draw")
        .value_counts()
        .rename_axis("winner")
        .reset_index(name="count")
    )


async def _generate_llm_executive_summary(
    summary: dict,
    agg_df: pd.DataFrame,
    winners_df: pd.DataFrame,
    *,
    model_name: str,
    model_base_url: str | None,
) -> str:
    model = get_model(model_name, base_url=model_base_url)

    polarix = summary.get("polarix", {})
    ratings = polarix.get("agent_ratings", {})
    probs = polarix.get("agent_equilibrium_prob", {})

    top_agents = sorted(ratings.items(), key=lambda x: x[1], reverse=True)[:5]
    top_agents_text = "\n".join(
        f"- {agent}: rating={rating:.4f}, eq_prob={float(probs.get(agent, 0.0)):.4f}"
        for agent, rating in top_agents
    )

    winner_counts = {row["winner"]: int(row["count"]) for _, row in winners_df.iterrows()}
    aggregate_row = agg_df.iloc[0].to_dict() if not agg_df.empty else {}

    prompt = f"""You are writing an executive summary for a Red vs. Blue Polarix evaluation report.

Write concise Markdown with the exact sections below:
1) Executive Summary
2) What Polarix Ratings Mean
3) Key Results
4) How To Utilize These Results
5) Recommended Next Experiments

Guidelines:
- Be specific and action-oriented.
- Explain how team members should use ratings and equilibrium probability in decision-making.
- Mention both limitations and confidence.
- Keep to ~350-550 words.

Evaluation data:
- Rollouts: {summary.get('rollouts')}
- Policy: {summary.get('policy')}
- Model: {summary.get('model')}
- Model base URL: {summary.get('model_base_url')}
- Win counts: {winner_counts}
- Mean reward: {summary.get('avg_reward')}
- Mean blue role reward: {aggregate_row.get('mean_blue_role_reward')}
- Mean red role reward: {aggregate_row.get('mean_red_role_reward')}

Top agents by Polarix rating:
{top_agents_text if top_agents_text else '- N/A'}

Polarix solver summary:
{json.dumps(polarix.get('summary', {}), indent=2)}
"""

    response = await model.generate(
        input=prompt,
        config=GenerateConfig(max_tokens=1400, temperature=0.3),
    )

    text = response.completion if getattr(response, "completion", None) else (
        response.choices[0].message.text if getattr(response, "choices", None) else ""
    )

    if not text or not text.strip():
        raise RuntimeError("LLM returned an empty executive summary")

    return text.strip()


def _fallback_summary(summary: dict, agg_df: pd.DataFrame, winners_df: pd.DataFrame) -> str:
    aggregate_row = agg_df.iloc[0].to_dict() if not agg_df.empty else {}
    winner_counts = {row["winner"]: int(row["count"]) for _, row in winners_df.iterrows()}
    return (
        "# Polarix Executive Summary\n\n"
        "## Executive Summary\n"
        "LLM summary generation was unavailable, so this deterministic summary was produced.\n\n"
        "## Key Results\n"
        f"- Rollouts: {summary.get('rollouts')}\n"
        f"- Policy: {summary.get('policy')}\n"
        f"- Model: {summary.get('model')}\n"
        f"- Winner counts: {winner_counts}\n"
        f"- Mean reward: {summary.get('avg_reward')}\n"
        f"- Mean blue role reward: {aggregate_row.get('mean_blue_role_reward')}\n"
        f"- Mean red role reward: {aggregate_row.get('mean_red_role_reward')}\n\n"
        "## How To Utilize These Results\n"
        "- Use `agent_ratings` to rank relative strength under game-theoretic pressure.\n"
        "- Use `agent_equilibrium_prob` to identify which agents remain strategically relevant in mixed play.\n"
        "- Use per-role rewards to diagnose specialization gaps (blue defense vs red deception).\n"
    )


def run(
    summary_json: str,
    output_dir: str = "analysis_output",
    *,
    generate_llm_summary: bool = True,
    summary_model: str = "ollama/gpt-oss:20b",
    summary_model_base_url: str | None = None,
) -> dict[str, str]:
    summary_path = Path(summary_json)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summary = _load_summary(summary_path)
    df_results = _results_dataframe(summary)

    if df_results.empty:
        raise RuntimeError("No rollout results found in benchmark summary")

    raw_csv = out_dir / "polarix_rollout_results.csv"
    df_results.to_csv(raw_csv, index=False)

    agg_df = aggregate_role_policy_rewards(df_results)
    agg_csv = out_dir / "polarix_aggregate_metrics.csv"
    agg_df.to_csv(agg_csv, index=False)

    scatter_png = out_dir / "polarix_role_policy_rewards.png"
    plot_role_policy_rewards(agg_df, out_path=str(scatter_png))

    ratings_csv = _save_agent_ratings(summary, out_dir)

    winners = _winner_counts(df_results)
    winners_csv = out_dir / "polarix_winner_counts.csv"
    winners.to_csv(winners_csv, index=False)

    summary_path = out_dir / "polarix_executive_summary.md"
    if generate_llm_summary:
        try:
            executive_summary = asyncio.run(
                _generate_llm_executive_summary(
                    summary,
                    agg_df,
                    winners,
                    model_name=summary_model,
                    model_base_url=summary_model_base_url,
                )
            )
        except Exception as e:
            executive_summary = (
                _fallback_summary(summary, agg_df, winners)
                + f"\n\n---\n\n**LLM generation error:** {e}\n"
            )
    else:
        executive_summary = _fallback_summary(summary, agg_df, winners)

    summary_path.write_text(executive_summary, encoding="utf-8")

    return {
        "raw_results": str(raw_csv),
        "aggregate_metrics": str(agg_csv),
        "role_policy_plot": str(scatter_png),
        "agent_ratings": str(ratings_csv),
        "winner_counts": str(winners_csv),
        "executive_summary": str(summary_path),
    }


def main():
    parser = argparse.ArgumentParser(description="Run analysis for Polarix benchmark summary")
    parser.add_argument(
        "summary_json",
        nargs="?",
        default="results_polarix_red_vs_blue/benchmark_summary.json",
        help="Path to benchmark_summary.json",
    )
    parser.add_argument(
        "--output-dir",
        default="results_polarix_red_vs_blue/analysis",
        help="Directory for analysis outputs",
    )
    parser.add_argument(
        "--summary-model",
        default="ollama/gpt-oss:20b",
        help="Model used for LLM executive summary",
    )
    parser.add_argument(
        "--summary-model-base-url",
        default=None,
        help="Optional base URL for summary model (e.g. http://localhost:11434/v1)",
    )
    parser.add_argument(
        "--no-llm-summary",
        action="store_true",
        help="Disable LLM summary generation and write deterministic summary only",
    )
    args = parser.parse_args()

    outputs = run(
        args.summary_json,
        args.output_dir,
        generate_llm_summary=not args.no_llm_summary,
        summary_model=args.summary_model,
        summary_model_base_url=args.summary_model_base_url,
    )
    print("Polarix analysis complete. Outputs:")
    for name, path in outputs.items():
        print(f"- {name}: {path}")


if __name__ == "__main__":
    main()
