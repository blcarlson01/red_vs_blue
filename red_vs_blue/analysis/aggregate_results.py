"""
aggregate_results.py

Aggregate Inspect results for the Red vs. Blue benchmark into
paper-ready tables.

Reads from .eval files (ZIP archives) in the results folder.
"""

from __future__ import annotations
import sys
import glob
import json
import zipfile
from pathlib import Path
from typing import Dict, List

import pandas as pd


# ============================================================
# Utilities
# ============================================================

def load_eval_file(eval_path: Path) -> List[Dict]:
    """
    Load all samples from an Inspect .eval ZIP file.
    Returns list of result dicts.
    """
    results = []
    try:
        with zipfile.ZipFile(eval_path, "r") as zf:
            # Find all sample files in samples/ directory
            sample_files = [f for f in zf.namelist() if f.startswith("samples/")]
            
            for sample_file in sorted(sample_files):
                try:
                    content = zf.read(sample_file).decode("utf-8")
                    data = json.loads(content)
                    results.append(data)
                except Exception as e:
                    print(f"Warning: Failed to load {sample_file}: {e}")
    except Exception as e:
        print(f"Error reading {eval_path}: {e}")
    
    return results


def collect_results(results_dir: Path) -> pd.DataFrame:
    """
    Load all .eval files in a directory and extract metrics.
    """
    eval_files = glob.glob(str(results_dir / "*.eval"))
    if not eval_files:
        raise FileNotFoundError(f"No .eval files found in {results_dir}")

    records = []
    for eval_file in eval_files:
        results = load_eval_file(Path(eval_file))
        
        for result in results:
            scores = result.get("scores", {})
            scorer = scores.get("red_vs_blue_scorer", {})
            metadata = scorer.get("metadata", {})
            
            # Extract model name from eval filename if available
            eval_filename = Path(eval_file).name
            model_name = "unknown"
            # Try to extract model from filename (format: timestamp_task_hash.eval)
            if "_red-vs-blue-task_" in eval_filename:
                model_name = "red_vs_blue_model"

            record = {
                "model": metadata.get("model", model_name),
                "value": scores.get("red_vs_blue_scorer", {}).get("value", 0),  # 1 = libs win, 0 = reds
            }

            # Flatten metadata
            for k, v in metadata.items():
                if k not in ["model", "roles", "public_log", "voting_history", "fired_players", "belief_histories"]:
                    # Skip non-numeric and complex fields
                    if isinstance(v, (int, float)):
                        record[k] = v

            # Extract token usage from model_usage field
            model_usage = result.get("model_usage", {})
            if model_usage:
                # Get usage for the first (usually only) model in the dict
                for model_key, usage_data in model_usage.items():
                    if isinstance(usage_data, dict):
                        record["tokens_used"] = usage_data.get("total_tokens", None)
                        record["input_tokens"] = usage_data.get("input_tokens", None)
                        record["output_tokens"] = usage_data.get("output_tokens", None)
                        break

            records.append(record)

    return pd.DataFrame.from_records(records)


# ============================================================
# Aggregation
# ============================================================

def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate metrics by model and player count.
    Also preserves token data.
    """
    # If no model column, create one
    if "model" not in df.columns or df["model"].isna().all():
        df["model"] = "default_model"
    
    # If no num_players column, try to infer from data
    if "num_players" not in df.columns:
        df["num_players"] = 5  # Default player count
    
    group_cols = ["model", "num_players"]

    agg_dict = {
        "games_played": ("value", "count"),
        "win_rate": ("value", "mean"),
        "avg_entropy_reduction": ("avg_entropy_reduction", "mean"),
        "avg_belief_alignment": ("avg_belief_alignment", "mean"),
        "avg_brier": ("avg_brier", "mean"),
        "apt_leader_deception": ("apt_leader_deception", "mean"),
        "avg_rounds": ("rounds_played", "mean"),
    }
    
    # Add token aggregations if available
    if "tokens_used" in df.columns:
        agg_dict["avg_tokens_used"] = ("tokens_used", "mean")
    if "input_tokens" in df.columns:
        agg_dict["avg_input_tokens"] = ("input_tokens", "mean")
    if "output_tokens" in df.columns:
        agg_dict["avg_output_tokens"] = ("output_tokens", "mean")

    agg = df.groupby(group_cols).agg(**agg_dict).reset_index()

    return agg


# ============================================================
# Main
# ============================================================

def main(results_dir: str):
    results_dir = Path(results_dir)
    
    if not results_dir.exists():
        raise FileNotFoundError(f"Results directory not found: {results_dir}")

    df = collect_results(results_dir)
    
    if df.empty:
        print("No results found in eval files.")
        return
    
    agg = aggregate(df)

    # Sort for readability
    agg = agg.sort_values(
        ["num_players", "model"],
        ascending=[True, True],
    )

    # Print paper-style table
    with pd.option_context("display.max_columns", None):
        print("\n=== Aggregated Results ===\n")
        print(agg.to_string(index=False, float_format="%.3f"))

    # Save artifacts
    out_dir = results_dir / "aggregated"
    out_dir.mkdir(exist_ok=True)

    agg.to_csv(out_dir / "summary.csv", index=False)
    df.to_json(out_dir / "all_results.jsonl", orient="records", lines=True)

    print(f"\nSaved aggregated results to {out_dir}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python aggregate_results.py <results_dir>")
        sys.exit(1)

    main(sys.argv[1])
