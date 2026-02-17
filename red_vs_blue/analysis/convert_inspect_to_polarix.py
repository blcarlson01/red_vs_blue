"""
convert_inspect_to_polarix.py

Convert Inspect .eval game outputs into a Polarix-ready benchmark summary.

This script:
1) Loads all samples from .eval files in a results directory
2) Builds per-rollout per-player scores from game outcomes and roles
3) Constructs an agent-vs-task score matrix
4) Solves Polarix equilibrium ratings (ce_maxent)
5) Writes benchmark_summary.json compatible with analysis/run_polarix_analysis.py
"""

from __future__ import annotations

import argparse
import glob
import json
import numbers
from pathlib import Path

import numpy as np
import polarix as plx


def _to_jsonable(value):
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, numbers.Number):
        return float(value)
    if hasattr(value, "tolist"):
        return _to_jsonable(value.tolist())
    try:
        return float(value)
    except Exception:
        return str(value)


def _load_eval_samples(eval_path: Path) -> list[dict]:
    import zipfile

    samples = []
    with zipfile.ZipFile(eval_path, "r") as zf:
        sample_files = sorted([f for f in zf.namelist() if f.startswith("samples/")])
        for sample_file in sample_files:
            try:
                data = json.loads(zf.read(sample_file).decode("utf-8"))
                samples.append(data)
            except Exception:
                continue
    return samples


def _collect_samples(results_dir: Path) -> list[dict]:
    eval_files = sorted(glob.glob(str(results_dir / "*.eval")))
    all_samples: list[dict] = []
    for eval_file in eval_files:
        all_samples.extend(_load_eval_samples(Path(eval_file)))
    return all_samples


def _winner_from_metadata(metadata: dict, scorer_value) -> str:
    blues_win = bool(metadata.get("blues_win", False))
    reds_win = bool(metadata.get("reds_win", False))

    if blues_win and not reds_win:
        return "blue"
    if reds_win and not blues_win:
        return "red"

    if scorer_value == 1:
        return "blue"
    if scorer_value == 0:
        return "red"
    return "draw"


def _rollout_from_sample(sample: dict, index: int) -> dict:
    scores = sample.get("scores", {})
    scorer = scores.get("red_vs_blue_scorer", {})
    metadata = scorer.get("metadata", {}) or {}

    roles = metadata.get("roles", {}) or {}
    patch_track = metadata.get("patch_track", {}) or {}

    winner = _winner_from_metadata(metadata, scorer.get("value"))

    per_player_scores: dict[str, float] = {}
    for pid, role in roles.items():
        is_blue_faction = role == "blue"
        if winner == "blue":
            per_player_scores[pid] = 1.0 if is_blue_faction else 0.0
        elif winner == "red":
            per_player_scores[pid] = 0.0 if is_blue_faction else 1.0
        else:
            per_player_scores[pid] = 0.5

    return {
        "rollout": index,
        "winner": winner,
        "rounds": metadata.get("rounds_played", 0),
        "total_reward": float(metadata.get("avg_belief_alignment", 0.0)),
        "blue_role_reward": float(patch_track.get("blue", 0)) / 6.0,
        "red_role_reward": float(patch_track.get("red", 0)) / 6.0,
        "per_player_scores": per_player_scores,
    }


def _build_score_matrix(rollout_results: list[dict], agent_labels: list[str]) -> tuple[np.ndarray, list[str]]:
    task_labels = [f"inspect_rollout_{r['rollout']}" for r in rollout_results]
    matrix = np.zeros((len(agent_labels), len(task_labels)), dtype=float)

    for col, rollout in enumerate(rollout_results):
        scores = rollout["per_player_scores"]
        for row, label in enumerate(agent_labels):
            matrix[row, col] = float(scores.get(label, 0.0))

    return matrix, task_labels


def _solve_polarix(agent_labels: list[str], task_labels: list[str], score_matrix: np.ndarray, normalizer: str) -> dict:
    game = plx.agent_vs_task_game(
        agents=np.asarray(agent_labels),
        tasks=np.asarray(task_labels),
        agent_vs_task=score_matrix,
        normalizer=normalizer,
    )
    result = plx.solve(game, plx.ce_maxent, disable_progress_bar=True)

    ratings = np.asarray(result.ratings[1], dtype=float)
    joint = np.asarray(result.joint, dtype=float)
    agent_marginal = joint.sum(axis=(0, 2))

    return {
        "solver": "ce_maxent",
        "normalizer": normalizer,
        "agent_ratings": {agent_labels[i]: float(ratings[i]) for i in range(len(agent_labels))},
        "agent_equilibrium_prob": {agent_labels[i]: float(agent_marginal[i]) for i in range(len(agent_labels))},
        "summary": {str(k): _to_jsonable(v) for k, v in result.summary.items()},
    }


def convert(
    results_dir: str,
    output_json: str,
    *,
    normalizer: str = "winrate",
    policy: str = "inspect-converted",
    model_name: str = "inspect-derived",
) -> str:
    results_path = Path(results_dir)
    if not results_path.exists():
        raise FileNotFoundError(f"Results directory not found: {results_path}")

    samples = _collect_samples(results_path)
    if not samples:
        raise FileNotFoundError(f"No .eval samples found in {results_path}")

    rollout_results = [_rollout_from_sample(sample, i) for i, sample in enumerate(samples)]

    # Build consistent agent set from all rollouts
    agent_set = set()
    for row in rollout_results:
        agent_set.update(row["per_player_scores"].keys())
    agent_labels = sorted(agent_set)

    if not agent_labels:
        raise RuntimeError("Could not infer any players/agents from Inspect metadata")

    score_matrix, task_labels = _build_score_matrix(rollout_results, agent_labels)
    polarix_result = _solve_polarix(agent_labels, task_labels, score_matrix, normalizer=normalizer)

    wins = {
        "blue": sum(1 for r in rollout_results if r["winner"] == "blue"),
        "red": sum(1 for r in rollout_results if r["winner"] == "red"),
        "draw": sum(1 for r in rollout_results if r["winner"] == "draw"),
    }

    summary = {
        "config": "inspect_eval_conversion",
        "rollouts": len(rollout_results),
        "policy": policy,
        "model": model_name,
        "model_base_url": None,
        "agents": agent_labels,
        "tasks": task_labels,
        "wins": wins,
        "avg_reward": float(np.mean([r["total_reward"] for r in rollout_results])) if rollout_results else 0.0,
        "score_matrix": score_matrix.tolist(),
        "polarix": polarix_result,
        "results": rollout_results,
    }

    out_path = Path(output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return str(out_path)


def main(results_dir: str | None = None, output_json: str | None = None):
    parser = argparse.ArgumentParser(description="Convert Inspect .eval outputs to Polarix benchmark summary")
    parser.add_argument("results_dir", nargs="?", default=results_dir, help="Directory with .eval files")
    parser.add_argument(
        "--output-json",
        default=output_json or "results_polarix_red_vs_blue/benchmark_summary_from_inspect.json",
        help="Path to write converted benchmark summary JSON",
    )
    parser.add_argument(
        "--normalizer",
        default="winrate",
        choices=["winrate", "ptp", "zscore"],
        help="Polarix normalization method",
    )
    parser.add_argument(
        "--model-name",
        default="inspect-derived",
        help="Model label to include in converted summary",
    )
    args = parser.parse_args([] if results_dir is not None else None)

    if not args.results_dir:
        raise ValueError("results_dir is required")

    out_path = convert(
        args.results_dir,
        args.output_json,
        normalizer=args.normalizer,
        model_name=args.model_name,
    )
    print(f"Converted Inspect results to Polarix summary: {out_path}")


if __name__ == "__main__":
    main()
