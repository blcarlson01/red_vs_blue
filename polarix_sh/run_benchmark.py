from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import numbers
from pathlib import Path
import sys

import numpy as np
import polarix as plx
import yaml
from inspect_ai.model import get_model


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from red_vs_blue.agents import create_agents


def _load_task(task_path: str):
    module_name, class_name = task_path.split(":", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def _winner_for_role(role: str) -> str:
    return "blue" if role == "blue" else "red"


def _default_actions(task_env):
    env = task_env.env
    phase = env.current_phase
    actions: dict[str, dict] = {}

    if phase == "discussion":
        for pid in env.player_ids:
            if pid not in env.fired_players:
                actions[pid] = {"message": f"{pid} analysis update"}

    elif phase == "nomination":
        ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
        candidates = [p for p in env.player_ids if p != ciso and p not in env.fired_players]
        actions[ciso] = {"nomination": candidates[0]} if candidates else {}

    elif phase == "voting":
        employed = [p for p in env.player_ids if p not in env.fired_players]
        for index, pid in enumerate(employed):
            actions[pid] = {"vote": "yes" if index < max(1, len(employed) // 2 + 1) else "no"}

    elif phase in ["legislative", "legislative_ciso"]:
        ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
        actions[ciso] = {"discard_patch": 0}

    elif phase == "legislative_soc":
        if env.nominated_soc_lead:
            actions[env.nominated_soc_lead] = {"discard_patch": 0}

    elif phase == "power":
        ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
        targets = [p for p in env.player_ids if p != ciso and p not in env.fired_players]
        red_count = env.patch_track["red"]
        if red_count >= 4 and targets:
            actions[ciso] = {"power_action": {"fire": targets[0]}}
        elif red_count == 3 and targets:
            actions[ciso] = {"power_action": {"special_election": targets[0]}}
        elif red_count == 2 and targets:
            actions[ciso] = {"power_action": {"investigate": targets[0]}}
        else:
            actions[ciso] = {"power_action": {}}

    return actions


def _phase_actor_ids(env) -> list[str]:
    if env.current_phase == "discussion":
        return [pid for pid in env.player_ids if pid not in env.fired_players]
    if env.current_phase == "nomination":
        return [env.player_ids[env.current_ciso_idx % len(env.player_ids)]]
    if env.current_phase == "voting":
        return [pid for pid in env.player_ids if pid not in env.fired_players]
    if env.current_phase in ["legislative", "legislative_ciso", "power"]:
        return [env.player_ids[env.current_ciso_idx % len(env.player_ids)]]
    if env.current_phase == "legislative_soc" and env.nominated_soc_lead:
        return [env.nominated_soc_lead]
    return []


async def _model_actions(task_env, agents_by_id: dict[str, object]) -> dict[str, dict]:
    env = task_env.env
    actions: dict[str, dict] = {}

    for pid in _phase_actor_ids(env):
        obs = env.observe(pid)
        action = await agents_by_id[pid].act(obs)
        actions[pid] = action if isinstance(action, dict) else {}

    return actions


def _run_single_rollout(task, *, policy: str = "model", agents_by_id: dict[str, object] | None = None):
    state = task.initial_state()
    done = False
    total_reward = 0.0
    last_info = {}

    if policy == "model" and agents_by_id is None:
        raise ValueError("agents_by_id is required when policy='model'")

    while not done:
        if policy == "model":
            actions = asyncio.run(_model_actions(task.env, agents_by_id))
        else:
            actions = _default_actions(task.env)
        state, reward, done, info = task.transition(state, actions)
        total_reward += float(reward)
        last_info = info

    wrapped_env = task.env
    inner_env = wrapped_env.env
    winner = last_info.get("winner")

    per_player_scores: dict[str, float] = {}
    for pid in inner_env.player_ids:
        role = inner_env.roles.get(pid, "blue")
        if winner in {"blue", "red"}:
            per_player_scores[pid] = 1.0 if _winner_for_role(role) == winner else 0.0
        else:
            per_player_scores[pid] = 0.5

    return {
        "winner": winner,
        "rounds": last_info.get("round"),
        "total_reward": total_reward,
        "blue_role_reward": last_info.get("blue_role_reward", 0.0),
        "red_role_reward": last_info.get("red_role_reward", 0.0),
        "per_player_scores": per_player_scores,
    }


def _build_score_matrix(rollout_results: list[dict], agent_labels: list[str]) -> tuple[np.ndarray, list[str]]:
    task_labels = [f"rollout_{r['rollout']}" for r in rollout_results]
    matrix = np.zeros((len(agent_labels), len(task_labels)), dtype=float)

    for col, rollout in enumerate(rollout_results):
        scores = rollout["per_player_scores"]
        for row, label in enumerate(agent_labels):
            matrix[row, col] = float(scores.get(label, 0.0))

    return matrix, task_labels


def _solve_polarix(agent_labels: list[str], task_labels: list[str], score_matrix: np.ndarray, *, normalizer: str = "winrate"):
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
            converted = value.tolist()
            return _to_jsonable(converted)
        try:
            return float(value)
        except Exception:
            return str(value)

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


def _init_agents_for_rollout(task, *, model_name: str, model_base_url: str | None = None):
    model = get_model(model_name, base_url=model_base_url)
    player_ids = list(task.env.env.player_ids)
    agents = create_agents(model=model, player_ids=player_ids)
    for agent in agents:
        agent.role = task.env.env.roles.get(agent.player_id)
    return {agent.player_id: agent for agent in agents}


def run(
    config_path: str,
    *,
    policy: str | None = None,
    model_name: str | None = None,
    model_base_url: str | None = None,
):
    cfg = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))

    task_cls = _load_task(cfg["task"])
    env_cfg = cfg.get("env", {})
    rollouts = int(cfg.get("rollouts", 1))
    polarix_cfg = cfg.get("polarix", {})
    normalizer = polarix_cfg.get("normalizer", "winrate")
    log_dir = Path(cfg.get("log_dir", "results_polarix_red_vs_blue"))
    log_dir.mkdir(parents=True, exist_ok=True)

    configured_models = cfg.get("models", [])
    cfg_policy = cfg.get("policy", "model")
    selected_policy = policy or cfg_policy

    selected_model_name = model_name
    if selected_model_name is None:
        if isinstance(configured_models, list) and configured_models:
            selected_model_name = str(configured_models[0])
        else:
            selected_model_name = "ollama/gpt-oss:20b"

    selected_model_base_url = model_base_url or cfg.get("model_base_url")

    rollout_results = []
    default_agent_labels: list[str] | None = None

    for rollout in range(rollouts):
        rollout_env_cfg = dict(env_cfg)
        if "seed" in rollout_env_cfg and rollout_env_cfg["seed"] is not None:
            rollout_env_cfg["seed"] = int(rollout_env_cfg["seed"]) + rollout

        task = task_cls(rollout_env_cfg)
        if default_agent_labels is None:
            default_agent_labels = list(task.env.env.player_ids)

        agents_by_id = None
        if selected_policy == "model":
            agents_by_id = _init_agents_for_rollout(
                task,
                model_name=selected_model_name,
                model_base_url=selected_model_base_url,
            )

        rollout_outcome = _run_single_rollout(
            task,
            policy=selected_policy,
            agents_by_id=agents_by_id,
        )

        rollout_results.append(
            {
                "rollout": rollout,
                **rollout_outcome,
            }
        )

    if not default_agent_labels:
        raise RuntimeError("No rollouts executed; unable to build Polarix score matrix")

    configured_labels = cfg.get("models", [])
    agent_labels = default_agent_labels
    if isinstance(configured_labels, list) and len(configured_labels) == len(default_agent_labels):
        agent_labels = [str(label) for label in configured_labels]
        for row in rollout_results:
            renamed = {}
            for index, pid in enumerate(default_agent_labels):
                renamed[agent_labels[index]] = row["per_player_scores"].get(pid, 0.0)
            row["per_player_scores"] = renamed

    score_matrix, task_labels = _build_score_matrix(rollout_results, agent_labels)
    polarix_result = _solve_polarix(agent_labels, task_labels, score_matrix, normalizer=normalizer)

    summary = {
        "config": config_path,
        "rollouts": rollouts,
        "policy": selected_policy,
        "model": selected_model_name,
        "model_base_url": selected_model_base_url,
        "agents": agent_labels,
        "tasks": task_labels,
        "wins": {
            "blue": sum(1 for r in rollout_results if r["winner"] == "blue"),
            "red": sum(1 for r in rollout_results if r["winner"] == "red"),
            "draw": sum(1 for r in rollout_results if r["winner"] == "draw"),
        },
        "avg_reward": sum(r["total_reward"] for r in rollout_results) / max(1, len(rollout_results)),
        "score_matrix": score_matrix.tolist(),
        "polarix": polarix_result,
        "results": rollout_results,
    }

    out_path = log_dir / "benchmark_summary.json"
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Run Polarix SH benchmark rollouts")
    parser.add_argument("--config", default="configs/sh_5p.yaml", help="Path to benchmark YAML config")
    parser.add_argument(
        "--policy",
        choices=["model", "heuristic"],
        default=None,
        help="Action policy for rollouts (default from config, otherwise model)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model name for model policy (e.g., ollama/gpt-oss:20b)",
    )
    parser.add_argument(
        "--model-base-url",
        default=None,
        help="Optional model base URL (e.g., http://localhost:11434/v1)",
    )
    args = parser.parse_args()

    out_path = run(
        args.config,
        policy=args.policy,
        model_name=args.model,
        model_base_url=args.model_base_url,
    )
    print(f"Benchmark complete. Summary written to {out_path}")


if __name__ == "__main__":
    main()
