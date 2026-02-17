"""
role_utilization_analysis.py

Analyzes how effectively each player leverages their assigned role.
"""

from __future__ import annotations
import json
import sys
import zipfile
from pathlib import Path
from typing import Dict, List, Any
import re

from red_vs_blue.analysis.llm_client import generate_json_with_retries, get_model


def load_eval_file(eval_path: Path) -> List[Dict]:
    results = []
    try:
        with zipfile.ZipFile(eval_path, "r") as zf:
            sample_files = [f for f in zf.namelist() if f.startswith("samples/")]
            for sample_file in sorted(sample_files):
                try:
                    results.append(json.loads(zf.read(sample_file).decode("utf-8")))
                except Exception as e:
                    print(f"Warning: Failed to load {sample_file}: {e}")
    except Exception as e:
        print(f"Error reading {eval_path}: {e}")
    return results


def extract_player_reasoning(sample_data: Dict) -> Dict[str, List[str]]:
    player_reasoning: Dict[str, List[str]] = {}
    metadata = sample_data.get("scores", {}).get("red_vs_blue_scorer", {}).get("metadata", {})
    for log_entry in metadata.get("public_log", []):
        if isinstance(log_entry, str):
            match = re.match(r"(p\d+):\s*(.*)", log_entry)
            if match:
                player_reasoning.setdefault(match.group(1), []).append(match.group(2))
    return player_reasoning


def extract_game_context(sample_data: Dict) -> Dict[str, Any]:
    metadata = sample_data.get("scores", {}).get("red_vs_blue_scorer", {}).get("metadata", {})
    return {
        "rounds_played": metadata.get("rounds_played", 0),
        "blues_win": metadata.get("blues_win", False),
        "roles": metadata.get("roles", {}),
        "patch_track": metadata.get("patch_track", {"blue": 0, "red": 0}),
        "fired_players": metadata.get("fired_players", []),
    }


async def analyze_player_role_utilization(model, player_id: str, role: str, reasoning: List[str], game_context: Dict[str, Any]) -> Dict[str, Any]:
    reasoning_text = "\n".join([f"- {r}" for r in reasoning[:30]])

    prompt = f"""You are evaluating role utilization quality in a social deduction game.

GAME CONTEXT:
- Rounds: {game_context['rounds_played']}
- Winner: {('Blue' if game_context['blues_win'] else 'Red')}
- Patch Track: {game_context['patch_track']}

PLAYER:
- ID: {player_id}
- Assigned Role: {role}
- Fired: {player_id in game_context['fired_players']}

STATEMENTS:
{reasoning_text if reasoning else 'No statements recorded'}

Assess whether the player effectively used role-specific objectives and constraints.

Return JSON only:
{{
  "role_utilized": true/false,
  "utilization_score": 0.0-1.0,
  "role_alignment": "misaligned/neutral/well-aligned",
  "role_objectives_supported": ["..."],
  "role_objectives_missed": ["..."],
  "explanation": "brief rationale"
}}"""

    parsed = await generate_json_with_retries(
        model,
        prompt,
        max_tokens=900,
        temperature=0.3,
        retries=6,
        error_context=f"role utilization analysis for {player_id}",
    )
    if parsed is not None:
        return parsed

    return {
        "role_utilized": False,
        "utilization_score": 0.0,
        "role_alignment": "unknown",
        "role_objectives_supported": [],
        "role_objectives_missed": [],
        "explanation": "Analysis failed",
    }


async def analyze_game(game_num: int, sample_data: Dict, model) -> Dict[str, Any]:
    game_context = extract_game_context(sample_data)
    player_reasoning = extract_player_reasoning(sample_data)
    roles = game_context.get("roles", {})

    print(f"\n{'='*70}")
    print(f"GAME {game_num}: ROLE UTILIZATION ANALYSIS")
    print(f"{'='*70}")

    player_analysis = {}
    utilized_count = 0

    for player_id in sorted(roles.keys()):
        role = roles.get(player_id, "unknown")
        analysis = await analyze_player_role_utilization(model, player_id, role, player_reasoning.get(player_id, []), game_context)
        player_analysis[player_id] = analysis
        if analysis.get("role_utilized"):
            utilized_count += 1

    return {
        "game_num": game_num,
        "game_context": game_context,
        "player_analysis": player_analysis,
        "utilized_count": utilized_count,
    }


def generate_role_utilization_markdown_report(all_results: List[Dict]) -> str:
    total_players = sum(len(r.get("game_context", {}).get("roles", {})) for r in all_results)
    utilized = sum(r.get("utilized_count", 0) for r in all_results)
    rate = (utilized / total_players * 100) if total_players else 0

    md = [
        "# Role Utilization Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"- **Games Analyzed**: {len(all_results)}",
        f"- **Total Players**: {total_players}",
        f"- **Players with Strong Role Utilization**: {utilized} ({rate:.1f}%)",
        "",
    ]

    for result in all_results:
        game_num = result.get("game_num", "?")
        game_context = result.get("game_context", {})
        md.append(f"## Game {game_num}")
        md.append(f"- **Result**: {'Blues Won' if game_context.get('blues_win') else 'Reds Won'}")
        md.append(f"- **Players Utilizing Role Well**: {result.get('utilized_count', 0)}/{len(game_context.get('roles', {}))}")
        md.append("")

    return "\n".join(md)


async def main(eval_file: str, model_name: str = "ollama/gpt-oss:20b"):
    eval_path = Path(eval_file)
    if not eval_path.exists():
        print(f"Error: Eval file not found: {eval_file}")
        sys.exit(1)

    model = get_model(model_name)
    samples = load_eval_file(eval_path)
    if not samples:
        print("No samples found in eval file")
        sys.exit(1)

    all_results = [await analyze_game(i, sample, model) for i, sample in enumerate(samples, 1)]

    output_json = eval_path.parent / f"role_utilization_analysis_{eval_path.stem}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    output_md = eval_path.parent / f"role_utilization_analysis_{eval_path.stem}.md"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(generate_role_utilization_markdown_report(all_results))

    print(f"Detailed results saved to: {output_json}")
    print(f"Markdown report saved to: {output_md}")


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) < 2:
        print("Usage: python role_utilization_analysis.py <eval_file> [model_name]")
        sys.exit(1)

    eval_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "ollama/gpt-oss:20b"
    asyncio.run(main(eval_file, model_name))
