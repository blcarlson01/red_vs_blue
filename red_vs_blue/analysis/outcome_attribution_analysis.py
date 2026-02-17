"""
outcome_attribution_analysis.py

Analyzes player-level contributions to game outcomes.
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
        "voting_history": metadata.get("voting_history", []),
    }


async def analyze_player_outcome_contribution(model, player_id: str, role: str, reasoning: List[str], game_context: Dict[str, Any]) -> Dict[str, Any]:
    reasoning_text = "\n".join([f"- {r}" for r in reasoning[:30]])

    prompt = f"""You are attributing game outcome impact in a social deduction game.

CONTEXT:
- Winner: {('Blue' if game_context['blues_win'] else 'Red')}
- Rounds: {game_context['rounds_played']}
- Patch Track: {game_context['patch_track']}

PLAYER:
- ID: {player_id}
- Role: {role}
- Fired: {player_id in game_context['fired_players']}

STATEMENTS:
{reasoning_text if reasoning else 'No statements recorded'}

Assess this player's impact on final outcome.

Return JSON only:
{{
  "impact_score": -1.0-1.0,
  "impact_direction": "helped_blue/helped_red/neutral",
  "impact_magnitude": "low/medium/high",
  "key_contributions": ["..."],
  "key_mistakes": ["..."],
  "explanation": "brief rationale"
}}"""

    parsed = await generate_json_with_retries(
        model,
        prompt,
        max_tokens=900,
        temperature=0.3,
        retries=6,
        error_context=f"outcome attribution analysis for {player_id}",
    )
    if parsed is not None:
        return parsed

    return {
        "impact_score": 0.0,
        "impact_direction": "neutral",
        "impact_magnitude": "unknown",
        "key_contributions": [],
        "key_mistakes": [],
        "explanation": "Analysis failed",
    }


async def analyze_game(game_num: int, sample_data: Dict, model) -> Dict[str, Any]:
    game_context = extract_game_context(sample_data)
    player_reasoning = extract_player_reasoning(sample_data)
    roles = game_context.get("roles", {})

    print(f"\n{'='*70}")
    print(f"GAME {game_num}: OUTCOME ATTRIBUTION ANALYSIS")
    print(f"{'='*70}")

    player_analysis = {}
    for player_id in sorted(roles.keys()):
        role = roles.get(player_id, "unknown")
        player_analysis[player_id] = await analyze_player_outcome_contribution(
            model, player_id, role, player_reasoning.get(player_id, []), game_context
        )

    return {
        "game_num": game_num,
        "game_context": game_context,
        "player_analysis": player_analysis,
    }


def generate_outcome_attribution_markdown_report(all_results: List[Dict]) -> str:
    md = [
        "# Outcome Attribution Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"- **Games Analyzed**: {len(all_results)}",
        "",
    ]

    for result in all_results:
        game_num = result.get("game_num", "?")
        game_context = result.get("game_context", {})
        md.append(f"## Game {game_num}")
        md.append(f"- **Result**: {'Blues Won' if game_context.get('blues_win') else 'Reds Won'}")
        md.append("")
        for player_id, analysis in sorted(result.get("player_analysis", {}).items()):
            role = game_context.get("roles", {}).get(player_id, "unknown")
            md.append(f"**{player_id}** ({role})")
            md.append(f"- Impact Score: {analysis.get('impact_score', 0.0)}")
            md.append(f"- Direction: {analysis.get('impact_direction', 'neutral')}")
            md.append(f"- Magnitude: {analysis.get('impact_magnitude', 'unknown')}")
            md.append(f"- Explanation: {analysis.get('explanation', 'N/A')}")
            md.append("")
        md.append("---")
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

    output_json = eval_path.parent / f"outcome_attribution_analysis_{eval_path.stem}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    output_md = eval_path.parent / f"outcome_attribution_analysis_{eval_path.stem}.md"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(generate_outcome_attribution_markdown_report(all_results))

    print(f"Detailed results saved to: {output_json}")
    print(f"Markdown report saved to: {output_md}")


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) < 2:
        print("Usage: python outcome_attribution_analysis.py <eval_file> [model_name]")
        sys.exit(1)

    eval_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "ollama/gpt-oss:20b"
    asyncio.run(main(eval_file, model_name))
