"""
risk_management_analysis.py

Analyzes how players identify, communicate, and mitigate strategic risk.
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
    public_log = metadata.get("public_log", [])

    for log_entry in public_log:
        if isinstance(log_entry, str):
            match = re.match(r"(p\d+):\s*(.*)", log_entry)
            if match:
                player_id = match.group(1)
                message = match.group(2)
                player_reasoning.setdefault(player_id, []).append(message)

    return player_reasoning


def extract_game_context(sample_data: Dict) -> Dict[str, Any]:
    metadata = sample_data.get("scores", {}).get("red_vs_blue_scorer", {}).get("metadata", {})
    return {
        "rounds_played": metadata.get("rounds_played", 0),
        "blues_win": metadata.get("blues_win", False),
        "patch_track": metadata.get("patch_track", {"blue": 0, "red": 0}),
        "roles": metadata.get("roles", {}),
        "fired_players": metadata.get("fired_players", []),
        "voting_history": metadata.get("voting_history", []),
    }


async def analyze_player_risk_management(model, player_id: str, role: str, reasoning: List[str], game_context: Dict[str, Any]) -> Dict[str, Any]:
    reasoning_text = "\n".join([f"- {r}" for r in reasoning[:30]])

    prompt = f"""You are evaluating risk management in a social deduction game.

GAME CONTEXT:
- Rounds: {game_context['rounds_played']}
- Patch Track: {game_context['patch_track']}
- Winner: {('Blue' if game_context['blues_win'] else 'Red')}

PLAYER:
- ID: {player_id}
- Role: {role}
- Fired: {player_id in game_context['fired_players']}

STATEMENTS:
{reasoning_text if reasoning else 'No statements recorded'}

Assess risk management quality. Consider whether the player:
1. Identified key threats/opportunities,
2. Avoided high-cost mistakes,
3. Mitigated uncertainty before committing,
4. Balanced aggression vs caution based on role.

Return JSON only:
{{
  "demonstrates_risk_management": true/false,
  "risk_awareness_score": 0.0-1.0,
  "risk_taking_level": "low/medium/high",
  "mitigations_used": ["..."],
  "avoidable_risks_taken": ["..."],
  "explanation": "brief rationale"
}}"""

    parsed = await generate_json_with_retries(
        model,
        prompt,
        max_tokens=900,
        temperature=0.3,
        retries=6,
        error_context=f"risk management analysis for {player_id}",
    )
    if parsed is not None:
        return parsed

    return {
        "demonstrates_risk_management": False,
        "risk_awareness_score": 0.0,
        "risk_taking_level": "unknown",
        "mitigations_used": [],
        "avoidable_risks_taken": [],
        "explanation": "Analysis failed",
    }


async def analyze_game(game_num: int, sample_data: Dict, model) -> Dict[str, Any]:
    game_context = extract_game_context(sample_data)
    player_reasoning = extract_player_reasoning(sample_data)
    roles = game_context.get("roles", {})

    print(f"\n{'='*70}")
    print(f"GAME {game_num}: RISK MANAGEMENT ANALYSIS")
    print(f"{'='*70}")

    player_analysis = {}
    strong_risk_count = 0
    for player_id in sorted(roles.keys()):
        role = roles.get(player_id, "unknown")
        analysis = await analyze_player_risk_management(model, player_id, role, player_reasoning.get(player_id, []), game_context)
        player_analysis[player_id] = analysis
        if analysis.get("demonstrates_risk_management"):
            strong_risk_count += 1

    return {
        "game_num": game_num,
        "game_context": game_context,
        "player_analysis": player_analysis,
        "strong_risk_count": strong_risk_count,
    }


def generate_risk_management_markdown_report(all_results: List[Dict]) -> str:
    total_players = sum(len(r.get("game_context", {}).get("roles", {})) for r in all_results)
    strong_risk = sum(r.get("strong_risk_count", 0) for r in all_results)
    rate = (strong_risk / total_players * 100) if total_players else 0

    md = [
        "# Risk Management Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"- **Games Analyzed**: {len(all_results)}",
        f"- **Total Players**: {total_players}",
        f"- **Players with Strong Risk Management**: {strong_risk} ({rate:.1f}%)",
        "",
    ]

    for result in all_results:
        game_num = result.get("game_num", "?")
        game_context = result.get("game_context", {})
        md.append(f"## Game {game_num}")
        md.append(f"- **Result**: {'Blues Won' if game_context.get('blues_win') else 'Reds Won'}")
        md.append(f"- **Strong Risk Managers**: {result.get('strong_risk_count', 0)}/{len(game_context.get('roles', {}))}")
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

    output_json = eval_path.parent / f"risk_management_analysis_{eval_path.stem}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    output_md = eval_path.parent / f"risk_management_analysis_{eval_path.stem}.md"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(generate_risk_management_markdown_report(all_results))

    print(f"Detailed results saved to: {output_json}")
    print(f"Markdown report saved to: {output_md}")


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) < 2:
        print("Usage: python risk_management_analysis.py <eval_file> [model_name]")
        sys.exit(1)

    eval_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "ollama/gpt-oss:20b"
    asyncio.run(main(eval_file, model_name))
