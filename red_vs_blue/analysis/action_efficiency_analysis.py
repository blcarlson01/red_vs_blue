"""
action_efficiency_analysis.py

Analyzes how efficiently players convert discussion and voting actions into
role-aligned progress toward win conditions.
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
    """Load all samples from an Inspect .eval ZIP file."""
    results = []
    try:
        with zipfile.ZipFile(eval_path, "r") as zf:
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


def extract_player_reasoning(sample_data: Dict) -> Dict[str, List[str]]:
    """Extract player statements from the public log."""
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
    """Extract game context and outcome for analysis."""
    metadata = sample_data.get("scores", {}).get("red_vs_blue_scorer", {}).get("metadata", {})
    return {
        "num_players": metadata.get("num_players", 5),
        "rounds_played": metadata.get("rounds_played", 0),
        "blues_win": metadata.get("blues_win", False),
        "patch_track": metadata.get("patch_track", {"blue": 0, "red": 0}),
        "roles": metadata.get("roles", {}),
        "fired_players": metadata.get("fired_players", []),
        "voting_history": metadata.get("voting_history", []),
    }


async def analyze_player_action_efficiency(
    model,
    player_id: str,
    role: str,
    reasoning: List[str],
    game_context: Dict[str, Any],
) -> Dict[str, Any]:
    """Use LLM to assess player-level action efficiency."""
    reasoning_text = "\n".join([f"- {r}" for r in reasoning[:30]])

    voting_behavior = []
    for round_data in game_context.get("voting_history", []):
        if isinstance(round_data, dict) and "votes" in round_data:
            votes = round_data.get("votes", {})
            player_vote = votes.get(player_id)
            if player_vote:
                voting_behavior.append(f"Round {round_data.get('round', '?')}: voted {player_vote}")
    voting_text = "\n".join(voting_behavior) if voting_behavior else "No voting data available"

    prompt = f"""You are analyzing player behavior efficiency in a social deduction game.

GAME CONTEXT:
- Rounds Played: {game_context['rounds_played']}
- Patch Track: Blue={game_context['patch_track'].get('blue', 0)}, Red={game_context['patch_track'].get('red', 0)}
- Winner: {('Blue' if game_context['blues_win'] else 'Red')}

PLAYER:
- ID: {player_id}
- Role: {role}
- Fired: {player_id in game_context['fired_players']}

PLAYER STATEMENTS:
{reasoning_text if reasoning else 'No statements recorded'}

PLAYER VOTING:
{voting_text}

Evaluate action efficiency:
1. Did the player take actions that directly supported role objectives?
2. Did they avoid redundant/noise-heavy discussion?
3. Were votes and claims timely and coherent?
4. Did actions produce useful information or progress?

Return JSON only in this format:
{{
  "efficient_action_taker": true/false,
  "efficiency_score": 0.0-1.0,
  "signal_to_noise": "low/medium/high",
  "decision_quality": "low/medium/high",
  "effective_actions": ["..."],
  "wasted_actions": ["..."],
  "explanation": "brief rationale"
}}"""

    parsed = await generate_json_with_retries(
        model,
        prompt,
        max_tokens=1000,
        temperature=0.3,
        retries=6,
        error_context=f"action efficiency analysis for {player_id}",
    )
    if parsed is not None:
        return parsed

    return {
        "efficient_action_taker": False,
        "efficiency_score": 0.0,
        "signal_to_noise": "unknown",
        "decision_quality": "unknown",
        "effective_actions": [],
        "wasted_actions": [],
        "explanation": "Analysis failed",
    }


async def analyze_game(game_num: int, sample_data: Dict, model) -> Dict[str, Any]:
    """Analyze one game for action efficiency."""
    game_context = extract_game_context(sample_data)
    player_reasoning = extract_player_reasoning(sample_data)
    roles = game_context.get("roles", {})

    print(f"\n{'='*70}")
    print(f"GAME {game_num}: ACTION EFFICIENCY ANALYSIS")
    print(f"{'='*70}")

    player_analysis = {}
    efficient_count = 0

    for player_id in sorted(roles.keys()):
        role = roles.get(player_id, "unknown")
        reasoning = player_reasoning.get(player_id, [])
        print(f"Analyzing {player_id} ({role})...", end=" ")
        analysis = await analyze_player_action_efficiency(model, player_id, role, reasoning, game_context)
        player_analysis[player_id] = analysis
        if analysis.get("efficient_action_taker"):
            efficient_count += 1
            print(f"EFFICIENT ({analysis.get('efficiency_score', 0):.0%})")
        else:
            print("INEFFICIENT/UNCLEAR")

    return {
        "game_num": game_num,
        "game_context": game_context,
        "player_analysis": player_analysis,
        "efficient_count": efficient_count,
    }


def generate_action_efficiency_markdown_report(all_results: List[Dict]) -> str:
    """Generate markdown summary for action efficiency analysis."""
    total_players = sum(len(r.get("game_context", {}).get("roles", {})) for r in all_results)
    total_efficient = sum(r.get("efficient_count", 0) for r in all_results)
    efficiency_rate = (total_efficient / total_players * 100) if total_players else 0

    md = [
        "# Action Efficiency Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"- **Games Analyzed**: {len(all_results)}",
        f"- **Total Players**: {total_players}",
        f"- **Efficient Players**: {total_efficient} ({efficiency_rate:.1f}%)",
        "",
        "## Per-Game Analysis",
        "",
    ]

    for result in all_results:
        game_num = result.get("game_num", "?")
        game_context = result.get("game_context", {})
        player_analysis = result.get("player_analysis", {})
        md.append(f"### Game {game_num}")
        md.append("")
        md.append(f"- **Result**: {'Blues Won' if game_context.get('blues_win') else 'Reds Won'}")
        md.append(f"- **Efficient Players**: {result.get('efficient_count', 0)}/{len(game_context.get('roles', {}))}")
        md.append("")
        for player_id in sorted(player_analysis.keys()):
            analysis = player_analysis[player_id]
            role = game_context.get("roles", {}).get(player_id, "unknown")
            md.append(f"**{player_id}** ({role})")
            md.append(f"- Efficiency Score: {analysis.get('efficiency_score', 0):.0%}")
            md.append(f"- Decision Quality: {analysis.get('decision_quality', 'unknown')}")
            md.append(f"- Explanation: {analysis.get('explanation', 'N/A')}")
            md.append("")
        md.append("---")
        md.append("")

    return "\n".join(md)


async def main(eval_file: str, model_name: str = "ollama/gpt-oss:20b"):
    """Main function to analyze all games in an eval file."""
    eval_path = Path(eval_file)
    if not eval_path.exists():
        print(f"Error: Eval file not found: {eval_file}")
        sys.exit(1)

    print(f"Loading model: {model_name}")
    model = get_model(model_name)

    samples = load_eval_file(eval_path)
    if not samples:
        print("No samples found in eval file")
        sys.exit(1)

    all_results = []
    for i, sample in enumerate(samples, 1):
        all_results.append(await analyze_game(i, sample, model))

    output_json = eval_path.parent / f"action_efficiency_analysis_{eval_path.stem}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"Detailed results saved to: {output_json}")

    output_md = eval_path.parent / f"action_efficiency_analysis_{eval_path.stem}.md"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(generate_action_efficiency_markdown_report(all_results))
    print(f"Markdown report saved to: {output_md}")


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) < 2:
        print("Usage: python action_efficiency_analysis.py <eval_file> [model_name]")
        sys.exit(1)

    eval_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "ollama/gpt-oss:20b"
    asyncio.run(main(eval_file, model_name))
