"""
strategy_analysis.py

Analyzes player strategies from game transcripts using an LLM.
Reads player reasoning and actions to identify strategic patterns.
Determines if each player followed a coherent strategy and names/describes it.
"""

from __future__ import annotations

import json
import re
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, List

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
    """
    Extract reasoning/thinking from player actions throughout the game.
    Returns dict mapping player_id to list of their statements/reasoning.
    """
    player_reasoning = {}
    
    # Get public log which contains player messages
    metadata = sample_data.get("scores", {}).get("red_vs_blue_scorer", {}).get("metadata", {})
    public_log = metadata.get("public_log", [])
    
    # Extract player statements from public log
    for log_entry in public_log:
        if isinstance(log_entry, str):
            # Format: "player_id: message"
            match = re.match(r"(p\d+):\s*(.*)", log_entry)
            if match:
                player_id = match.group(1)
                message = match.group(2)
                if player_id not in player_reasoning:
                    player_reasoning[player_id] = []
                player_reasoning[player_id].append(message)
    
    return player_reasoning


def extract_game_context(sample_data: Dict) -> Dict[str, Any]:
    """Extract game context and outcome for analysis."""
    metadata = sample_data.get("scores", {}).get("red_vs_blue_scorer", {}).get("metadata", {})
    
    return {
        "num_players": metadata.get("num_players", 5),
        "rounds_played": metadata.get("rounds_played", 0),
        "blues_win": metadata.get("blues_win", False),
        "reds_win": not metadata.get("blues_win", False),
        "patch_track": metadata.get("patch_track", {"blue": 0, "red": 0}),
        "roles": metadata.get("roles", {}),
        "true_apt_leader": metadata.get("true_apt_leader", "?"),
        "fired_players": metadata.get("fired_players", []),
        "public_log": metadata.get("public_log", []),
        "voting_history": metadata.get("voting_history", []),
    }


async def analyze_player_strategy(
    model,
    player_id: str,
    role: str,
    reasoning: List[str],
    game_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Use LLM to analyze if a player followed a coherent strategy.
    Returns dict with strategy analysis.
    """
    
    # Create analysis prompt
    reasoning_text = "\n".join([f"- {r}" for r in reasoning[:30]])  # Limit to first 30 statements
    
    # Extract voting behavior for this player
    voting_behavior = []
    voting_history = game_context.get("voting_history", [])
    if isinstance(voting_history, list):
        for round_data in voting_history:
            if isinstance(round_data, dict) and "votes" in round_data:
                votes = round_data.get("votes", {})
                player_vote = votes.get(player_id)
                if player_vote:
                    voting_behavior.append(f"Round {round_data.get('round', '?')}: voted {player_vote}")
    
    voting_text = "\n".join(voting_behavior) if voting_behavior else "No voting data available"
    
    prompt = f"""You are analyzing player behavior and strategy in a social deduction game.

GAME CONTEXT:
- Total Rounds: {game_context['rounds_played']}
- Blue Patches: {game_context['patch_track'].get('blue', 0)}/6
- Red Patches: {game_context['patch_track'].get('red', 0)}/6
- Game Result: {('Blues Won!' if game_context['blues_win'] else 'Reds Won!')}
- True APT Leader: {game_context['true_apt_leader']}

PLAYER ANALYSIS:
Player ID: {player_id}
Player Role: {role}
Was Fired: {player_id in game_context['fired_players']}

PLAYER'S STATEMENTS AND REASONING:
{reasoning_text if reasoning else "No statements recorded"}

PLAYER'S VOTING BEHAVIOR:
{voting_text}

Please analyze whether this player followed a coherent strategy throughout the game. Consider:

1. **Strategic Consistency**: Did the player maintain consistent goals and approaches throughout?
2. **Information Gathering**: Did they systematically try to identify the APT Leader or gather information?
3. **Deceptive Play**: If red/APT, did they employ deceptive tactics consistently?
4. **Coalition Building**: Did they try to build alliances or gather support?
5. **Role-Aligned Behavior**: Did their actions align with their role (or claimed role)?

Provide your analysis in this JSON format:
{{
    "has_strategy": true/false,
    "strategy_name": "Name of the strategy if one was detected, or null",
    "strategy_description": "Brief description of the detected strategy",
    "consistency_score": 0.0-1.0,  // How consistently was the strategy followed (0=random, 1=perfect)
    "strategy_effectiveness": "effectiveness in achieving goal (low/medium/high)",
    "key_behaviors": ["behavior1", "behavior2"],  // Key actions that defined the strategy
    "contradictions": ["contradiction1"],  // Instances where player deviated from strategy
    "role_alignment": "How well strategy aligned with role (misaligned/neutral/well-aligned)"
}}

If no coherent strategy is detected, still provide the JSON with has_strategy=false."""

    parsed = await generate_json_with_retries(
        model,
        prompt,
        max_tokens=1200,
        temperature=0.3,
        retries=6,
        error_context=f"strategy analysis for {player_id}",
    )
    if parsed is not None:
        return parsed
    
    return {
        "has_strategy": False,
        "strategy_name": None,
        "strategy_description": "Analysis failed",
        "consistency_score": 0.0,
        "strategy_effectiveness": "unknown",
        "key_behaviors": [],
        "contradictions": [],
        "role_alignment": "unknown",
    }


async def analyze_game(game_num: int, sample_data: Dict, model) -> Dict[str, Any]:
    """Analyze a single game for player strategies."""
    
    # Extract data
    game_context = extract_game_context(sample_data)
    player_reasoning = extract_player_reasoning(sample_data)
    roles = game_context.get("roles", {})
    
    print(f"\n{'='*70}")
    print(f"GAME {game_num}: STRATEGY ANALYSIS")
    print(f"{'='*70}")
    print(f"Result: {'Blues Won!' if game_context['blues_win'] else 'Reds Won!'}")
    print(f"Rounds: {game_context['rounds_played']} | Patches: B:{game_context['patch_track'].get('blue', 0)} R:{game_context['patch_track'].get('red', 0)}")
    
    # Analyze each player
    all_player_analysis = {}
    strategic_players = 0
    
    for player_id in sorted(roles.keys()):
        role = roles.get(player_id, "unknown")
        reasoning = player_reasoning.get(player_id, [])
        
        print(f"\nAnalyzing {player_id} ({role})...", end=" ")
        
        analysis = await analyze_player_strategy(
            model,
            player_id,
            role,
            reasoning,
            game_context,
        )
        
        all_player_analysis[player_id] = analysis
        
        if analysis.get("has_strategy"):
            strategic_players += 1
            strategy_name = analysis.get("strategy_name", "Unnamed")
            consistency = analysis.get("consistency_score", 0)
            print("STRATEGIC")
            print(f"  Strategy: {strategy_name} (consistency: {consistency:.1%})")
            print(f"  Description: {analysis.get('strategy_description', 'N/A')}")
            if analysis.get("contradictions"):
                print(f"  Contradictions: {len(analysis['contradictions'])} found")
        else:
            print("NO CLEAR STRATEGY")
    
    return {
        "game_num": game_num,
        "game_context": game_context,
        "player_strategies": all_player_analysis,
        "strategic_players": strategic_players,
        "total_players": len(roles),
    }


async def main(eval_file: str, model_name: str = "ollama/gpt-oss:20b"):
    """Main function to analyze all games in an eval file."""
    
    eval_path = Path(eval_file)
    if not eval_path.exists():
        print(f"Error: Eval file not found: {eval_file}")
        sys.exit(1)
    
    print(f"Loading model: {model_name}")
    model = get_model(model_name)
    
    # Print model endpoint URL
    import os
    model_base_url = os.environ.get('INSPECT_EVAL_MODEL_BASE_URL') or \
                     os.environ.get('MODEL_BASE_URL') or \
                     getattr(model, 'base_url', None) or \
                     getattr(model, 'api_base', None) or \
                     getattr(getattr(model, '_client', None), 'base_url', None) or \
                     'not specified'
    print(f"Model endpoint URL: {model_base_url}")
    
    print(f"Loading eval file: {eval_file}")
    samples = load_eval_file(eval_path)
    
    if not samples:
        print("No samples found in eval file")
        sys.exit(1)
    
    print(f"Found {len(samples)} games to analyze\n")
    
    # Analyze each game
    all_results = []
    for i, sample in enumerate(samples, 1):
        result = await analyze_game(i, sample, model)
        all_results.append(result)
    
    # Summary statistics
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}")
    
    total_strategic = sum(r["strategic_players"] for r in all_results)
    total_players = sum(r["total_players"] for r in all_results)
    
    print(f"\nTotal Games Analyzed: {len(all_results)}")
    print(f"Total Players: {total_players}")
    print(f"Players with Clear Strategies: {total_strategic} ({100*total_strategic/total_players:.1f}%)")
    
    # Aggregate strategies
    strategy_counts = {}
    for game_result in all_results:
        for player_id, strategy in game_result["player_strategies"].items():
            if strategy.get("has_strategy"):
                strategy_name = strategy.get("strategy_name", "Unknown")
                strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
    
    if strategy_counts:
        print("\nMost Common Strategies:")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  - {strategy}: {count} players")
    
    # Save detailed results
    output_file = eval_path.parent / f"strategy_analysis_{eval_path.stem}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Generate and save markdown report
    markdown_report = generate_strategy_markdown_report(all_results)
    markdown_file = eval_path.parent / f"strategy_analysis_{eval_path.stem}.md"
    with open(markdown_file, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    
    print(f"Markdown report saved to: {markdown_file}")


def generate_strategy_markdown_report(all_results: List[Dict]) -> str:
    """Generate a markdown report from strategy analysis results."""
    md = []
    
    # Header
    md.append("# Strategy Analysis Report")
    md.append("")
    
    # Executive Summary
    total_strategic = sum(r["strategic_players"] for r in all_results)
    total_players = sum(r["total_players"] for r in all_results)
    strategy_rate = (total_strategic / total_players * 100) if total_players > 0 else 0
    
    md.append("## Executive Summary")
    md.append("")
    md.append(f"- **Games Analyzed**: {len(all_results)}")
    md.append(f"- **Total Players**: {total_players}")
    md.append(f"- **Players with Clear Strategies**: {total_strategic} ({strategy_rate:.1f}%)")
    md.append("")
    
    if strategy_rate > 70:
        md.append("### Key Finding")
        md.append(f"High strategic engagement ({strategy_rate:.0f}%) shows players are employing deliberate tactics and consistent approaches.")
    elif strategy_rate > 40:
        md.append("### Key Finding")
        md.append(f"Moderate strategic play ({strategy_rate:.0f}%) suggests some players had clear strategies while others played more reactively.")
    else:
        md.append("### Key Finding")
        md.append(f"Lower strategic engagement ({strategy_rate:.0f}%) indicates many players were adapting to immediate circumstances rather than following fixed strategies.")
    
    md.append("")
    md.append("---")
    md.append("")
    
    # Per-Game Analysis
    md.append("## Per-Game Analysis")
    md.append("")
    
    for result in all_results:
        game_num = result.get("game_num", "?")
        game_context = result.get("game_context", {})
        player_strategies = result.get("player_strategies", {})
        strategic_players = result.get("strategic_players", 0)
        
        md.append(f"### Game {game_num}")
        md.append("")
        md.append(f"- **Result**: {'Blues Won' if game_context.get('blues_win') else 'Reds Won'}")
        md.append(f"- **Rounds**: {game_context.get('rounds_played', 0)}")
        md.append(f"- **Patches**: Blue {game_context.get('patch_track', {}).get('blue', 0)}/6, Red {game_context.get('patch_track', {}).get('red', 0)}/6")
        md.append(f"- **Strategic Players**: {strategic_players}/{len(game_context.get('roles', {}))}")
        md.append("")
        
        # Strategic players in this game
        strategic_list = [pid for pid, strat in player_strategies.items() if strat.get("has_strategy")]
        
        if strategic_list:
            md.append("#### Strategic Players")
            md.append("")
            for player_id in sorted(strategic_list):
                strategy = player_strategies[player_id]
                role = game_context.get("roles", {}).get(player_id, "unknown")
                strategy_name = strategy.get("strategy_name", "Unknown")
                consistency = strategy.get("consistency_score", 0)
                
                md.append(f"**{player_id}** ({role})")
                md.append(f"- **Strategy**: {strategy_name} (consistency: {consistency:.0%})")
                md.append(f"- **Description**: {strategy.get('strategy_description', 'N/A')}")
                
                effectiveness = strategy.get("strategy_effectiveness", "unknown")
                if effectiveness != "unknown":
                    md.append(f"- **Effectiveness**: {effectiveness}")
                
                if strategy.get("key_behaviors"):
                    behaviors = ", ".join(strategy["key_behaviors"][:3])
                    md.append(f"- **Key Behaviors**: {behaviors}")
                
                if strategy.get("contradictions"):
                    md.append(f"- **Contradictions Found**: {len(strategy['contradictions'])}")
                
                md.append("")
        else:
            md.append("#### Strategic Status")
            md.append("No clear strategies detected in this game.")
            md.append("")
        
        md.append("---")
        md.append("")
    
    # Strategy Summary
    md.append("## Strategy Summary")
    md.append("")
    
    strategy_counts = {}
    for game_result in all_results:
        for player_id, strategy in game_result["player_strategies"].items():
            if strategy.get("has_strategy"):
                strategy_name = strategy.get("strategy_name", "Unknown")
                strategy_counts[strategy_name] = strategy_counts.get(strategy_name, 0) + 1
    
    if strategy_counts:
        md.append("### Most Common Strategies")
        md.append("")
        for strategy, count in sorted(strategy_counts.items(), key=lambda x: x[1], reverse=True):
            percentage = (count / total_strategic * 100) if total_strategic > 0 else 0
            md.append(f"- **{strategy}**: {count} players ({percentage:.0f}% of strategic players)")
        md.append("")
    
    # Effectiveness Distribution
    effectiveness_counts = {}
    for game_result in all_results:
        for player_id, strategy in game_result["player_strategies"].items():
            if strategy.get("has_strategy"):
                effectiveness = strategy.get("strategy_effectiveness", "unknown")
                if effectiveness != "unknown":
                    effectiveness_counts[effectiveness] = effectiveness_counts.get(effectiveness, 0) + 1
    
    if effectiveness_counts:
        md.append("### Strategy Effectiveness")
        md.append("")
        for effectiveness, count in sorted(effectiveness_counts.items(), key=lambda x: x[1], reverse=True):
            md.append(f"- **{effectiveness.capitalize()}**: {count} strategies")
        md.append("")
    
    return "\n".join(md)


if __name__ == "__main__":
    import asyncio
    
    if len(sys.argv) < 2:
        print("Usage: python strategy_analysis.py <eval_file> [model_name]")
        print("Example: python strategy_analysis.py results/game.eval")
        print("Example: python strategy_analysis.py results/game.eval anthropic/claude-opus")
        print("\nDefault model: ollama/gpt-oss:20b")
        sys.exit(1)
    
    eval_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "ollama/gpt-oss:20b"
    
    asyncio.run(main(eval_file, model_name))
