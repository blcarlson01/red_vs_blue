"""
confusion_analysis.py

Analyzes player confusion from game transcripts using an LLM.
Reads reasoning from agent actions and determines if players showed confusion.
Provides suggestions for game improvements based on confusion patterns found.
"""

from __future__ import annotations
import json
import sys
import glob
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
    }


async def analyze_player_confusion(
    model,
    player_id: str,
    role: str,
    reasoning: List[str],
    game_context: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Use LLM to analyze if a player showed confusion and why.
    Returns dict with confusion analysis.
    """
    
    # Create analysis prompt
    reasoning_text = "\n".join([f"- {r}" for r in reasoning[:20]])  # Limit to first 20 statements
    
    prompt = f"""You are analyzing player behavior in a social deduction game.

GAME CONTEXT:
- Total Rounds: {game_context['rounds_played']}
- Blue Patches: {game_context['patch_track'].get('blue', 0)}/6
- Red Patches: {game_context['patch_track'].get('red', 0)}/6
- Game Result: {('Blues Won!' if game_context['blues_win'] else 'Reds Won!')}
- Fired Players: {', '.join(game_context['fired_players']) if game_context['fired_players'] else 'None'}

PLAYER ANALYSIS:
Player ID: {player_id}
Player Role: {role}

PLAYER'S STATEMENTS AND REASONING:
{reasoning_text if reasoning else "No statements recorded"}

Please analyze the above player's behavior for signs of confusion. Consider:

1. **Logical Inconsistency**: Did the player contradict themselves or change positions without explanation?
2. **Misunderstanding of Game Rules**: Did they seem confused about how voting, patches, roles, or powers work?
3. **Misunderstanding of State**: Did they seem confused about the current game state (round, patches, who's fired)?
4. **Strategic Confusion**: Did they make moves that don't align with their stated goals or role?
5. **Role Confusion**: Did they act inconsistently with their claimed or actual role?

Provide your analysis in this JSON format:
{{
    "confused": true/false,
    "confusion_types": ["type1", "type2"],  // If confused, list types from above
    "explanation": "Brief explanation of what confusion was observed, if any",
    "evidence": ["quote1", "quote2"],  // Specific statements showing confusion if any
    "improvement_suggestions": ["suggestion1", "suggestion2"]  // How to help prevent this confusion
}}

If no confusion is detected, still provide the JSON with confused=false and a brief note that the player was not confused."""

    parsed = await generate_json_with_retries(
        model,
        prompt,
        max_tokens=1000,
        temperature=0.3,
        retries=6,
        error_context=f"confusion analysis for {player_id}",
    )
    if parsed is not None:
        return parsed
    
    return {
        "confused": False,
        "confusion_types": [],
        "explanation": "Analysis failed",
        "evidence": [],
        "improvement_suggestions": [],
    }


async def analyze_game_improvements(
    model,
    game_context: Dict[str, Any],
    all_player_analysis: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Use LLM to suggest game improvements based on confusion patterns.
    """
    
    # Summarize confusion found
    confused_players = [p for p, a in all_player_analysis.items() if a.get("confused")]
    
    confusion_types_all = []
    for analysis in all_player_analysis.values():
        confusion_types_all.extend(analysis.get("confusion_types", []))
    
    from collections import Counter
    confusion_counts = Counter(confusion_types_all)
    
    prompt = f"""You are a game designer analyzing player confusion in a social deduction game.

GAME OVERVIEW:
- Result: {'Blues Won!' if game_context['blues_win'] else 'Reds Won!'}
- Rounds Played: {game_context['rounds_played']}
- Final Patch Track: {game_context['patch_track']}
- Total Players: {game_context['num_players']}

CONFUSION SUMMARY:
- Players who showed confusion: {len(confused_players)}/{game_context['num_players']}
- {dict(confusion_counts) if confusion_counts else 'No confusion detected'}

Based on the player-level confusion analysis provided, suggest 3-5 specific, actionable improvements to the game design or rules that could reduce player confusion and improve gameplay clarity.

Consider:
1. Rule clarity and communication
2. Game state visibility and information flow
3. Strategic decision-making support
4. Tutorial or onboarding improvements
5. UI/presentation of game information

Provide suggestions in JSON format:
{{
    "overall_confusion_level": "low/medium/high",
    "key_insights": ["insight1", "insight2"],
    "improvement_suggestions": [
        {{
            "category": "Rules/Information/Strategy/Tutorial",
            "suggestion": "Specific improvement",
            "rationale": "Why this helps",
            "implementation": "How to implement this"
        }}
    ]
}}"""

    parsed = await generate_json_with_retries(
        model,
        prompt,
        max_tokens=1500,
        temperature=0.3,
        retries=6,
        error_context="game-level confusion improvements",
    )
    if parsed is not None:
        return parsed
    
    return {
        "overall_confusion_level": "unknown",
        "key_insights": [],
        "improvement_suggestions": [],
    }


async def analyze_game(game_num: int, sample_data: Dict, model) -> Dict[str, Any]:
    """Analyze a single game for player confusion."""
    
    # Extract data
    game_context = extract_game_context(sample_data)
    player_reasoning = extract_player_reasoning(sample_data)
    roles = game_context.get("roles", {})
    
    print(f"\n{'='*70}")
    print(f"GAME {game_num}: CONFUSION ANALYSIS")
    print(f"{'='*70}")
    print(f"Result: {'Blues Won!' if game_context['blues_win'] else 'Reds Won!'}")
    print(f"Rounds: {game_context['rounds_played']} | Patches: B:{game_context['patch_track'].get('blue', 0)} R:{game_context['patch_track'].get('red', 0)}")
    
    # Analyze each player
    all_player_analysis = {}
    confused_count = 0
    
    for player_id in sorted(roles.keys()):
        role = roles.get(player_id, "unknown")
        reasoning = player_reasoning.get(player_id, [])
        
        print(f"\nAnalyzing {player_id} ({role})...", end=" ")
        
        analysis = await analyze_player_confusion(
            model,
            player_id,
            role,
            reasoning,
            game_context,
        )
        
        all_player_analysis[player_id] = analysis
        
        if analysis.get("confused"):
            confused_count += 1
            print("CONFUSED")
            print(f"  Confusion types: {', '.join(analysis.get('confusion_types', []))}")
            print(f"  Explanation: {analysis.get('explanation', 'N/A')}")
            if analysis.get("evidence"):
                print(f"  Evidence: {analysis['evidence'][0][:60]}...")
        else:
            print("NOT CONFUSED")
    
    print(f"\n{'-'*70}")
    print(f"GAME-LEVEL IMPROVEMENTS")
    print(f"{'-'*70}")
    
    # Get game-level improvement suggestions
    improvements = await analyze_game_improvements(
        model,
        game_context,
        all_player_analysis,
    )
    
    print(f"Overall Confusion Level: {improvements.get('overall_confusion_level', 'unknown').upper()}")
    if improvements.get("key_insights"):
        print("Key Insights:")
        for insight in improvements.get("key_insights", []):
            print(f"  - {insight}")
    
    print("\nImprovement Suggestions:")
    for suggestion in improvements.get("improvement_suggestions", []):
        print(f"\n  [{suggestion.get('category', 'General')}] {suggestion.get('suggestion', 'N/A')}")
        print(f"    Rationale: {suggestion.get('rationale', 'N/A')}")
        print(f"    Implementation: {suggestion.get('implementation', 'N/A')}")
    
    return {
        "game_num": game_num,
        "game_context": game_context,
        "player_analysis": all_player_analysis,
        "confused_count": confused_count,
        "improvements": improvements,
    }


def generate_confusion_markdown_report(all_results: List[Dict]) -> str:
    """Generate a markdown report from confusion analysis results."""
    md = []
    
    # Header
    md.append("# Confusion Analysis Report")
    md.append("")
    
    # Executive Summary
    total_confused = sum(r["confused_count"] for r in all_results)
    total_players = sum(len(r["game_context"]["roles"]) for r in all_results)
    confusion_rate = (total_confused / total_players * 100) if total_players > 0 else 0
    
    md.append("## Executive Summary")
    md.append("")
    md.append(f"- **Games Analyzed**: {len(all_results)}")
    md.append(f"- **Total Players**: {total_players}")
    md.append(f"- **Confused Players**: {total_confused} ({confusion_rate:.1f}%)")
    md.append("")
    
    if confusion_rate == 0:
        md.append("### Key Finding")
        md.append("No player confusion was detected in the analyzed games. Players demonstrated clear understanding of game rules, state, and strategy.")
    elif confusion_rate > 50:
        md.append("### Key Finding")
        md.append(f"High confusion rates ({confusion_rate:.0f}%) suggest potential issues with game clarity or complexity.")
    else:
        md.append("### Key Finding")
        md.append(f"Moderate confusion rates ({confusion_rate:.0f}%) indicate some players struggled with specific aspects of the game.")
    
    md.append("")
    md.append("---")
    md.append("")
    
    # Per-Game Analysis
    md.append("## Per-Game Analysis")
    md.append("")
    
    for result in all_results:
        game_num = result.get("game_num", "?")
        game_context = result.get("game_context", {})
        player_analysis = result.get("player_analysis", {})
        confused_count = result.get("confused_count", 0)
        improvements = result.get("improvements", {})
        
        total_game_players = len(game_context.get("roles", {}))
        
        md.append(f"### Game {game_num}")
        md.append("")
        md.append(f"- **Result**: {'Blues Won' if game_context.get('blues_win') else 'Reds Won'}")
        md.append(f"- **Rounds**: {game_context.get('rounds_played', 0)}")
        md.append(f"- **Patches**: Blue {game_context.get('patch_track', {}).get('blue', 0)}/6, Red {game_context.get('patch_track', {}).get('red', 0)}/6")
        md.append(f"- **Confused Players**: {confused_count}/{total_game_players}")
        md.append("")
        
        # Confused players in this game
        confused_players = [pid for pid, analysis in player_analysis.items() if analysis.get("confused")]
        if confused_players:
            md.append("#### Confused Players")
            md.append("")
            for player_id in sorted(confused_players):
                analysis = player_analysis[player_id]
                role = game_context.get("roles", {}).get(player_id, "unknown")
                md.append(f"**{player_id}** ({role})")
                md.append(f"- Confusion Types: {', '.join(analysis.get('confusion_types', ['Unknown']))}")
                md.append(f"- Explanation: {analysis.get('explanation', 'N/A')}")
                if analysis.get("evidence"):
                    md.append(f"- Evidence: \"{analysis['evidence'][0]}\"")
                md.append("")
        else:
            md.append("#### Confusion Status")
            md.append("No player confusion detected in this game.")
            md.append("")
        
        # Improvement suggestions
        if improvements.get("improvement_suggestions"):
            md.append("#### Suggested Improvements")
            md.append("")
            md.append(f"Overall Confusion Level: **{improvements.get('overall_confusion_level', 'unknown').upper()}**")
            md.append("")
            for suggestion in improvements.get("improvement_suggestions", []):
                category = suggestion.get("category", "General")
                suggestion_text = suggestion.get("suggestion", "N/A")
                rationale = suggestion.get("rationale", "N/A")
                md.append(f"**[{category}]** {suggestion_text}")
                md.append(f"- Rationale: {rationale}")
                md.append("")
        
        md.append("---")
        md.append("")
    
    # Confusion Type Summary
    md.append("## Confusion Types Summary")
    md.append("")
    
    confusion_types = {}
    for result in all_results:
        for player_analysis in result.get("player_analysis", {}).values():
            if player_analysis.get("confused"):
                for confusion_type in player_analysis.get("confusion_types", []):
                    confusion_types[confusion_type] = confusion_types.get(confusion_type, 0) + 1
    
    if confusion_types:
        md.append("Most common sources of player confusion:")
        md.append("")
        for confusion_type, count in sorted(confusion_types.items(), key=lambda x: x[1], reverse=True):
            md.append(f"- **{confusion_type}**: {count} instances")
        md.append("")
    else:
        md.append("No confusion detected across all games.")
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
    
    total_confused = sum(r["confused_count"] for r in all_results)
    total_players = sum(len(r["game_context"]["roles"]) for r in all_results)
    
    print(f"\nTotal Games Analyzed: {len(all_results)}")
    print(f"Total Players: {total_players}")
    print(f"Total Players Who Showed Confusion: {total_confused} ({100*total_confused/total_players:.1f}%)")
    
    # Save detailed results
    output_file = eval_path.parent / f"confusion_analysis_{eval_path.stem}.json"
    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Generate and save markdown report
    markdown_report = generate_confusion_markdown_report(all_results)
    markdown_file = eval_path.parent / f"confusion_analysis_{eval_path.stem}.md"
    with open(markdown_file, "w", encoding="utf-8") as f:
        f.write(markdown_report)
    
    print(f"Markdown report saved to: {markdown_file}")


if __name__ == "__main__":
    import asyncio
    
    if len(sys.argv) < 2:
        print("Usage: python confusion_analysis.py <eval_file> [model_name]")
        print("Example: python confusion_analysis.py results/game.eval")
        print("Example: python confusion_analysis.py results/game.eval anthropic/claude-opus")
        print("\nDefault model: ollama/gpt-oss:20b")
        sys.exit(1)
    
    eval_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "ollama/gpt-oss:20b"
    
    asyncio.run(main(eval_file, model_name))
