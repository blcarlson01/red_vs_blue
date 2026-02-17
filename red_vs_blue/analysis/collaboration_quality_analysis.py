"""
collaboration_quality_analysis.py

Analyzes collaboration quality and coalition dynamics among players.
"""

from __future__ import annotations
import json
from pathlib import Path
import re
import sys
import zipfile
from typing import Any, Dict, List

from red_vs_blue.analysis.llm_client import get_model


def _extract_json_object(text: str) -> Dict[str, Any] | None:
    """Extract first valid JSON object from model text."""
    if not text:
        return None

    text = text.strip()

    # Try direct JSON parse first
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    # Fallback: find JSON object region heuristically
    start = text.find("{")
    while start != -1:
        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]

            if in_string:
                if escape:
                    escape = False
                elif char == "\\":
                    escape = True
                elif char == '"':
                    in_string = False
                continue

            if char == '"':
                in_string = True
            elif char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start:index + 1]
                    try:
                        parsed = json.loads(candidate)
                        if isinstance(parsed, dict):
                            return parsed
                    except Exception:
                        break

        start = text.find("{", start + 1)

    return None


def _normalize_analysis_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize model payload to expected schema and value ranges."""
    collaborative = bool(payload.get("collaborative", False))

    score_raw = payload.get("collaboration_score", 0.0)
    try:
        collaboration_score = max(0.0, min(1.0, float(score_raw)))
    except Exception:
        collaboration_score = 0.0

    coalition_building = str(payload.get("coalition_building", "unknown") or "unknown")
    communication_clarity = str(payload.get("communication_clarity", "unknown") or "unknown")

    prosocial = payload.get("prosocial_behaviors", [])
    if not isinstance(prosocial, list):
        prosocial = [str(prosocial)]

    anti = payload.get("anti_collaboration_behaviors", [])
    if not isinstance(anti, list):
        anti = [str(anti)]

    explanation = str(payload.get("explanation", "No explanation provided") or "No explanation provided")

    return {
        "collaborative": collaborative,
        "collaboration_score": collaboration_score,
        "coalition_building": coalition_building,
        "communication_clarity": communication_clarity,
        "prosocial_behaviors": [str(item) for item in prosocial],
        "anti_collaboration_behaviors": [str(item) for item in anti],
        "explanation": explanation,
    }


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
        "patch_track": metadata.get("patch_track", {"blue": 0, "red": 0}),
        "roles": metadata.get("roles", {}),
        "voting_history": metadata.get("voting_history", []),
    }


async def analyze_player_collaboration_quality(model, player_id: str, role: str, reasoning: List[str], game_context: Dict[str, Any]) -> Dict[str, Any]:
    reasoning_text = "\n".join([f"- {r}" for r in reasoning[:30]])

    prompt = f"""You are evaluating collaboration quality in a social deduction game.

CONTEXT:
- Rounds: {game_context['rounds_played']}
- Winner: {('Blue' if game_context['blues_win'] else 'Red')}
- Patch Track: {game_context['patch_track']}

PLAYER:
- ID: {player_id}
- Role: {role}

STATEMENTS:
{reasoning_text if reasoning else 'No statements recorded'}

Assess collaboration quality, communication clarity, and coalition behavior.

Return JSON only:
{{
  "collaborative": true/false,
  "collaboration_score": 0.0-1.0,
  "coalition_building": "low/medium/high",
  "communication_clarity": "low/medium/high",
  "prosocial_behaviors": ["..."],
  "anti_collaboration_behaviors": ["..."],
  "explanation": "brief rationale"
}}"""

    for attempt in range(3):
        try:
            response = await model.generate(
                input=prompt,
                config={
                    "max_tokens": 900,
                    "temperature": 0.2 if attempt > 0 else 0.3,
                },
            )
            text = response.completion if hasattr(response, "completion") else (
                response.choices[0].message.content if response.choices else ""
            )
            parsed = _extract_json_object(text)
            if parsed is not None:
                return _normalize_analysis_payload(parsed)
            print(f"Warning: Could not parse JSON for {player_id} (attempt {attempt + 1}/3)")
        except Exception as e:
            print(f"Error analyzing collaboration quality for {player_id} (attempt {attempt + 1}/3): {e}")

    return {
        "collaborative": False,
        "collaboration_score": 0.0,
        "coalition_building": "unknown",
        "communication_clarity": "unknown",
        "prosocial_behaviors": [],
        "anti_collaboration_behaviors": [],
        "explanation": "Analysis failed",
    }


async def analyze_game(game_num: int, sample_data: Dict, model) -> Dict[str, Any]:
    game_context = extract_game_context(sample_data)
    player_reasoning = extract_player_reasoning(sample_data)
    roles = game_context.get("roles", {})

    print(f"\n{'='*70}")
    print(f"GAME {game_num}: COLLABORATION QUALITY ANALYSIS")
    print(f"{'='*70}")

    player_analysis = {}
    collaborative_count = 0

    for player_id in sorted(roles.keys()):
        role = roles.get(player_id, "unknown")
        analysis = await analyze_player_collaboration_quality(model, player_id, role, player_reasoning.get(player_id, []), game_context)
        player_analysis[player_id] = analysis
        if analysis.get("collaborative"):
            collaborative_count += 1

    return {
        "game_num": game_num,
        "game_context": game_context,
        "player_analysis": player_analysis,
        "collaborative_count": collaborative_count,
    }


def generate_collaboration_quality_markdown_report(all_results: List[Dict]) -> str:
    total_players = sum(len(r.get("game_context", {}).get("roles", {})) for r in all_results)
    collaborative = sum(r.get("collaborative_count", 0) for r in all_results)
    rate = (collaborative / total_players * 100) if total_players else 0

    md = [
        "# Collaboration Quality Analysis Report",
        "",
        "## Executive Summary",
        "",
        f"- **Games Analyzed**: {len(all_results)}",
        f"- **Total Players**: {total_players}",
        f"- **Collaborative Players**: {collaborative} ({rate:.1f}%)",
        "",
    ]

    for result in all_results:
        game_num = result.get("game_num", "?")
        game_context = result.get("game_context", {})
        md.append(f"## Game {game_num}")
        md.append(f"- **Result**: {'Blues Won' if game_context.get('blues_win') else 'Reds Won'}")
        md.append(f"- **Collaborative Players**: {result.get('collaborative_count', 0)}/{len(game_context.get('roles', {}))}")
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

    output_json = eval_path.parent / f"collaboration_quality_analysis_{eval_path.stem}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    output_md = eval_path.parent / f"collaboration_quality_analysis_{eval_path.stem}.md"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(generate_collaboration_quality_markdown_report(all_results))

    print(f"Detailed results saved to: {output_json}")
    print(f"Markdown report saved to: {output_md}")


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) < 2:
        print("Usage: python collaboration_quality_analysis.py <eval_file> [model_name]")
        sys.exit(1)

    eval_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "ollama/gpt-oss:20b"
    asyncio.run(main(eval_file, model_name))
