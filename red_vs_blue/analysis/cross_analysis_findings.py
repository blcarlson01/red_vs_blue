"""
cross_analysis_findings.py

Generates cross-cutting findings that synthesize strategy, confusion, efficiency,
risk management, collaboration, role utilization, and outcome impact.
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


def extract_game_context(sample_data: Dict) -> Dict[str, Any]:
    metadata = sample_data.get("scores", {}).get("red_vs_blue_scorer", {}).get("metadata", {})
    return {
        "rounds_played": metadata.get("rounds_played", 0),
        "blues_win": metadata.get("blues_win", False),
        "roles": metadata.get("roles", {}),
        "patch_track": metadata.get("patch_track", {"blue": 0, "red": 0}),
        "fired_players": metadata.get("fired_players", []),
        "public_log": metadata.get("public_log", []),
        "voting_history": metadata.get("voting_history", []),
    }


async def analyze_cross_findings(model, game_num: int, game_context: Dict[str, Any]) -> Dict[str, Any]:
    prompt = f"""You are producing cross-analysis findings for a social deduction game.

GAME {game_num} CONTEXT:
- Winner: {('Blue' if game_context['blues_win'] else 'Red')}
- Rounds: {game_context['rounds_played']}
- Patch Track: {game_context['patch_track']}
- Fired Players: {game_context['fired_players']}
- Roles: {game_context['roles']}
- Voting History: {game_context['voting_history']}
- Public Log (truncated): {game_context['public_log'][:80]}

Provide synthesis across these dimensions:
1. Action efficiency
2. Risk management
3. Collaboration quality
4. Role utilization
5. Outcome attribution
6. Strategic coherence and confusion interplay

Return JSON only:
{{
  "overall_game_quality": "low/medium/high",
  "cross_analysis_findings": ["..."],
  "high_impact_players": ["p0", "p1"],
  "systemic_patterns": ["..."],
  "recommended_interventions": [
    {{"area": "rules/agent_prompt/scoring/ui", "recommendation": "...", "rationale": "..."}}
  ]
}}"""

    parsed = await generate_json_with_retries(
        model,
        prompt,
        max_tokens=1300,
        temperature=0.3,
        retries=6,
        error_context=f"cross-analysis findings for game {game_num}",
    )
    if parsed is not None:
        return parsed

    return {
        "overall_game_quality": "unknown",
        "cross_analysis_findings": [],
        "high_impact_players": [],
        "systemic_patterns": [],
        "recommended_interventions": [],
    }


def generate_cross_findings_markdown_report(all_results: List[Dict]) -> str:
    md = [
        "# Cross-Analysis Findings Report",
        "",
        "## Executive Summary",
        "",
        f"- **Games Analyzed**: {len(all_results)}",
        "",
    ]

    for result in all_results:
        game_num = result.get("game_num", "?")
        findings = result.get("cross_findings", {})
        context = result.get("game_context", {})

        md.append(f"## Game {game_num}")
        md.append(f"- **Result**: {'Blues Won' if context.get('blues_win') else 'Reds Won'}")
        md.append(f"- **Overall Game Quality**: {findings.get('overall_game_quality', 'unknown')}" )
        md.append("")

        if findings.get("cross_analysis_findings"):
            md.append("### Key Findings")
            for finding in findings["cross_analysis_findings"]:
                md.append(f"- {finding}")
            md.append("")

        if findings.get("recommended_interventions"):
            md.append("### Recommended Interventions")
            for intervention in findings["recommended_interventions"]:
                md.append(f"- **{intervention.get('area', 'general')}**: {intervention.get('recommendation', 'N/A')}")
                md.append(f"  - Rationale: {intervention.get('rationale', 'N/A')}")
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

    all_results = []
    for i, sample in enumerate(samples, 1):
        game_context = extract_game_context(sample)
        cross_findings = await analyze_cross_findings(model, i, game_context)
        all_results.append({
            "game_num": i,
            "game_context": game_context,
            "cross_findings": cross_findings,
        })

    output_json = eval_path.parent / f"cross_analysis_findings_{eval_path.stem}.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2, default=str)

    output_md = eval_path.parent / f"cross_analysis_findings_{eval_path.stem}.md"
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(generate_cross_findings_markdown_report(all_results))

    print(f"Detailed results saved to: {output_json}")
    print(f"Markdown report saved to: {output_md}")


if __name__ == "__main__":
    import asyncio

    if len(sys.argv) < 2:
        print("Usage: python cross_analysis_findings.py <eval_file> [model_name]")
        sys.exit(1)

    eval_file = sys.argv[1]
    model_name = sys.argv[2] if len(sys.argv) > 2 else "ollama/gpt-oss:20b"
    asyncio.run(main(eval_file, model_name))
