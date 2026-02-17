# Cross-Analysis Findings Tool

## Overview

The `cross_analysis_findings.py` script produces game-level synthesis across strategy, confusion, efficiency, risk, collaboration, role utilization, and outcome impact.

## Output

- `cross_analysis_findings_<eval_id>.json`
- `cross_analysis_findings_<eval_id>.md`

## Usage

```bash
python -m red_vs_blue.analysis.cross_analysis_findings results/game.eval [model_name]
```

## Key Fields

- `overall_game_quality`
- `cross_analysis_findings`
- `high_impact_players`
- `systemic_patterns`
- `recommended_interventions`
