# Collaboration Quality Analysis Tool

## Overview

The `collaboration_quality_analysis.py` script evaluates communication clarity, coalition building, and cooperative behavior quality.

## Output

- `collaboration_quality_analysis_<eval_id>.json`
- `collaboration_quality_analysis_<eval_id>.md`

## Usage

```bash
python -m red_vs_blue.analysis.collaboration_quality_analysis results/game.eval [model_name]
```

## Key Fields

- `collaborative`
- `collaboration_score` (0.0-1.0)
- `coalition_building`
- `communication_clarity`
- `prosocial_behaviors`
- `anti_collaboration_behaviors`
