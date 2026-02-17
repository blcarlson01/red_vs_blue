# Outcome Attribution Analysis Tool

## Overview

The `outcome_attribution_analysis.py` script attributes game outcomes to player-level contributions and mistakes.

## Output

- `outcome_attribution_analysis_<eval_id>.json`
- `outcome_attribution_analysis_<eval_id>.md`

## Usage

```bash
python -m red_vs_blue.analysis.outcome_attribution_analysis results/game.eval [model_name]
```

## Key Fields

- `impact_score` (-1.0 to 1.0)
- `impact_direction`
- `impact_magnitude`
- `key_contributions`
- `key_mistakes`
