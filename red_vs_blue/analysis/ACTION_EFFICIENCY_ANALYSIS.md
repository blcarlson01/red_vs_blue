# Action Efficiency Analysis Tool

## Overview

The `action_efficiency_analysis.py` script evaluates how efficiently each player converts discussion and voting into role-aligned progress.

## Output

- `action_efficiency_analysis_<eval_id>.json`
- `action_efficiency_analysis_<eval_id>.md`

## Usage

```bash
python -m red_vs_blue.analysis.action_efficiency_analysis results/game.eval [model_name]
```

## Key Fields

- `efficient_action_taker`
- `efficiency_score` (0.0-1.0)
- `signal_to_noise`
- `decision_quality`
- `effective_actions`
- `wasted_actions`
