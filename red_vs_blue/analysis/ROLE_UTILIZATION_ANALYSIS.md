# Role Utilization Analysis Tool

## Overview

The `role_utilization_analysis.py` script measures whether each player used their assigned role effectively and aligned actions with role objectives.

## Output

- `role_utilization_analysis_<eval_id>.json`
- `role_utilization_analysis_<eval_id>.md`

## Usage

```bash
python -m red_vs_blue.analysis.role_utilization_analysis results/game.eval [model_name]
```

## Key Fields

- `role_utilized`
- `utilization_score` (0.0-1.0)
- `role_alignment`
- `role_objectives_supported`
- `role_objectives_missed`
