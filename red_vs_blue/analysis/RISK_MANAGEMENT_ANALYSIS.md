# Risk Management Analysis Tool

## Overview

The `risk_management_analysis.py` script analyzes how players identify threats, manage uncertainty, and mitigate avoidable risks.

## Output

- `risk_management_analysis_<eval_id>.json`
- `risk_management_analysis_<eval_id>.md`

## Usage

```bash
python -m red_vs_blue.analysis.risk_management_analysis results/game.eval [model_name]
```

## Key Fields

- `demonstrates_risk_management`
- `risk_awareness_score` (0.0-1.0)
- `risk_taking_level`
- `mitigations_used`
- `avoidable_risks_taken`
