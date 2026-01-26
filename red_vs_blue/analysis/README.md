# Red vs. Blue Analysis Tools

This directory contains tools for analyzing and reviewing Red vs. Blue benchmark evaluation results.

## Contents

- **results_viewer.py** - Main analysis and visualization tool for Inspect eval results
- **aggregate_results.py** - Aggregates metrics across multiple games
- **advanced_analysis.py** - Advanced statistical analysis (per-role, voting, beliefs, predictors)
- **advanced_infographics.py** - Publication-ready infographics and visualizations
- **plots.py** - Visualization utilities for benchmark metrics
- **statistics.py** - Statistical significance testing

## Quick Start

### View Results as Interactive HTML

After running an evaluation:

```bash
python -m red_vs_blue.analysis.results_viewer results/*.eval results/review.html
```

This generates an interactive HTML file that includes:
- Summary statistics (win rates, average metrics)
- Game viewer with transcript display
- Copy-to-clipboard functionality
- Works offline in any browser

### Aggregating and Statistical Analysis

After running one or more eval tasks:

```bash
# Step 1: Aggregate all .eval files into JSONL
python -m red_vs_blue.analysis.aggregate_results results/

# Step 2: Run advanced analysis
python -m red_vs_blue.analysis.advanced_analysis results/

# Step 3 (optional): Statistical significance testing
python -m red_vs_blue.analysis.statistics results/
```

### Advanced Analysis Features

The `advanced_analysis.py` script generates comprehensive insights:

**Per-Role Performance**
- Win rates by role (blue, red, apt_leader)
- Survival rates per role
- Average game length by role
- Belief accuracy by role

**Voting Pattern Analysis**
- Total votes recorded
- Red coalition strength (do they vote together?)
- Vote consistency metrics
- Swing voter identification

**Belief Dynamics**
- Information gathering rate (entropy reduction)
- Convergence patterns
- High vs low information games
- Final belief alignment

**Patch Track Momentum**
- Average patches applied (blue and red)
- Games with patch sweeps
- Timing of patch enactment
- Momentum shifts

**Deception Effectiveness**
- APT Leader deception scores
- Correlation with red win rate
- High deception vs low deception win rates
- Deception strategy analysis

**Early-Game Predictors**
- Which early-game metrics predict outcomes
- Game length distribution
- Information gathering patterns

**Model-Comparative Analysis**
- Win rates by model (when running multiple models)
- Model performance across all metrics
- Model differences in strategy

### Command Line Usage

```bash
# View summary in terminal
python advanced_analysis.py results/

# Results saved to:
# - results/aggregated/advanced_analysis.json (detailed report)
# - results/aggregated/summary.csv (aggregated metrics)
# - results/aggregated/all_results.jsonl (per-game data)
```

### Generating Publication-Ready Infographics

After running advanced analysis, generate camera-ready publication figures:

```bash
python -m red_vs_blue.analysis.advanced_infographics results/
```

This generates the following infographics (saved as PDF and PNG in `results/figures/`):

1. **Game Length Distribution** - Histogram showing frequency of game durations with mean and median
2. **Belief Dynamics Progression** - Line plot of entropy reduction across games showing information gathering rate
3. **Deception Effectiveness** - Scatter plot of APT Leader deception scores vs game outcomes with trend line
4. **Entropy Box Plot** - Box plot comparing entropy reduction across low/medium/high information gathering levels
5. **Model Performance Heatmap** - Normalized heatmap of model performance across all key metrics
6. **Game Outcome Summary** - Pie chart of blue vs red win rates
7. **Early-Game Correlations** - Correlation heatmap of early-game predictors (entropy, rounds, belief alignment)
8. **Metrics Summary Panel** - 4-panel summary dashboard with games statistics, win rates, epistemic metrics, and computational cost

All infographics are publication-ready with:
- High-resolution output (300 DPI PNG + PDF)
- NeurIPS/ICML-compatible styling
- Professional color palette
- Detailed annotations and legends

## Understanding the Output

### Game Transcript Format

Each game transcript includes:

1. **SETUP** - Role assignments and true identity of APT Leader
2. **GAME LOG** - All public messages from players during discussion and voting
3. **VOTING HISTORY** - Accusations and voting results
4. **OUTCOME** - Winner, final patch track, and fired players
5. **METRICS** - Performance metrics:
   - Avg Entropy Reduction: How much agents' beliefs changed
   - Avg Belief Alignment: How well beliefs predicted true roles
   - Avg Brier Score: Calibration of final belief predictions
   - APT Leader Deception: How well APT Leader hid their identity
6. **FINAL BELIEFS** - Each player's final belief distribution

### Example Transcript Entry

```
SETUP:
  Players: 5
  Roles: p0=blue, p1=red, p2=blue, p3=blue, p4=apt_leader
  True APT Leader: p4

GAME LOG:
  p0: I think p4 is suspicious
  p1: p4 looks innocent to me
  ...

VOTING HISTORY:
  Accused: p4 | Votes: 3 yes, 2 no | PASSED

OUTCOME:
  Winner: BLUES
  Rounds: 8
  Final Patches: Blue=6, Red=2
  Fired: ['p4']
```

## Key Metrics Explained

### Entropy Reduction
- Measures how much agents' beliefs evolved
- Higher = agents updated beliefs more
- Indicates learning about other players' roles

### Belief Alignment
- How well final beliefs predicted actual roles
- 1.0 = perfect prediction
- 0.0 = complete misalignment
- Used to measure reasoning quality

### APT Leader Deception
- Score showing how well APT Leader (red leader) hid identity
- Higher = better deception (harder to detect as red)
- Only computed for APT Leader player

### Brier Score
- Calibration metric for probability predictions
- 0.0 = perfectly calibrated
- 1.0 = completely wrong
- Measures confidence vs accuracy

## Data Flow

1. **Inspect Evaluation** runs games and collects results in `.eval` ZIP files
2. **Results Viewer** extracts game metadata from ZIP
3. **HTML Export** generates shareable interactive report
4. **Other Tools** aggregate and visualize metrics

## Troubleshooting

### Empty Game Logs
- Ensure evaluation was run with updated task.py that captures public_log
- Old eval files (before fixes) may have empty transcripts
- Regenerate results with: `inspect eval red_vs_blue --limit 5`

### Missing Voting History
- Indicates no accusations were made during games
- Check that agents are providing accusations in their responses
- Review agent prompts in agents.py to ensure voting instructions are clear

### Missing Messages
- Check that agent responses include actual message content
- Avoid empty first lines in responses
- Review _parse_response in agents.py if parsing seems incorrect

## File Format Details

The `.eval` file is a ZIP archive containing:
- `samples/` directory with JSON files
- Each JSON has structure: `{id, epoch, scores, metadata, events, ...}`
- Game data stored in `scores['red_vs_blue_scorer']['metadata']`

## Extending Analysis

To add custom analysis:

1. Subclass `ResultsExtractor` or call its methods
2. Access `self.results` (list of all game JSON objects)
3. Extract metadata as needed for analysis
4. Generate custom reports or visualizations

Example:
```python
from results_viewer import ResultsExtractor

extractor = ResultsExtractor(Path("eval.eval"))
for i, result in enumerate(extractor.results):
    metadata = result['scores']['red_vs_blue_scorer']['metadata']
    # Custom analysis here
```
