# Strategy Analysis Tool

## Overview

The `strategy_analysis.py` script analyzes player strategies from game transcripts using an LLM. It examines each player's statements and voting behavior to determine if they followed a coherent strategy throughout the game.

## Features

### Individual Player Strategy Analysis
For each player, the tool:
1. **Extracts all player statements and reasoning** from the public game log
2. **Analyzes voting behavior** across all rounds
3. **Uses LLM to identify strategies** by examining:
   - **Strategic Consistency**: Did the player maintain consistent goals?
   - **Information Gathering**: Did they systematically try to identify the APT Leader?
   - **Deceptive Play**: If red/APT, did they employ consistent deceptive tactics?
   - **Coalition Building**: Did they try to build alliances or gather support?
   - **Role-Aligned Behavior**: Did actions align with their role?

### Strategy Detection Output
For each player, the tool provides:
- **has_strategy** (true/false): Whether a coherent strategy was detected
- **strategy_name**: Name of the identified strategy (e.g., "Information Aggregator", "Consistent Deceiver")
- **strategy_description**: Brief description of the detected strategy
- **consistency_score** (0.0-1.0): How consistently the strategy was followed
- **strategy_effectiveness**: How well the strategy worked (low/medium/high)
- **key_behaviors**: List of specific actions that defined the strategy
- **contradictions**: Instances where the player deviated from their strategy
- **role_alignment**: How well the strategy aligned with the player's role

## Usage

### Command Line

```bash
# Basic usage - analyzes eval file with default model
python -m red_vs_blue.analysis.strategy_analysis results/your_eval_file.eval

# With custom model
python -m red_vs_blue.analysis.strategy_analysis results/your_eval_file.eval anthropic/claude-opus

# Set model base URL
export INSPECT_EVAL_MODEL_BASE_URL=http://192.168.86.230:11434/v1
python -m red_vs_blue.analysis.strategy_analysis results/your_eval_file.eval ollama/gpt-oss:20b
```

### In Unified Analysis Pipeline

```bash
# Run strategy analysis as part of complete pipeline
python -m red_vs_blue.analysis.run_all_analysis results/

# Skip strategy analysis
python -m red_vs_blue.analysis.run_all_analysis results/ --skip-strategy

# Run only strategy analysis
python -m red_vs_blue.analysis.run_all_analysis results/ --only strategy
```

## Output

The tool generates:
- **strategy_analysis_[eval_id].json**: Detailed JSON with full strategy analysis for each player
- **Console output**: Human-readable summary with:
  - Per-game strategy analysis for each player
  - Overall summary of strategic players
  - Most common strategies detected

## Example Output

```
======================================================================
GAME 1: STRATEGY ANALYSIS
======================================================================
Result: Reds Won!
Rounds: 8 | Patches: B:1 R:4

Analyzing p0 (apt_leader)... STRATEGIC
  Strategy: Consistent Deceiver (consistency: 0.9)
  Description: APT Leader maintained consistent deceptive messaging while voting strategically
  Contradictions: 1 found

Analyzing p1 (blue)... STRATEGIC
  Strategy: Information Aggregator (consistency: 0.8)
  Description: Blue player systematically gathered information through voting patterns
  
Analyzing p2 (blue)... NO CLEAR STRATEGY
```

## Available Strategies

Some common strategies the tool detects:

### Red Team Strategies
- **Consistent Deceiver**: Maintains consistent false narratives
- **Silent Conspirator**: Coordinates with other reds through voting patterns
- **Cautious APT**: Minimizes deception to avoid detection
- **Aggressive Deception**: Makes bold false claims

### Blue Team Strategies
- **Information Aggregator**: Systematically gathers voting pattern insights
- **Alignment Seeker**: Tries to find and coordinate with other blues
- **Scrutiny Raiser**: Questions other players to reveal inconsistencies
- **Fence Sitter**: Cautiously votes without committing to positions

## Model Requirements

- Uses the same LLM model as the benchmark evaluation
- Default: `ollama/gpt-oss:20b`
- Requires model to be accessible at configured endpoint
- Temperature set to 0.3 for consistent analysis

## Integration with Other Analysis Tools

Works alongside other analysis tools:
- **confusion_analysis.py**: Identifies player confusion vs. strategic play
- **advanced_analysis.py**: Quantitative performance metrics
- **run_all_analysis.py**: Unified pipeline that runs all tools

## Performance Considerations

- Each player analysis requires one LLM API call
- Analyzes first 30 player statements per game (can be modified)
- Expected runtime: ~30 seconds per game (with typical models)

## Future Enhancements

Potential improvements:
- Detect strategy evolution (does strategy change mid-game?)
- Compare strategy effectiveness by role
- Identify counter-strategies and their success rates
- Visualize strategy patterns across multiple games
