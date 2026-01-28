# Confusion Analysis Tool

## Overview

The `confusion_analysis.py` script analyzes player behavior and reasoning from Red vs. Blue game transcripts to detect player confusion. It uses an LLM to identify when players misunderstood game rules, game state, or their own strategic goals.

## Features

### 1. Player Statement Extraction
- Extracts all statements players made during the game
- Rebuilds narrative context from public game log
- Tracks each player's reasoning throughout the game

### 2. Per-Player Confusion Detection
For each player, the script:
- Analyzes their statements against game context
- Detects 5 types of confusion:
  - **Logical Inconsistency**: Player contradicts themselves
  - **Rule Misunderstanding**: Player confused about game mechanics
  - **State Confusion**: Player confused about game state (rounds, patches, fired players)
  - **Strategic Confusion**: Moves don't align with stated goals or role
  - **Role Confusion**: Acting inconsistent with their role (blue/red/APT Leader)

### 3. Evidence-Based Analysis
- Provides specific quotes/evidence for detected confusion
- Explains what confusion was detected and why
- Includes suggestions for helping that specific player

### 4. Game-Level Improvements
- Aggregates confusion patterns across all players
- Suggests improvements in categories:
  - **Rules**: Clarify rule language or mechanics
  - **Information**: Make game state more visible
  - **Strategy**: Add decision-making guides
  - **Tutorial**: Improve onboarding/setup

## Usage

### Basic Usage

```bash
# Set UTF-8 encoding (required on Windows)
$env:PYTHONIOENCODING = "utf-8"

# Analyze a single game for player confusion
python -m red_vs_blue.analysis.confusion_analysis <eval_file> [model_name]

# Examples
python -m red_vs_blue.analysis.confusion_analysis results/game.eval anthropic/claude-opus
python -m red_vs_blue.analysis.confusion_analysis results/game.eval anthropic/claude-haiku
```

### Output

The script produces:

1. **Console Output**: 
   - Per-player confusion analysis
   - Game-level improvement suggestions
   - Overall summary statistics

2. **JSON File**: `results/confusion_analysis_<game_id>.json`
   - Complete analysis data
   - All evidence and explanations
   - Improvement suggestions
   - Can be parsed for further analysis

### Example Output

```
======================================================================
GAME 1: CONFUSION ANALYSIS
======================================================================
Result: Blues Won!
Rounds: 5 | Patches: B:6 R:2

Analyzing p0 (blue)... NOT CONFUSED
Analyzing p1 (red)... CONFUSED
  Confusion types: Rule Misunderstanding, Strategic Confusion
  Explanation: Player seemed unsure about when patches are applied
  Evidence: "I thought we needed 5 blue patches to win..." 

Analyzing p2 (blue)... NOT CONFUSED
Analyzing p3 (apt_leader)... NOT CONFUSED
Analyzing p4 (blue)... NOT CONFUSED

======================================================================
GAME-LEVEL IMPROVEMENTS
======================================================================
Overall Confusion Level: LOW

Key Insights:
- One player was confused about win conditions
- Most players understood their roles correctly

Improvement Suggestions:

  [Rules] Make win conditions more explicit
    Rationale: Player p1 was uncertain about the 6-patch threshold
    Implementation: Add visual indicators showing X/6 progress for each team

  [Information] Display player role changes
    Rationale: Would help APT Leader track who knows what
    Implementation: Show role discoveries in a separate log section
```

## Understanding the Analysis

### Confusion Types Explained

**Logical Inconsistency**
- Player says "I think p0 is blue" then later "p0 must be red"
- No explanation for change
- Suggests player is confused about current state

**Rule Misunderstanding**
- "When do patches actually get applied?"
- "Do I need to be SOC Lead to use my power?"
- "Can fired players still vote?"

**State Confusion**
- "How many patches do we have now?"
- "Who is still employed?"
- "What round are we on?"

**Strategic Confusion**
- Blue player helping reds reach 6 patches
- Red player voting to fire other reds
- Actions contradicting stated goals

**Role Confusion**
- APT Leader acting too obviously red/blue
- Reds not coordinating when they should
- Blues accusing random players without reasoning

### Improvement Categories

**Rules**
- Make rule language clearer
- Add visual decision aids
- Simplify complex mechanics

**Information**
- Show more game state visually
- Make everyone's role history visible
- Track belief changes over time

**Strategy**
- Add tips for good decision-making
- Show information value of different actions
- Suggest strategic moves

**Tutorial**
- Better initial explanation
- Interactive walkthrough
- Example games with narration

## Implementation Details

### LLM Prompts

The script uses carefully designed prompts:

1. **Per-Player Analysis Prompt**
   - Provides full game context
   - Shows player's statements (max 20)
   - Asks LLM to detect confusion types
   - Requests evidence and suggestions

2. **Game Improvement Prompt**
   - Summarizes confusion patterns
   - Asks for actionable improvements
   - Requires justification and implementation plan

### Temperature Setting

- **Temperature: 0.3** - Makes analysis more consistent and conservative
- Avoids over-detecting confusion from minor inconsistencies
- Focuses on significant comprehension gaps

## JSON Output Format

```json
{
  "game_num": 1,
  "game_context": {
    "num_players": 5,
    "rounds_played": 5,
    "blues_win": true,
    "patch_track": {"blue": 6, "red": 2},
    "roles": {"p0": "blue", ...},
    "fired_players": ["p1"]
  },
  "player_analysis": {
    "p0": {
      "confused": false,
      "confusion_types": [],
      "explanation": "Player was not confused",
      "evidence": [],
      "improvement_suggestions": []
    },
    "p1": {
      "confused": true,
      "confusion_types": ["Rule Misunderstanding"],
      "explanation": "Player seemed unsure about patch application timing",
      "evidence": ["I thought we needed 5 patches..."],
      "improvement_suggestions": ["Make win condition more explicit"]
    }
  },
  "confused_count": 1,
  "improvements": {
    "overall_confusion_level": "low",
    "key_insights": ["One player was confused..."],
    "improvement_suggestions": [
      {
        "category": "Rules",
        "suggestion": "...",
        "rationale": "...",
        "implementation": "..."
      }
    ]
  }
}
```

## Tips for Best Results

1. **Use Strong Models**: Claude-opus gives better analysis than smaller models
2. **Multiple Games**: Run on several games to find patterns
3. **Review JSON Output**: Machine analysis + human review is ideal
4. **Check Evidence**: Look at quoted statements to verify LLM analysis
5. **Focus on Changes**: Suggestions about game improvements are most actionable

## Limitations

1. **LLM Limitations**
   - Confusion detection depends on LLM quality
   - May miss subtle confusion
   - May over-detect from normal variation

2. **Data Limitations**
   - Only analyzes public statements
   - Can't see internal reasoning directly
   - Limited context from sparse transcripts

3. **Scope**
   - Focuses on confusion, not skill or strategy quality
   - Doesn't analyze game balance
   - Doesn't measure fun/engagement

## Future Enhancements

Potential improvements to the tool:

1. **Comparative Analysis**
   - Compare confusion rates across models
   - Track confusion reduction over time
   - Identify which game variants reduce confusion

2. **Detailed Metrics**
   - Quantify confusion severity
   - Track confusion trends
   - Identify high-confusion roles

3. **Visualization**
   - Confusion timeline per player
   - Heatmap of confusion types
   - Network graphs of confused players

4. **Interactivity**
   - Web interface to explore confusion
   - Filter by player, round, type
   - Side-by-side comparisons

## Troubleshooting

### Model Not Found Error
```
ValueError: Model name 'claude-opus' should be in the format of <api_name>/<model_name>
```
**Solution**: Use full model name `anthropic/claude-opus`

### Anthropic Module Not Installed
```
ModuleNotFoundError: No module named 'anthropic'
```
**Solution**: Install with `pip install anthropic`

### Empty Game Log
If no statements are extracted, the game transcript might not have public log data. This occurs in very old eval files.

### Slow Analysis
Analysis takes ~30-60 seconds per game with claude-opus. Use claude-haiku for faster results (lower quality).
