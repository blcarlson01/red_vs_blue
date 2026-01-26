# Quick Reference: Refactored Task Usage

## Creating a Task

### Single Game
```python
from red_vs_blue.task import red_vs_blue_task

task = red_vs_blue_task(
    num_players=5,
    max_rounds=10,
    seed=42
)
```

### Multiple Games
```python
task = red_vs_blue_task(
    num_games=10,  # Run 10 games
    num_players=5,
    max_rounds=10,
    seed=42  # Seeds: 42, 43, 44, ..., 51
)
```

## Task Structure

```python
task.dataset      # List[Sample] with game parameters in metadata
task.solver       # Function that executes games
task.scorer       # Function that evaluates game outcomes

# Access samples
for sample in task.dataset:
    print(sample.input)           # "Red vs. Blue Game 1"
    print(sample.metadata)        # {'game_id': 0, 'num_players': 5, ...}
```

## Metrics Available

```python
# All games tracked with these metrics:
1. blue_win_rate      # % of games blues won
2. red_win_rate      # % of games reds won
3. avg_rounds_played     # Average game length
```

## Component Architecture

### Solver: Executes Games
```python
@solver
def game_solver() -> Solver:
    async def solve(state: TaskState, generate: Generate) -> TaskState:
        # 1. Extract parameters from sample metadata
        metadata = state.current_sample.metadata
        num_players = metadata.get("num_players", 5)
        seed = metadata.get("seed")
        
        # 2. Create environment and agents
        env = RedvsBlueEnv(num_players=num_players, seed=seed)
        agents = create_agents(...)
        
        # 3. Run game
        results = await _run_game_loop(env, agents)
        
        # 4. Store in state for scorer
        state.output = {
            "env": env,
            "game_results": results,
        }
        return state
```

### Scorer: Evaluates Results
```python
@scorer(
    metrics=[blue_win_metric, red_win_metric, avg_rounds_played_metric],
    name="red_vs_blue_scorer"
)
def result_scorer() -> ScorerType:
    async def score(state: TaskState, target: str) -> Score:
        # 1. Access solver results
        game_results = state.output.get("game_results", {})
        
        # 2. Determine winner
        blues_win = game_results.get("blues_win", False)
        
        # 3. Return Score with metadata
        return Score(
            value=int(blues_win),
            answer=str(blues_win),
            explanation=f"Game outcome",
            metadata={...game_results...}
        )
```

## Score Object

Each game returns a Score with:

```python
Score(
    value=1,  # 1 for blue win, 0 for red
    answer="True",  # String representation
    explanation="Blues won after 5 rounds",
    metadata={
        "num_players": 5,
        "rounds_played": 5,
        "blues_win": True,
        "reds_win": False,
        "avg_entropy_reduction": 0.45,
        "avg_belief_alignment": 0.62,
        "avg_brier": 0.38,
        "apt_leader_deception": 0.71,
        "public_log": [...],
        "voting_history": {...},
        "roles": {...},
        "fired_players": [...],
        "patch_track": {...},
        "true_apt_leader": "p2",
        "belief_histories": {...}
    }
)
```

## Testing

```bash
# Test refactored architecture
python test_refactored_task.py

# Test voting system
python test_game_voting.py

# Test game flow
python test_e2e_game_flow.py

# Test critical mechanics
python test_critical.py
```

## Implementation Details

### Solver Output Structure
```python
state.output = {
    "env": RedvsBlueEnv,  # Game environment
    "game_results": {
        "blues_win": bool,
        "reds_win": bool,
        "true_apt_leader": str,
        "rounds": int,
        "avg_entropy": float,
        "avg_alignment": float,
        "avg_brier": float,
        "apt_leader_deception": float,
        "public_log": list,
        "voting_history": dict,
        "roles": dict,
        "fired_players": list,
        "patch_track": dict,
        "belief_histories": dict,
    }
}
```

### Sample Metadata Structure
```python
sample.metadata = {
    "game_id": int,          # 0-indexed game number
    "num_players": int,       # Players per game
    "max_rounds": int,        # Max rounds per game
    "seed": int | None,       # Unique seed or None
}
```

## Example: Multi-Game Evaluation

```python
from red_vs_blue.task import red_vs_blue_task

# Create task with 5 games
task = red_vs_blue_task(
    num_games=5,
    num_players=5,
    max_rounds=10,
    seed=42
)

# When evaluated, will:
# 1. Create 5 samples with seeds 42-46
# 2. Run solver for each sample
# 3. Run scorer for each sample
# 4. Compute aggregate metrics:
#    - Blue win rate: X/5
#    - Red win rate: Y/5
#    - Average rounds: Z

print(f"Samples: {len(task.dataset)}")  # 5
print(f"Metrics: {len(task.metrics)}")  # 3
```

## Advanced Topics

### Custom Metrics
```python
def my_metric() -> Callable[[list[SampleScore]], float]:
    def metric(sample_scores: list[SampleScore]) -> float:
        # Process scores and return float
        return 0.0
    metric.__name__ = "my_metric"
    return metric
```

### Parametrized Games
```python
# Game parameters are in sample metadata
for sample in task.dataset:
    seed = sample.metadata["seed"]
    num_players = sample.metadata["num_players"]
    # Can customize game parameters per sample
```

### TaskState Access
```python
# In solver
state.current_sample.metadata  # Game parameters
state.output                   # Store game results

# In scorer
state.output["game_results"]   # Access solver results
```

---
