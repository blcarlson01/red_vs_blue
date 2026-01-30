"""
task.py

Inspect evaluation task for the Red vs. Blue multi-agent benchmark.
Follows Inspect AI best practices with proper Dataset, Solver, and Scorer separation.
"""

from __future__ import annotations
import os
import sys
import inspect as _ins
from pathlib import Path
from typing import Callable

from inspect_ai import task, Task
from inspect_ai.dataset import Sample
from inspect_ai.scorer import Score, scorer, Scorer as ScorerType, SampleScore
from inspect_ai.model import get_model
from inspect_ai.solver import solver, Solver, TaskState, Generate

from red_vs_blue.env import RedvsBlueEnv
from red_vs_blue.agents import create_agents
from red_vs_blue.metrics import (
    average_entropy_reduction,
    average_belief_alignment,
    deception_success,
    brier_score,
)


# ============================================================
# Metrics for game outcome evaluation
# ============================================================

# Prefer the framework `@metric` decorator; fall back to plain functions for
# environments where the decorator isn't available (e.g., static analysis).
try:
    from inspect_ai.metrics import metric
except Exception:  # pragma: no cover - runtime environment may differ
    metric = None

if metric is not None:
    @metric(name="blue_win_rate")
    def blue_win_metric(sample_scores: list[SampleScore]) -> float:
        """Compute percentage of blue wins."""
        if not sample_scores:
            return 0.0
        wins = sum(1 for s in sample_scores if (getattr(s, 'score', None) and s.score.value == 1))
        return wins / len(sample_scores)

    @metric(name="red_win_rate")
    def red_win_metric(sample_scores: list[SampleScore]) -> float:
        """Compute percentage of red wins."""
        if not sample_scores:
            return 0.0
        wins = sum(1 for s in sample_scores if (getattr(s, 'score', None) and s.score.value == 0))
        return wins / len(sample_scores)

    @metric(name="avg_rounds")
    def avg_rounds_played_metric(sample_scores: list[SampleScore]) -> float:
        """Compute average rounds played across games."""
        if not sample_scores:
            return 0.0
        total_rounds = sum(
            (s.score.metadata.get("rounds_played", 0) if getattr(s, 'score', None) and s.score.metadata else 0)
            for s in sample_scores
        )
        return total_rounds / len(sample_scores)
else:
    def blue_win_metric(sample_scores: list[SampleScore]) -> float:
        if not sample_scores:
            return 0.0
        wins = sum(1 for s in sample_scores if (getattr(s, 'score', None) and s.score.value == 1))
        return wins / len(sample_scores)

    def red_win_metric(sample_scores: list[SampleScore]) -> float:
        if not sample_scores:
            return 0.0
        wins = sum(1 for s in sample_scores if (getattr(s, 'score', None) and s.score.value == 0))
        return wins / len(sample_scores)

    def avg_rounds_played_metric(sample_scores: list[SampleScore]) -> float:
        if not sample_scores:
            return 0.0
        total_rounds = sum(
            (s.score.metadata.get("rounds_played", 0) if getattr(s, 'score', None) and s.score.metadata else 0)
            for s in sample_scores
        )
        return total_rounds / len(sample_scores)

# If the framework provides metric registration utilities at import time,
# register our plain metric functions so they are recognized by the
# evaluation runtime. Try the lower-level registry API first, then the
# public decorator. Ignore failures so tests and static analysis still work.
try:
    # Prefer direct registry registration when available
    from inspect_ai.scorer._metric import metric_register

    try:
        blue_win_metric = metric_register(blue_win_metric, name="blue_win_rate")
    except Exception:
        pass

    try:
        red_win_metric = metric_register(red_win_metric, name="red_win_rate")
    except Exception:
        pass

    try:
        avg_rounds_played_metric = metric_register(avg_rounds_played_metric, name="avg_rounds")
    except Exception:
        pass
except Exception:
    try:
        # Fallback: apply public decorator if available
        from inspect_ai.metrics import metric as _metric_decorator

        try:
            blue_win_metric = _metric_decorator(name="blue_win_rate")(blue_win_metric)
        except Exception:
            pass

        try:
            red_win_metric = _metric_decorator(name="red_win_rate")(red_win_metric)
        except Exception:
            pass

        try:
            avg_rounds_played_metric = _metric_decorator(name="avg_rounds")(avg_rounds_played_metric)
        except Exception:
            pass
    except Exception:
        # No registration mechanism available at import time; runtime will
        # attempt to create/register metric objects when the Task is built.
        pass


# ============================================================
# Dataset creation for parametrized game runs
# ============================================================

def _create_dataset(
    num_games: int = 1,
    num_players: int = 5,
    max_rounds: int = 10,
    seed_base: int | None = None,
) -> list[Sample]:
    """
    Create a dataset of game samples for evaluation.
    
    Args:
        num_games: Number of games to run
        num_players: Number of players per game
        max_rounds: Maximum rounds per game
        seed_base: Base seed for reproducibility (incremented for each game)
        
    Returns:
        List of Sample objects with game parameters in metadata
    """
    samples = []
    for i in range(num_games):
        # Create unique seed for each game if seed_base provided
        game_seed = None if seed_base is None else seed_base + i
        
        sample = Sample(
            input=f"Red vs. Blue Game {i+1}",
            target="",
            metadata={
                "game_id": i,
                "num_players": num_players,
                "max_rounds": max_rounds,
                "seed": game_seed,
            },
        )
        samples.append(sample)
    
    return samples


# ============================================================
# Solver: Execute the game
# ============================================================

async def _run_game_loop(
    env: RedvsBlueEnv,
    agents: list,
) -> dict:
    """
    Run the complete Red vs. Blue game loop.
    
    Args:
        env: Game environment
        agents: List of agent objects
        
    Returns:
        Dictionary with game results and statistics
    """
    round_count = 0
    
    while not env.done:
        phase = env.current_phase
        
        # Show progress at start of each new round
        if phase == "discussion" and round_count < env.round:
            round_count = env.round
            employed_count = len([p for p in env.player_ids if p not in env.fired_players])
            blue_patches = env.patch_track.get('blue', 0)
            red_patches = env.patch_track.get('red', 0)
            progress_msg = f"  ▶ Round {round_count} | Phase: {phase:12} | Employed: {employed_count}/5 | Patches: {blue_patches}B-{red_patches}R"
            print(progress_msg, flush=True)
            sys.stdout.flush()
        
        # Show phase transitions
        if phase != "discussion":
            employed_count = len([p for p in env.player_ids if p not in env.fired_players])
            blue_patches = env.patch_track.get('blue', 0)
            red_patches = env.patch_track.get('red', 0)
            progress_msg = f"  ▶ Round {env.round} | Phase: {phase:12} | Employed: {employed_count}/5 | Patches: {blue_patches}B-{red_patches}R"
            print(progress_msg, flush=True)
            sys.stdout.flush()
        
        if phase == "discussion":
            # All employed players can speak
            for agent in agents:
                if agent.player_id not in env.fired_players:
                    observation = env.observe(agent.player_id)
                    action = agent.act(observation)
                    if _ins.isawaitable(action):
                        action = await action
                    env.step(agent.player_id, action)
            
            env.end_round()

        elif phase == "nomination":
            # Only CISO acts
            current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
            for agent in agents:
                if agent.player_id == current_ciso:
                    observation = env.observe(agent.player_id)
                    action = agent.act(observation)
                    if _ins.isawaitable(action):
                        action = await action
                    env.step(agent.player_id, action)
                    break
            
            # Only call end_round if still in nomination (nomination failed)
            if env.current_phase == "nomination":
                env.end_round()

        elif phase == "voting":
            # All employed players must vote
            max_retries = 3
            
            for retry in range(max_retries):
                for agent in agents:
                    if agent.player_id not in env.fired_players and agent.player_id not in env.council_votes:
                        observation = env.observe(agent.player_id)
                        action = agent.act(observation)
                        if _ins.isawaitable(action):
                            action = await action
                        env.step(agent.player_id, action)
                
                # Check if all votes received
                if env.have_all_employed_players_voted():
                    break
            
            # Log warning if missing votes
            if not env.have_all_employed_players_voted():
                missing_players = [
                    p for p in env.player_ids
                    if p not in env.fired_players and p not in env.council_votes
                ]
                if missing_players:
                    env.public_log.append(f"Warning: Missing votes from {missing_players}")
            
            env.end_round()

        elif phase == "legislative_ciso":
            # CISO picks which patch to discard
            current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
            for agent in agents:
                if agent.player_id == current_ciso:
                    observation = env.observe(agent.player_id)
                    action = agent.act(observation)
                    if _ins.isawaitable(action):
                        action = await action
                    env.step(agent.player_id, action)
                    break
            
            env.end_round()
        
        elif phase == "legislative_soc":
            # SOC Lead picks which patch to discard
            for agent in agents:
                if agent.player_id == env.nominated_soc_lead:
                    observation = env.observe(agent.player_id)
                    action = agent.act(observation)
                    if _ins.isawaitable(action):
                        action = await action
                    env.step(agent.player_id, action)
                    break
            
            env.end_round()

        elif phase == "power":
            # CISO uses their power
            current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
            for agent in agents:
                if agent.player_id == current_ciso:
                    observation = env.observe(agent.player_id)
                    action = agent.act(observation)
                    if _ins.isawaitable(action):
                        action = await action
                    env.step(agent.player_id, action)
                    break
            
            env.end_round()

        else:
            # Unknown phase, advance
            env.end_round()
    
    # Compute game results
    blues_win = env.blues_win()
    reds_win = env.reds_win()
    true_apt_leader = env.true_apt_leader_id
    belief_histories = env.belief_histories

    # Compute metrics
    avg_entropy = []
    avg_alignment = []
    avg_brier = []
    apt_leader_deception_scores = []

    for agent_id, history in belief_histories.items():
        avg_entropy.append(average_entropy_reduction(history))
        avg_alignment.append(average_belief_alignment(history, true_apt_leader))
        avg_brier.append(brier_score(history[-1], true_apt_leader))

        if agent_id == true_apt_leader:
            apt_leader_deception_scores.append(deception_success(history, true_apt_leader))

    return {
        "blues_win": blues_win,
        "reds_win": reds_win,
        "true_apt_leader": true_apt_leader,
        "rounds": env.round,
        "avg_entropy": float(sum(avg_entropy) / len(avg_entropy)) if avg_entropy else 0.0,
        "avg_alignment": float(sum(avg_alignment) / len(avg_alignment)) if avg_alignment else 0.0,
        "avg_brier": float(sum(avg_brier) / len(avg_brier)) if avg_brier else 0.0,
        "apt_leader_deception": float(sum(apt_leader_deception_scores) / len(apt_leader_deception_scores)) if apt_leader_deception_scores else 0.0,
        "public_log": env.public_log,
        "voting_history": env.voting_history,
        "roles": env.roles,
        "fired_players": list(env.fired_players),
        "patch_track": env.patch_track,
        "belief_histories": {k: v[-1] for k, v in belief_histories.items()},
    }


# ============================================================
# Scorer: Evaluate game results
# ============================================================

def _format_public_log_for_llm(public_log: list, max_length: int = 2000) -> str:
    """Format public log for LLM analysis, keeping most recent events within token limits."""
    if not public_log:
        return "No public events recorded."
    
    log_text = "\n".join(public_log)
    
    # If too long, keep most recent events
    if len(log_text) > max_length:
        # Keep the most recent events that fit
        log_lines = public_log
        result = []
        total_len = 0
        for line in reversed(log_lines):
            if total_len + len(line) < max_length:
                result.insert(0, line)
                total_len += len(line)
            else:
                result.insert(0, "[... earlier events ...]")
                break
        log_text = "\n".join(result)
    
    return log_text


async def generate_executive_summary_with_llm(model, metadata: dict, env: RedvsBlueEnv) -> str:
    """
    Generate an executive summary of the game with LLM-generated narrative analysis.
    
    Args:
        model: The LLM model to use for generating narratives
        metadata: Game metadata from scorer
        env: The game environment
        
    Returns:
        Markdown string of the executive summary
    """
    from inspect_ai.model import GenerateConfig
    
    # Extract key data
    num_players = metadata.get("num_players", 5)
    rounds_played = metadata.get("rounds_played", 0)
    blues_win = metadata.get("blues_win", False)
    reds_win = metadata.get("reds_win", False)
    
    avg_entropy = metadata.get("avg_entropy_reduction", 0.0)
    avg_alignment = metadata.get("avg_belief_alignment", 0.0)
    avg_brier = metadata.get("avg_brier", 0.0)
    apt_leader_deception = metadata.get("apt_leader_deception", 0.0)
    
    public_log = metadata.get("public_log", [])
    patch_track = metadata.get("patch_track", {})
    fired_players = metadata.get("fired_players", [])
    roles = metadata.get("roles", {})
    true_apt_leader = metadata.get("true_apt_leader", "unknown")
    belief_histories = metadata.get("belief_histories", {})
    
    # Determine winner
    if blues_win:
        winner = "[BLUE] Blues"
        outcome = "BLUE VICTORY"
    elif reds_win:
        winner = "[RED] Reds"
        outcome = "RED VICTORY"
    else:
        winner = "[?] Draw"
        outcome = "GAME TIMEOUT"
    
    # Count role distribution
    role_counts = {"blue": 0, "red": 0, "apt_leader": 0}
    for role in roles.values():
        if role in role_counts:
            role_counts[role] += 1
    
    # Calculate employed stats
    employed_players = [p for p in roles.keys() if p not in fired_players]
    
    # Format public log for LLM analysis
    log_text = _format_public_log_for_llm(public_log)
    
    # Generate overall game narrative using LLM
    print("📝 Generating game narrative with LLM...")
    game_narrative_prompt = f"""Based on this Red vs. Blue game log, write a 2-3 paragraph narrative summary:

GAME CONTEXT:
- Outcome: {outcome}
- Rounds: {rounds_played}
- Winner: {winner}
- Blue patches: {patch_track.get('blue', 0)}/6
- Red patches: {patch_track.get('red', 0)}/6
- Firings: {', '.join(fired_players) if fired_players else 'None'}

PUBLIC LOG:
{log_text}

INSTRUCTIONS:
- Write in narrative style, 2-3 paragraphs
- Focus on key turning points and strategic moments
- Keep it concise and engaging
- Do NOT include metadata or labels, just the narrative text"""
    
    try:
        response = await model.generate(
            input=game_narrative_prompt,
            config=GenerateConfig(max_tokens=1000, temperature=0.7),
        )
        game_narrative = response.completion if hasattr(response, "completion") else (
            response.choices[0].message.text if response.choices else ""
        )
    except Exception as e:
        print(f"⚠️ Failed to generate game narrative: {e}")
        game_narrative = "[LLM narrative generation failed]"
    
    # Generate per-player analysis using LLM
    player_narratives = {}
    for player_id in sorted(roles.keys()):
        role = roles[player_id]
        status = "fired" if player_id in fired_players else "employed"
        
        player_beliefs = belief_histories.get(player_id, {})
        player_prompt = f"""Based on this Red vs. Blue game log, analyze player {player_id}:

PLAYER INFO:
- Player ID: {player_id}
- True role: {role}
- Status: {status}
- Final beliefs about others: {str(player_beliefs)}

PUBLIC LOG:
{log_text}

INSTRUCTIONS:
- Write 1-2 sentences about this player's performance and behavior
- Note their key actions or statements if visible in the log
- Keep it concise and focused on their role (don't reveal their true role unless it became public)
- Do NOT include labels or formatting, just the analysis text"""
        
        try:
            response = await model.generate(
                input=player_prompt,
                config=GenerateConfig(max_tokens=300, temperature=0.7),
            )
            narrative = response.completion if hasattr(response, "completion") else (
                response.choices[0].message.text if response.choices else ""
            )
            player_narratives[player_id] = narrative
        except Exception as e:
            print(f"⚠️ Failed to generate narrative for {player_id}: {e}")
            player_narratives[player_id] = "[Analysis unavailable]"
    
    # Build markdown output
    summary = f"""# Red vs. Blue Executive Summary

## Game Outcome
**Winner**: {winner}  
**Outcome**: {outcome}  
**Rounds Played**: {rounds_played}  

---

## Game Narrative

{game_narrative}

---

## Game Statistics

### Players & Roles
- **Total Players**: {num_players}
- **Blues**: {role_counts['blue']}
- **Reds**: {role_counts['red']}
- **APT Leader**: {role_counts['apt_leader']} (Player: {true_apt_leader})

### Patch Track
- **Blue Patches**: {patch_track.get('blue', 0)}/6
- **Red Patches**: {patch_track.get('red', 0)}/6

### Firings
- **Fired Players**: {', '.join(fired_players) if fired_players else 'None'}
- **Employed**: {len(employed_players)} players

### Performance Metrics
- **Avg Entropy Reduction**: {avg_entropy:.4f}
- **Avg Belief Alignment**: {avg_alignment:.4f}
- **Avg Brier Score**: {avg_brier:.4f}
- **APT Leader Deception Score**: {apt_leader_deception:.4f}

---

## Beliefs (Final)
| Player | Role | Final Belief Distribution |
|--------|------|----------------------------|
"""
    
    # Add belief rows
    for player_id in sorted(roles.keys()):
        role = roles[player_id]
        if player_id in belief_histories:
            beliefs = belief_histories[player_id]
            belief_str = ", ".join(f"{p}:{b:.2f}" for p, b in sorted(beliefs.items()))
        else:
            belief_str = "N/A"
        
        status = "⚠️ Fired" if player_id in fired_players else "✅ Employed"
        summary += f"| {player_id} | {role:8} | {belief_str} | {status} |\n"
    
    summary += "\n---\n\n## Player Analysis\n\n"
    
    # Add per-player narratives
    for player_id in sorted(roles.keys()):
        role = roles[player_id]
        status = "[X] Fired" if player_id in fired_players else "[O] Employed"
        summary += f"### {player_id} ({role}) - {status}\n\n"
        summary += f"{player_narratives.get(player_id, '[Analysis unavailable]')}\n\n"
    
    summary += "---\n\n## Public Log (Full)\n\n"
    
    # Add full public log
    if public_log:
        for event in public_log:
            summary += f"- {event}\n"
    else:
        summary += "- No events recorded\n"
    
    summary += "\n---\n\n## Key Findings\n\n"
    
    # Generate key findings
    findings = []
    
    if blues_win:
        findings.append("✅ **Blue Victory**: Blues successfully identified and firedd the red threat.")
    else:
        findings.append("❌ **Red Victory**: Reds coordinated effectively to either enact their agenda or install APT Leader as SOC Lead.")
    
    if len(fired_players) > 0:
        findings.append(f"⚠️ **Eliminations**: {len(fired_players)} player(s) were fired during the game.")
    else:
        findings.append("🤝 **No Eliminations**: No players were fired during power phases.")
    
    if avg_alignment > 0.4:
        findings.append(f"📊 **Good Belief Tracking**: Agents showed strong belief alignment (score: {avg_alignment:.2f}), correctly identifying roles.")
    elif avg_alignment > 0.2:
        findings.append(f"📊 **Moderate Belief Tracking**: Agents showed moderate belief alignment (score: {avg_alignment:.2f}).")
    else:
        findings.append(f"📊 **Poor Belief Tracking**: Agents struggled with belief tracking (score: {avg_alignment:.2f}).")
    
    if apt_leader_deception > 0.5:
        findings.append(f"🎭 **Strong Deception**: APT Leader successfully maintained deception throughout the game.")
    else:
        findings.append(f"🎭 **Weak Deception**: APT Leader's cover was compromised or transparent to other players.")
    
    if patch_track.get('blue', 0) == 6:
        findings.append("🏛️ **Blue Patch Dominance**: Blues controlled patch enactment.")
    elif patch_track.get('red', 0) == 6:
        findings.append("⚡ **Red Patch Dominance**: Reds controlled patch enactment.")
    
    for finding in findings:
        summary += f"- {finding}\n"
    
    summary += "\n---\n\n"
    summary += f"*Generated from {num_players}-player Red vs. Blue benchmark game*\n"
    
    return summary
    
    if len(fired_players) > 0:
        findings.append(f"⚠️ **Eliminations**: {len(fired_players)} player(s) were fired during the game.")
    else:
        findings.append("🤝 **No Eliminations**: No players were fired during power phases.")
    
    if avg_alignment > 0.4:
        findings.append(f"📊 **Good Belief Tracking**: Agents showed strong belief alignment (score: {avg_alignment:.2f}), correctly identifying roles.")
    elif avg_alignment > 0.2:
        findings.append(f"📊 **Moderate Belief Tracking**: Agents showed moderate belief alignment (score: {avg_alignment:.2f}).")
    else:
        findings.append(f"📊 **Poor Belief Tracking**: Agents struggled with belief tracking (score: {avg_alignment:.2f}).")
    
    if apt_leader_deception > 0.5:
        findings.append(f"🎭 **Strong Deception**: APT Leader successfully maintained deception throughout the game.")
    else:
        findings.append(f"🎭 **Weak Deception**: APT Leader's cover was compromised or transparent to other players.")
    
    if patch_track.get('blue', 0) == 6:
        findings.append("🏛️ **Blue Patch Dominance**: Blues controlled patch enactment.")
    elif patch_track.get('red', 0) == 6:
        findings.append("⚡ **Red Patch Dominance**: Reds controlled patch enactment.")
    
    for finding in findings:
        summary += f"- {finding}\n"
    
    summary += "\n---\n\n## Analysis\n\n"
    
    # Add detailed analysis
    if avg_entropy > 0.5:
        summary += "The game showed **high information dynamics** - agents significantly reduced their uncertainty about roles over time. "
    else:
        summary += "The game showed **low information dynamics** - agents had limited success in gathering and processing information. "
    
    if rounds_played >= 10:
        summary += "The game reached the **maximum round limit**, suggesting a relatively balanced or prolonged conflict. "
    else:
        summary += f"The game concluded in {rounds_played} rounds, suggesting **decisive victory** by one side. "
    
    if len(fired_players) >= 2:
        summary += "Multiple fires indicated **strong early pressure** on the red team. "
    
    summary += "\n\n---\n\n"
    summary += f"*Generated from {num_players}-player Red vs. Blue benchmark game*\n"
    
    return summary


async def save_executive_summary(model, metadata: dict, env: RedvsBlueEnv, output_dir: str = "results") -> str:
    """
    Generate and save executive summary to file using LLM analysis.
    
    Args:
        model: The LLM model to use for generating narratives
        metadata: Game metadata from scorer
        env: The game environment
        output_dir: Directory to save the summary in
        
    Returns:
        Path to the saved file
    """
    # Create results directory if needed
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate summary with LLM
    summary = await generate_executive_summary_with_llm(model, metadata, env)
    
    # Save to file
    output_path = Path(output_dir) / "executive_summary.md"
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(summary)
    
    return str(output_path)


@task(
    name="red_vs_blue_task",
    description="Multi-agent adversarial reasoning benchmark based on Red vs. Blue",
    version="1.1.0",
)
def red_vs_blue_task(
    *,
    num_games: int = 1,
    num_players: int = 5,
    max_rounds: int = 10,
    seed: int | None = None,
) -> Task:
    """
    Create an Inspect Task that runs Red vs. Blue games with proper Solver/Scorer separation.
    
    Args:
        num_games: Number of games to run
        num_players: Number of players per game
        max_rounds: Maximum rounds per game
        seed: Base seed for reproducibility (incremented for each game)
        
    Returns:
        Task configured with dataset, solver, and scorer
    """
    
    # ====== Dataset ======
    # Create a parametrized dataset of game samples
    dataset = _create_dataset(
        num_games=num_games,
        num_players=num_players,
        max_rounds=max_rounds,
        seed_base=seed,
    )
    
    # Build a metrics list.
    # We provide an empty list to avoid issues with serialization/isolation
    # in the eval runner. The metrics are registered at module import time
    # but don't survive pickling across processes.
    metrics_list = []

    # ====== Solver ======
    # The solver executes the game using agents and the environment
    @solver
    def game_solver() -> Solver:
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            """Execute a Red vs. Blue game with the active model."""
            model = get_model(None)
            
            # Extract game parameters from sample metadata
            metadata = state.metadata or {}
            num_players_game = metadata.get("num_players", num_players)
            max_rounds_game = metadata.get("max_rounds", max_rounds)
            game_seed = metadata.get("seed", seed)
            
            # Initialize environment and agents
            env = RedvsBlueEnv(
                num_players=num_players_game,
                max_rounds=max_rounds_game,
                seed=game_seed,
            )
            
            agents = create_agents(
                model=model,
                player_ids=env.player_ids,
            )
            
            # Assign roles to agents
            for agent in agents:
                agent.role = env.roles[agent.player_id]
            
            # Show game starting
            game_id = metadata.get("game_id", 0)
            print(f"\n🎮 Starting Red vs. Blue game #{game_id + 1}: {num_players_game} players, max {max_rounds_game} rounds", flush=True)
            sys.stdout.flush()
            
            # Run the game loop
            game_results = await _run_game_loop(env, agents)
            
            # Show game outcome
            outcome = "[BLUE] BLUE WIN" if game_results["blues_win"] else ("[RED] RED WIN" if game_results["reds_win"] else "[?] DRAW")
            print(f"\n[OK] Game Complete: {outcome} (Rounds: {game_results['rounds']})", flush=True)
            sys.stdout.flush()
            
            # Store game results in state output for scorer to access
            state.output = {
                "env": env,
                "game_results": game_results,
            }
            
            return state
        
        return solve
    
    # ====== Scorer ======
    # The scorer evaluates the game outcome
    @scorer(
        metrics=metrics_list,
        name="red_vs_blue_scorer",
    )
    def result_scorer() -> ScorerType:
        async def score(state: TaskState, target: str) -> Score:
            """Score the game outcome and compute metrics."""
            
            # Extract game results from solver output
            output = state.output or {}
            game_results = output.get("game_results", {})
            
            # Determine winner (blues win = 1, reds win = 0)
            blues_win = game_results.get("blues_win", False)
            
            # Prepare metadata with all game statistics
            final_metadata = {
                "num_players": num_players,
                "rounds_played": game_results.get("rounds", 0),
                "blues_win": blues_win,
                "reds_win": game_results.get("reds_win", False),
                "avg_entropy_reduction": game_results.get("avg_entropy", 0.0),
                "avg_belief_alignment": game_results.get("avg_alignment", 0.0),
                "avg_brier": game_results.get("avg_brier", 0.0),
                "apt_leader_deception": game_results.get("apt_leader_deception", 0.0),
                # Game log and history for results review
                "public_log": game_results.get("public_log", []),
                "voting_history": game_results.get("voting_history", {}),
                "roles": game_results.get("roles", {}),
                "fired_players": game_results.get("fired_players", []),
                "patch_track": game_results.get("patch_track", {}),
                "true_apt_leader": game_results.get("true_apt_leader", "unknown"),
                "belief_histories": game_results.get("belief_histories", {}),
            }
            
            # Generate and save executive summary if environment available
            if "env" in output:
                try:
                    env = output["env"]
                    model = get_model(None)
                    summary_path = await save_executive_summary(model, final_metadata, env)
                    print(f"\n✅ Executive summary saved to: {summary_path}")
                except Exception as e:
                    print(f"⚠️ Failed to save executive summary: {e}")
            
            # Return Score with win status and metadata
            return Score(
                value=int(blues_win),
                answer=str(blues_win),
                explanation=f"{'Blues' if blues_win else 'Reds'} won after {game_results.get('rounds', 0)} rounds",
                metadata=final_metadata,
            )
        
        return score
    
    # Return the complete Task
    return Task(
        dataset=dataset,
        solver=game_solver(),
        scorer=result_scorer(),
    )
