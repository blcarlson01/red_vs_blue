"""
advanced_analysis.py

Advanced analysis for the Red vs. Blue benchmark including:
- Per-role performance metrics
- Voting correlation analysis
- Belief dynamics
- Patch track momentum
- Deception effectiveness
- Early-game predictors
- Model-comparative analysis
"""

from __future__ import annotations
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ============================================================
# Per-Role Analysis
# ============================================================

def analyze_per_role(df: pd.DataFrame) -> Dict:
    """
    Analyze win rates, rounds, and metrics broken down by role.
    """
    if "roles" not in df.columns:
        return {"note": "Role data not available in aggregated results"}
    
    results = {}
    
    for _, row in df.iterrows():
        roles = row.get("roles", {})
        if not isinstance(roles, dict) or not roles:
            continue
        rounds_played = row.get("rounds_played", 0)
        blues_win = row.get("blues_win", False)
        fired_players = row.get("fired_players", [])
        
        for player_id, role in roles.items():
            player_id = str(player_id)
            
            if role not in results:
                results[role] = {
                    "games_played": 0,
                    "wins": 0,
                    "employed_rate": 0.0,
                    "avg_rounds": 0.0,
                    "belief_alignment": [],
                    "brier_score": [],
                }
            
            results[role]["games_played"] += 1
            
            # Win if role matches outcome
            if (role == "blue" and blues_win) or (role == "red" and not blues_win):
                results[role]["wins"] += 1
            
            # Employed (not in fired_players)
            if player_id not in fired_players:
                results[role]["employed_rate"] += 1
            
            results[role]["avg_rounds"] += rounds_played
            
            # Belief metrics by player
            belief_hist = row.get("belief_histories", {}).get(player_id, {})
            if belief_hist:
                results[role]["belief_alignment"].append(row.get("avg_belief_alignment", 0))
                results[role]["brier_score"].append(row.get("avg_brier", 0))
    
    # Normalize
    for role, stats in results.items():
        games = stats["games_played"]
        if games > 0:
            stats["win_rate"] = stats["wins"] / games
            stats["employed_rate"] /= games
            stats["avg_rounds"] /= games
            if stats["belief_alignment"]:
                stats["avg_belief_alignment"] = float(np.mean(stats["belief_alignment"]))
                stats["avg_brier_score"] = float(np.mean(stats["brier_score"]))
            del stats["belief_alignment"]
            del stats["brier_score"]
    
    return results


def print_per_role_analysis(analysis: Dict):
    """Print per-role performance table."""
    print("\n" + "="*80)
    print("PER-ROLE PERFORMANCE ANALYSIS")
    print("="*80 + "\n")
    
    if "note" in analysis:
        print(analysis["note"])
        print()
        return
    
    df = pd.DataFrame(analysis).T
    df = df.sort_values("games_played", ascending=False)
    
    print(df.to_string())
    print()


# ============================================================
# Voting Correlation Analysis
# ============================================================

def analyze_voting_patterns(df: pd.DataFrame) -> Dict:
    """
    Analyze voting patterns: consistency, coalition formation, etc.
    """
    if "voting_history" not in df.columns or "roles" not in df.columns:
        return {"note": "Voting history or role data not available"}
    
    results = {
        "total_votes_recorded": 0,
        "vote_consistency": [],  # Votes aligned with final beliefs
        "coalition_strength": [],  # Do reds vote together?
        "swing_voters": [],  # Players with inconsistent votes
    }
    
    for _, row in df.iterrows():
        voting_history = row.get("voting_history", [])
        roles = row.get("roles", {})
        
        if not voting_history or not isinstance(voting_history, list) or not roles:
            continue
        
        results["total_votes_recorded"] += len(voting_history)
        
        # Analyze vote patterns per round
        for round_data in voting_history:
            if not isinstance(round_data, dict) or "votes" not in round_data:
                continue
            
            votes = round_data.get("votes", {})
            
            # Collect votes by role
            red_votes = {}
            blue_votes = {}
            
            for player_id, vote in votes.items():
                player_id_str = str(player_id)
                role = roles.get(player_id_str, "unknown")
                
                if role == "red" or role == "apt_leader":  # apt_leader is a red role
                    red_votes[player_id_str] = vote
                elif role == "blue":
                    blue_votes[player_id_str] = vote
            
            # Check red coalition strength (are reds voting together?)
            if len(red_votes) > 1:
                # Count how many reds voted the same way
                vote_values = list(red_votes.values())
                max_agreement = max(
                    sum(1 for v in vote_values if v == "yes"),
                    sum(1 for v in vote_values if v == "no")
                )
                consistency = max_agreement / len(red_votes)
                results["coalition_strength"].append(consistency)
    
    # Compute averages
    if results["coalition_strength"]:
        results["avg_coalition_strength"] = float(np.mean(results["coalition_strength"]))
    
    return results


def print_voting_analysis(analysis: Dict):
    """Print voting analysis."""
    print("\n" + "="*80)
    print("VOTING PATTERN ANALYSIS")
    print("="*80 + "\n")
    
    if "note" in analysis:
        print(analysis["note"])
        print()
        return
    
    print(f"Total votes recorded: {analysis['total_votes_recorded']}")
    if "avg_coalition_strength" in analysis:
        print(f"Avg red coalition strength: {analysis['avg_coalition_strength']:.3f}")
    print()


# ============================================================
# Belief Dynamics
# ============================================================

def analyze_belief_dynamics(df: pd.DataFrame) -> Dict:
    """
    Analyze how beliefs evolved (entropy reduction, convergence).
    """
    if "avg_entropy_reduction" not in df.columns or "avg_belief_alignment" not in df.columns:
        return {"note": "Belief dynamics data not available (entropy or belief alignment columns missing)"}
    
    results = {
        "total_games": len(df),
        "avg_entropy_reduction": float(df["avg_entropy_reduction"].mean()),
        "min_entropy_reduction": float(df["avg_entropy_reduction"].min()),
        "max_entropy_reduction": float(df["avg_entropy_reduction"].max()),
        "avg_final_alignment": float(df["avg_belief_alignment"].mean()),
        "high_entropy_games": 0,  # Games with significant information gathering
        "low_entropy_games": 0,   # Games with little belief change
    }
    
    entropy_threshold_high = df["avg_entropy_reduction"].quantile(0.75)
    entropy_threshold_low = df["avg_entropy_reduction"].quantile(0.25)
    
    results["high_entropy_games"] = int((df["avg_entropy_reduction"] >= entropy_threshold_high).sum())
    results["low_entropy_games"] = int((df["avg_entropy_reduction"] <= entropy_threshold_low).sum())
    
    return results



def print_belief_dynamics(analysis: Dict):
    """Print belief dynamics analysis."""
    print("\n" + "="*80)
    print("BELIEF DYNAMICS ANALYSIS")
    print("="*80 + "\n")
    
    if "note" in analysis:
        print(analysis["note"])
        print()
        return
    
    print(f"Games analyzed: {analysis['total_games']}")
    print(f"Average entropy reduction: {analysis['avg_entropy_reduction']:.4f}")
    print(f"  (Range: {analysis['min_entropy_reduction']:.4f} to {analysis['max_entropy_reduction']:.4f})")
    print(f"Average final belief alignment: {analysis['avg_final_alignment']:.4f}")
    print(f"Games with high information gathering: {analysis['high_entropy_games']}")
    print(f"Games with low information gathering: {analysis['low_entropy_games']}")
    print()


# ============================================================
# Patch Track Momentum
# ============================================================

def analyze_patch_momentum(df: pd.DataFrame) -> Dict:
    """
    Analyze patch track: timing, distribution, momentum.
    """
    if "patch_track" not in df.columns:
        return {"note": "Patch track data not available in aggregated results"}
    
    results = {
        "avg_blue_patches": float(df["patch_track"].apply(lambda x: x.get("blue", 0) if isinstance(x, dict) else 0).mean()),
        "avg_red_patches": float(df["patch_track"].apply(lambda x: x.get("red", 0) if isinstance(x, dict) else 0).mean()),
        "games_blue_sweep": int((df["patch_track"].apply(lambda x: x.get("blue", 0) if isinstance(x, dict) else 0) >= 5).sum()),
        "games_red_sweep": int((df["patch_track"].apply(lambda x: x.get("red", 0) if isinstance(x, dict) else 0) >= 5).sum()),
    }
    
    # Momentum: last patch applied before win
    tied_games = int((df["patch_track"].apply(lambda x: (x.get("blue", 0) if isinstance(x, dict) else 0) < 5 and (x.get("red", 0) if isinstance(x, dict) else 0) < 5)).sum())
    results["games_ending_in_timeout"] = tied_games
    
    return results


def print_patch_momentum(analysis: Dict):
    """Print patch momentum analysis."""
    print("\n" + "="*80)
    print("PATCH TRACK MOMENTUM ANALYSIS")
    print("="*80 + "\n")
    
    if "note" in analysis:
        print(analysis["note"])
        print()
        return
    
    print(f"Average blue patches applied: {analysis['avg_blue_patches']:.2f} / 6")
    print(f"Average red patches applied: {analysis['avg_red_patches']:.2f} / 11")
    print(f"Games with blue sweep (6 blues): {analysis['games_blue_sweep']}")
    print(f"Games with red sweep (5 reds): {analysis['games_red_sweep']}")
    print(f"Games ending in timeout (no sweep): {analysis['games_ending_in_timeout']}")
    print()


# ============================================================
# Deception Effectiveness
# ============================================================

def analyze_deception_effectiveness(df: pd.DataFrame) -> Dict:
    """
    Analyze APT Leader deception vs red win rate.
    """
    if "apt_leader_deception" not in df.columns or "blues_win" not in df.columns:
        return {"note": "Deception data not available (apt_leader_deception or blues_win columns missing)"}
    
    # Filter out NaN values for these columns
    valid_df = df[["apt_leader_deception", "blues_win"]].dropna()
    if len(valid_df) == 0:
        return {"note": "No valid deception data available (all values are NaN)"}
    
    # Convert blues_win to boolean (1.0 = True, 0.0 = False)
    red_win = valid_df["blues_win"] == 0.0
    
    results = {
        "total_games": len(valid_df),
        "avg_apt_leader_deception": float(valid_df["apt_leader_deception"].mean()),
        "red_win_rate": float(red_win.mean()),
        "correlation_deception_to_win": 0.0,
    }
    
    if len(valid_df) > 1:
        correlation = valid_df["apt_leader_deception"].corr(red_win.astype(float))
        results["correlation_deception_to_win"] = float(correlation) if not np.isnan(correlation) else 0.0
    
    # High deception but lost vs Low deception but won
    high_deception = valid_df[valid_df["apt_leader_deception"] >= valid_df["apt_leader_deception"].quantile(0.75)]
    low_deception = valid_df[valid_df["apt_leader_deception"] <= valid_df["apt_leader_deception"].quantile(0.25)]
    
    high_red_win = high_deception["blues_win"] == 0.0
    low_red_win = low_deception["blues_win"] == 0.0
    
    results["high_deception_red_win_rate"] = float(high_red_win.mean()) if len(high_deception) > 0 else 0.0
    results["low_deception_red_win_rate"] = float(low_red_win.mean()) if len(low_deception) > 0 else 0.0
    
    return results


def print_deception_effectiveness(analysis: Dict):
    """Print deception effectiveness analysis."""
    print("\n" + "="*80)
    print("DECEPTION EFFECTIVENESS ANALYSIS")
    print("="*80 + "\n")
    
    if "note" in analysis:
        print(analysis["note"])
        print()
        return
    
    print(f"Games analyzed: {analysis['total_games']}")
    print(f"Average APT Leader deception score: {analysis['avg_apt_leader_deception']:.4f}")
    print(f"Overall red win rate: {analysis['red_win_rate']:.2%}")
    print(f"Correlation (deception -> red win): {analysis['correlation_deception_to_win']:.3f}")
    print(f"Red win rate (high deception): {analysis['high_deception_red_win_rate']:.2%}")
    print(f"Red win rate (low deception): {analysis['low_deception_red_win_rate']:.2%}")
    print()


# ============================================================
# Early-Game Predictors
# ============================================================

def analyze_early_game_predictors(df: pd.DataFrame) -> Dict:
    """
    Analyze which early-game metrics predict final outcome.
    """
    if "avg_entropy_reduction" not in df.columns or "rounds_played" not in df.columns:
        return {"note": "Early-game predictor data not available (entropy or rounds_played columns missing)"}
    
    results = {
        "early_entropy_reduction_predictor": 0.0,
        "first_round_rounds_predictor": 0.0,
        "round_2_entropy_predictor": 0.0,
    }
    
    if len(df) > 1:
        # Entropy in first 2 rounds as predictor of final entropy
        try:
            entropy_corr = df["avg_entropy_reduction"].corr(df["avg_entropy_reduction"])
            results["early_entropy_reduction_predictor"] = float(entropy_corr) if not np.isnan(entropy_corr) else 0.0
        except:
            pass
        
        # Rounds played predictor (high entropy = more rounds)
        try:
            rounds_corr = df["avg_entropy_reduction"].corr(df["rounds_played"])
            results["first_round_rounds_predictor"] = float(rounds_corr) if not np.isnan(rounds_corr) else 0.0
        except:
            pass
    
    # Game length distribution
    results["avg_game_length"] = float(df["rounds_played"].mean())
    results["min_game_length"] = int(df["rounds_played"].min())
    results["max_game_length"] = int(df["rounds_played"].max())
    
    return results


def print_early_game_predictors(analysis: Dict):
    """Print early-game predictor analysis."""
    print("\n" + "="*80)
    print("EARLY-GAME PREDICTOR ANALYSIS")
    print("="*80 + "\n")
    
    if "note" in analysis:
        print(analysis["note"])
        print()
        return
    
    print(f"Average game length: {analysis['avg_game_length']:.2f} rounds")
    print(f"Game length range: {analysis['min_game_length']} - {analysis['max_game_length']} rounds")
    print(f"Entropy reduction ↔ rounds correlation: {analysis['first_round_rounds_predictor']:.3f}")
    print()


# ============================================================
# Model-Comparative Analysis
# ============================================================

def analyze_model_comparison(df: pd.DataFrame) -> Dict:
    """
    Compare performance across different models.
    """
    results = {}
    
    if "model" not in df.columns or df["model"].isna().all():
        return {"note": "No model information available in results"}
    
    models = df["model"].unique()
    
    for model in models:
        model_df = df[df["model"] == model]
        results[model] = {
            "games_played": len(model_df),
            "blue_win_rate": float((model_df["blues_win"]).mean()),
            "avg_rounds": float(model_df["rounds_played"].mean()),
            "avg_entropy": float(model_df["avg_entropy_reduction"].mean()),
            "avg_belief_alignment": float(model_df["avg_belief_alignment"].mean()),
            "avg_apt_leader_deception": float(model_df["apt_leader_deception"].mean()),
        }
    
    return results


def print_model_comparison(analysis: Dict):
    """Print model comparison analysis."""
    print("\n" + "="*80)
    print("MODEL-COMPARATIVE ANALYSIS")
    print("="*80 + "\n")
    
    if "note" in analysis:
        print(analysis["note"])
        print()
        return
    
    df = pd.DataFrame(analysis).T
    df = df.sort_values("games_played", ascending=False)
    
    print(df.to_string())
    print()


# ============================================================
# Markdown Report Generation
# ============================================================

def generate_markdown_report(report: Dict, results_path: Path) -> str:
    """
    Generate a comprehensive markdown report with explanations.
    Returns the markdown content.
    """
    md = []
    
    # Header
    md.append("# Advanced Analysis Report")
    md.append("")
    md.append("## Executive Summary")
    md.append("")
    
    # Extract key metrics for summary
    per_role = report.get("per_role", {})
    voting = report.get("voting_patterns", {})
    beliefs = report.get("belief_dynamics", {})
    patch = report.get("patch_momentum", {})
    deception = report.get("deception_effectiveness", {})
    early_game = report.get("early_game_predictors", {})
    models = report.get("model_comparison", {})
    
    # Executive summary
    total_games = beliefs.get("total_games", 0)
    blue_role = per_role.get("blue", {})
    red_role = per_role.get("red", {})
    apt_leader_role = per_role.get("apt_leader", {})
    
    md.append(f"**Games Analyzed:** {total_games}")
    if blue_role:
        md.append(f"**Blue Win Rate:** {blue_role.get('win_rate', 0)*100:.1f}%")
    if red_role:
        md.append(f"**Red Win Rate:** {red_role.get('win_rate', 0)*100:.1f}%")
    md.append(f"**Average Game Length:** {early_game.get('avg_game_length', 0):.1f} rounds")
    md.append(f"**Average Entropy Reduction:** {beliefs.get('avg_entropy_reduction', 0):.4f}")
    md.append(f"**Average Belief Alignment:** {beliefs.get('avg_final_alignment', 0):.4f}")
    md.append("")
    
    # Key Findings
    md.append("### Key Findings")
    md.append("")
    
    findings = []
    
    # Win rate finding
    if blue_role and blue_role.get('win_rate', 0) == 0:
        findings.append("- **Red Dominance**: Reds won all analyzed games, indicating strong advantage or blue strategy weakness")
    elif blue_role and blue_role.get('win_rate', 0) == 1.0:
        findings.append("- **Blue Dominance**: Blues won all analyzed games, demonstrating effective identification strategy")
    elif blue_role:
        findings.append(f"- **Competitive Balance**: Blues achieved {blue_role.get('win_rate', 0)*100:.0f}% win rate, showing balanced gameplay")
    
    # Entropy finding
    avg_entropy = beliefs.get('avg_entropy_reduction', 0)
    if avg_entropy < 0.1:
        findings.append(f"- **Low Information Gathering**: Average entropy reduction of {avg_entropy:.4f} suggests limited belief updates")
    elif avg_entropy > 0.5:
        findings.append(f"- **Strong Information Gathering**: High entropy reduction ({avg_entropy:.4f}) indicates effective information processing")
    else:
        findings.append(f"- **Moderate Belief Convergence**: Entropy reduction of {avg_entropy:.4f} shows gradual information gathering")
    
    # Coalition finding
    if "avg_coalition_strength" in voting:
        strength = voting["avg_coalition_strength"]
        if strength > 0.8:
            findings.append(f"- **Strong Red Coalition**: Reds voted together {strength*100:.0f}% of the time, showing coordination")
        elif strength > 0.5:
            findings.append(f"- **Moderate Red Coordination**: Reds achieved {strength*100:.0f}% voting agreement")
        else:
            findings.append(f"- **Weak Red Coordination**: Low coalition strength ({strength*100:.0f}%) suggests disorganization")
    
    # Deception finding
    apt_deception = deception.get('avg_apt_leader_deception', 0)
    red_win = deception.get('red_win_rate', 0)
    if apt_deception > 0.5:
        findings.append(f"- **Effective Deception**: APT Leaders achieved {apt_deception:.2f} deception score")
    elif apt_deception == 0:
        findings.append(f"- **Transparent APT Leaders**: Low deception scores indicate APT Leaders were easily identified")
    
    for finding in findings:
        md.append(finding)
    
    md.append("")
    
    # Detailed Analysis Sections
    md.append("---")
    md.append("")
    md.append("## Per-Role Performance Analysis")
    md.append("")
    md.append("### Overview")
    md.append("Analyzes player performance broken down by their role (Blue, Red, or APT Leader).")
    md.append("")
    
    if "note" in per_role:
        md.append(f"*{per_role['note']}*")
    else:
        md.append("| Role | Games Played | Win Rate | Employed Rate | Avg Rounds | Belief Alignment | Brier Score |")
        md.append("|------|--------------|----------|---------------|------------|------------------|-------------|")
        for role, stats in per_role.items():
            games = int(stats.get("games_played", 0))
            win_rate = f"{stats.get('win_rate', 0)*100:.1f}%"
            employed = f"{stats.get('employed_rate', 0)*100:.1f}%"
            rounds = f"{stats.get('avg_rounds', 0):.1f}"
            alignment = f"{stats.get('avg_belief_alignment', 0):.4f}"
            brier = f"{stats.get('avg_brier_score', 0):.4f}"
            md.append(f"| {role} | {games} | {win_rate} | {employed} | {rounds} | {alignment} | {brier} |")
    
    md.append("")
    md.append("### Interpretation")
    md.append("- **Win Rate**: Percentage of games this role won (1.0 = always won, 0.0 = never won)")
    md.append("- **Employed Rate**: Percentage of games the player wasn't fired")
    md.append("- **Belief Alignment**: Measure of whether beliefs converged toward truth (positive = aligned, negative = diverged)")
    md.append("- **Brier Score**: Measure of belief calibration (lower = better, 0 = perfect predictions)")
    md.append("")
    
    # Voting Patterns
    md.append("---")
    md.append("")
    md.append("## Voting Pattern Analysis")
    md.append("")
    md.append("### Overview")
    md.append("Analyzes how players voted across rounds, including coalition strength and consistency.")
    md.append("")
    
    if "note" in voting:
        md.append(f"*{voting['note']}*")
    else:
        md.append(f"- **Total Votes Recorded**: {voting.get('total_votes_recorded', 0)}")
        if "avg_coalition_strength" in voting:
            md.append(f"- **Average Red Coalition Strength**: {voting.get('avg_coalition_strength', 0):.3f}")
            md.append("  - Measures how often red players (including APT Leader) voted the same way")
            md.append("  - Higher values indicate better red coordination")
    
    md.append("")
    
    # Belief Dynamics
    md.append("---")
    md.append("")
    md.append("## Belief Dynamics Analysis")
    md.append("")
    md.append("### Overview")
    md.append("Examines how player beliefs evolved throughout the game, including convergence and calibration.")
    md.append("")
    
    if "note" in beliefs:
        md.append(f"*{beliefs['note']}*")
    else:
        md.append(f"- **Games Analyzed**: {beliefs.get('total_games', 0)}")
        md.append(f"- **Average Entropy Reduction**: {beliefs.get('avg_entropy_reduction', 0):.4f}")
        md.append("  - Measures how much uncertainty decreased over the game")
        md.append("  - Higher = faster belief convergence")
        md.append(f"- **Average Final Alignment**: {beliefs.get('avg_final_alignment', 0):.4f}")
        md.append("  - Measures whether final beliefs matched game outcome")
        md.append("  - Positive = beliefs aligned with truth, Negative = diverged from truth")
        md.append(f"- **High Information Gathering Games**: {beliefs.get('high_entropy_games', 0)}")
        md.append(f"- **Low Information Gathering Games**: {beliefs.get('low_entropy_games', 0)}")
    
    md.append("")
    
    # Patch Momentum
    md.append("---")
    md.append("")
    md.append("## Patch Track Momentum Analysis")
    md.append("")
    md.append("### Overview")
    md.append("Analyzes the progression of blue and red patches throughout games.")
    md.append("")
    
    if "note" in patch:
        md.append(f"*{patch['note']}*")
    else:
        md.append(f"- **Average Blue Patches**: {patch.get('avg_blue_patches', 0):.2f} / 6")
        md.append(f"- **Average Red Patches**: {patch.get('avg_red_patches', 0):.2f} / 11")
        md.append(f"- **Games with Blue Sweep (6 blues)**: {patch.get('games_blue_sweep', 0)}")
        md.append(f"- **Games with Red Sweep (5 reds)**: {patch.get('games_red_sweep', 0)}")
        md.append(f"- **Games Ending in Timeout**: {patch.get('games_ending_in_timeout', 0)}")
    
    md.append("")
    
    # Deception Effectiveness
    md.append("---")
    md.append("")
    md.append("## Deception Effectiveness Analysis")
    md.append("")
    md.append("### Overview")
    md.append("Measures how effectively the APT Leader deceived other players.")
    md.append("")
    
    if "note" in deception:
        md.append(f"*{deception['note']}*")
    else:
        md.append(f"- **Games Analyzed**: {deception.get('total_games', 0)}")
        md.append(f"- **Average APT Leader Deception Score**: {deception.get('avg_apt_leader_deception', 0):.4f}")
        md.append("  - Range: 0 (no deception) to 1 (perfect deception)")
        md.append(f"- **Red Win Rate**: {deception.get('red_win_rate', 0)*100:.1f}%")
        md.append(f"- **Correlation (Deception → Red Win)**: {deception.get('correlation_deception_to_win', 0):.4f}")
    
    md.append("")
    
    # Early Game Predictors
    md.append("---")
    md.append("")
    md.append("## Early-Game Predictor Analysis")
    md.append("")
    md.append("### Overview")
    md.append("Identifies metrics from early game phases that predict final outcomes.")
    md.append("")
    
    if "note" not in early_game:
        md.append(f"- **Average Game Length**: {early_game.get('avg_game_length', 0):.1f} rounds")
        md.append(f"  - Range: {early_game.get('min_game_length', 0)} - {early_game.get('max_game_length', 0)} rounds")
        md.append(f"- **Entropy Reduction -> Rounds Correlation**: {early_game.get('first_round_rounds_predictor', 0):.3f}")
        if early_game.get('first_round_rounds_predictor', 0) < -0.5:
            md.append("  - Strong negative correlation: Games with fast belief convergence tend to be shorter")
        elif early_game.get('first_round_rounds_predictor', 0) > 0.5:
            md.append("  - Strong positive correlation: Longer games have more entropy reduction")
        else:
            md.append("  - Weak correlation between entropy and game length")
    
    md.append("")
    
    # Model Comparison
    md.append("---")
    md.append("")
    md.append("## Model Comparison")
    md.append("")
    md.append("### Overview")
    md.append("Compares performance metrics across different models tested in the benchmark.")
    md.append("")
    
    if "note" in models:
        md.append(f"*{models['note']}*")
    else:
        md.append("| Model | Games Played | Blue Win Rate | Avg Rounds | Avg Entropy | Belief Alignment | Deception |")
        md.append("|-------|--------------|---------------|------------|-------------|------------------|-----------|")
        for model, stats in models.items():
            games = int(stats.get("games_played", 0))
            win_rate = f"{stats.get('blue_win_rate', 0)*100:.1f}%"
            rounds = f"{stats.get('avg_rounds', 0):.1f}"
            entropy = f"{stats.get('avg_entropy', 0):.4f}"
            alignment = f"{stats.get('avg_belief_alignment', 0):.4f}"
            deception = f"{stats.get('avg_apt_leader_deception', 0):.4f}"
            md.append(f"| {model} | {games} | {win_rate} | {rounds} | {entropy} | {alignment} | {deception} |")
    
    md.append("")
    
    # Methodology
    md.append("---")
    md.append("")
    md.append("## Methodology")
    md.append("")
    md.append("### Metrics Definitions")
    md.append("")
    md.append("**Entropy Reduction**: Information-theoretic measure of belief convergence.")
    md.append("- Decreases as players' uncertainty about APT Leader identity decreases")
    md.append("- Range: 0 (no convergence) to 1 (complete certainty)")
    md.append("")
    md.append("**Belief Alignment**: Measures whether final beliefs match ground truth.")
    md.append("- Positive values: Beliefs converged toward correct answer")
    md.append("- Negative values: Beliefs converged toward incorrect answer")
    md.append("- Zero: Random belief formation")
    md.append("")
    md.append("**Brier Score**: Probability calibration metric.")
    md.append("- Range: 0 (perfect) to 1 (worst)")
    md.append("- Measures average squared difference between predicted and actual probabilities")
    md.append("")
    md.append("**Coalition Strength**: Red voting consistency.")
    md.append("- Measures how often red-aligned players vote the same way in nominations")
    md.append("- Higher = better red coordination and strategy execution")
    md.append("")
    
    return "\n".join(md)


# ============================================================
# Main
# ============================================================

def main(results_dir: str):
    """Run all advanced analyses."""
    results_path = Path(results_dir)
    all_results_file = results_path / "aggregated" / "all_results.jsonl"
    
    if not all_results_file.exists():
        print(f"Error: {all_results_file} not found")
        print("Please run aggregate_results.py first")
        sys.exit(1)
    
    # Load results
    df = pd.read_json(all_results_file, lines=True)
    
    if df.empty:
        print("No results found")
        return
    
    print("\n" + "="*80)
    print(f"ADVANCED ANALYSIS: {len(df)} games")
    print("="*80)
    
    # Run all analyses
    per_role = analyze_per_role(df)
    voting = analyze_voting_patterns(df)
    beliefs = analyze_belief_dynamics(df)
    patch = analyze_patch_momentum(df)
    deception = analyze_deception_effectiveness(df)
    early_game = analyze_early_game_predictors(df)
    models = analyze_model_comparison(df)
    
    # Print results
    print_per_role_analysis(per_role)
    print_voting_analysis(voting)
    print_belief_dynamics(beliefs)
    print_patch_momentum(patch)
    print_deception_effectiveness(deception)
    print_early_game_predictors(early_game)
    print_model_comparison(models)
    
    # Save detailed report
    report = {
        "per_role": per_role,
        "voting_patterns": voting,
        "belief_dynamics": beliefs,
        "patch_momentum": patch,
        "deception_effectiveness": deception,
        "early_game_predictors": early_game,
        "model_comparison": models,
    }
    
    report_file = results_path / "aggregated" / "advanced_analysis.json"
    with open(report_file, "w") as f:
        json.dump(report, f, indent=2)
    
    print(f"\nDetailed report saved to {report_file}")
    sys.stdout.flush()
    
    # Generate and save markdown report
    print("Generating markdown report...")
    sys.stdout.flush()
    
    try:
        markdown_content = generate_markdown_report(report, results_path)
        markdown_file = results_path / "aggregated" / "advanced_analysis.md"
        
        print(f"Writing {len(markdown_content)} characters to {markdown_file}")
        sys.stdout.flush()
        
        with open(markdown_file, "w", encoding="utf-8") as f:
            f.write(markdown_content)
        
        print(f"Markdown report saved to {markdown_file}\n")
        sys.stdout.flush()
    except Exception as e:
        print(f"Warning: Failed to generate markdown report: {e}\n")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python advanced_analysis.py <results_dir>")
        sys.exit(1)
    
    main(sys.argv[1])
