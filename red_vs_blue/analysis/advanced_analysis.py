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
                    "survival_rate": 0.0,
                    "avg_rounds": 0.0,
                    "belief_alignment": [],
                    "brier_score": [],
                }
            
            results[role]["games_played"] += 1
            
            # Win if role matches outcome
            if (role == "blue" and blues_win) or (role == "red" and not blues_win):
                results[role]["wins"] += 1
            
            # Survival (not in fired_players)
            if player_id not in fired_players:
                results[role]["survival_rate"] += 1
            
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
            stats["survival_rate"] /= games
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
        voting_history = row.get("voting_history", {})
        roles = row.get("roles", {})
        belief_histories = row.get("belief_histories", {})
        
        if not voting_history or not roles:
            continue
        
        results["total_votes_recorded"] += len(voting_history)
        
        # Analyze vote patterns per player
        for player_id, vote_record in voting_history.items():
            if "votes" not in vote_record:
                continue
            
            votes = vote_record["votes"]
            role = roles.get(str(player_id), "unknown")
            
            # If red, check coalition (voting with other reds)
            if role == "red":
                red_votes = [v for pid, v in votes.items() if roles.get(str(pid)) == "red"]
                if len(red_votes) > 1:
                    # Check if votes align (all yes or all no)
                    consistency = len([v for v in red_votes if v == red_votes[0]]) / len(red_votes)
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
    print(f"Correlation (deception → red win): {analysis['correlation_deception_to_win']:.3f}")
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
    
    print(f"\nDetailed report saved to {report_file}\n")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python advanced_analysis.py <results_dir>")
        sys.exit(1)
    
    main(sys.argv[1])
