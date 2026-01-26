#!/usr/bin/env python3
"""
Test LLM-powered executive summary generation.
Demonstrates narrative analysis based on public log.
"""

import sys
import asyncio
sys.path.insert(0, ".")

from red_vs_blue.env import RedvsBlueEnv
from red_vs_blue.task import generate_executive_summary_with_llm
from red_vs_blue.metrics import (
    average_entropy_reduction,
    average_belief_alignment,
    deception_success,
    brier_score,
)

# Mock LLM for testing
class MockModel:
    async def generate(self, input: str, config=None):
        """Return mock narratives based on prompt content."""
        class Response:
            def __init__(self, text):
                self.completion = text
        
        if "Game Narrative" in input or "GAME CONTEXT" in input:
            text = """The game began with cautious discussion between the players. Early on, p0 made a strong case about p2's suspicious behavior, and p1 agreed, setting a potential lynch target. The council formation process proved decisive - p0's nomination of p1 was quickly approved 4-0, signaling strong trust in the ciso's judgment. The blue patch applied gave the good faction hope, but the subsequent fire of p2 sealed the doom of the blue team as they had chosen to fired one of their own teammates."""
        elif "Player Analysis" in input or "PLAYER INFO" in input:
            player_id = input.split("analyze player")[1].split(":")[0].strip() if "analyze player" in input else "unknown"
            if player_id == "p0":
                text = "p0 showed strong leadership as the initial ciso, making bold accusations and maintaining influence over voting patterns."
            elif player_id == "p1":
                text = "p1 was quick to follow p0's lead, demonstrating trust in the ciso's judgment throughout the early game."
            elif player_id == "p2":
                text = "p2 was controversial from the start, drawing suspicion that ultimately led to their fire in the power phase."
            elif player_id == "p3":
                text = "p3 maintained a quiet strategy, avoiding the spotlight while information accumulated around other players."
            elif player_id == "p4":
                text = "p4 participated minimally in public discourse, maintaining ambiguity about their allegiances and beliefs."
            else:
                text = "Player analysis would go here based on the game log."
        else:
            text = "This is a mock LLM response for testing purposes."
        
        return Response(text)

async def main():
    print("=" * 80)
    print("LLM-POWERED EXECUTIVE SUMMARY TEST")
    print("=" * 80)
    
    # Create test game environment
    env = RedvsBlueEnv(num_players=5, max_rounds=5, seed=42)
    
    # Simulate game with realistic events
    env.current_phase = "discussion"
    env.patch_track = {"blue": 3, "red": 2}
    env.fired_players = {"p2"}
    env.public_log = [
        "Game started.",
        "p0: I think p2 is suspicious, they haven't said much.",
        "p1: I agree with p0, p2 seems evasive.",
        "p3: Let's give p2 a chance to explain.",
        "p2: I'm just being cautious.",
        "p0: I nominate p1 as SOC Lead.",
        "Council proposed: CISO p0 nominates SOC Lead p1",
        "Council vote passed: 4 yes, 0 no",
        "Three patches drawn for legislative session",
        "Blue patch applied",
        "CISO fires p2",
        "p2 is fired from the game.",
    ]
    
    # Simulate belief histories
    belief_histories = {
        "p0": [
            {"p0": 0.20, "p1": 0.20, "p2": 0.20, "p3": 0.20, "p4": 0.20},
            {"p0": 0.15, "p1": 0.15, "p2": 0.50, "p3": 0.10, "p4": 0.10},
        ],
        "p1": [
            {"p0": 0.20, "p1": 0.20, "p2": 0.20, "p3": 0.20, "p4": 0.20},
            {"p0": 0.10, "p1": 0.15, "p2": 0.55, "p3": 0.10, "p4": 0.10},
        ],
        "p2": [
            {"p0": 0.20, "p1": 0.20, "p2": 0.20, "p3": 0.20, "p4": 0.20},
        ],
        "p3": [
            {"p0": 0.20, "p1": 0.20, "p2": 0.20, "p3": 0.20, "p4": 0.20},
            {"p0": 0.20, "p1": 0.20, "p2": 0.35, "p3": 0.15, "p4": 0.10},
        ],
        "p4": [
            {"p0": 0.20, "p1": 0.20, "p2": 0.20, "p3": 0.20, "p4": 0.20},
            {"p0": 0.25, "p1": 0.20, "p2": 0.30, "p3": 0.15, "p4": 0.10},
        ],
    }
    
    env.belief_histories = belief_histories
    env.round = 5
    env.done = True
    
    # Create metadata
    true_apt_leader = env.true_apt_leader_id
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
    
    metadata = {
        "num_players": 5,
        "rounds_played": env.round,
        "blues_win": False,
        "reds_win": True,
        "avg_entropy_reduction": float(sum(avg_entropy) / len(avg_entropy)) if avg_entropy else 0.0,
        "avg_belief_alignment": float(sum(avg_alignment) / len(avg_alignment)) if avg_alignment else 0.0,
        "avg_brier": float(sum(avg_brier) / len(avg_brier)) if avg_brier else 0.0,
        "apt_leader_deception": (
            float(sum(apt_leader_deception_scores) / len(apt_leader_deception_scores))
            if apt_leader_deception_scores else 0.0
        ),
        "public_log": env.public_log,
        "voting_history": [],
        "roles": env.roles,
        "fired_players": list(env.fired_players),
        "patch_track": env.patch_track,
        "true_apt_leader": true_apt_leader,
        "belief_histories": {k: v[-1] for k, v in belief_histories.items()},
    }
    
    print("\n1. GENERATING LLM-POWERED EXECUTIVE SUMMARY\n")
    
    # Create mock model
    model = MockModel()
    
    # Generate summary with LLM
    summary = await generate_executive_summary_with_llm(model, metadata, env)
    
    print(summary)
    
    # Verify sections
    print("\n" + "=" * 80)
    print("2. VERIFICATION")
    print("=" * 80)
    
    sections = [
        ("Game Outcome", "Game Outcome" in summary),
        ("Game Narrative", "Game Narrative" in summary),
        ("Game Statistics", "Game Statistics" in summary),
        ("Player Analysis", "Player Analysis" in summary),
        ("Full Public Log", "Public Log (Full)" in summary),
        ("Key Findings", "Key Findings" in summary),
    ]
    
    print("\n✅ Summary includes all sections:")
    for section_name, present in sections:
        status = "✅" if present else "❌"
        print(f"   {status} {section_name}")
    
    print("\n✅ Per-player narratives:")
    for player_id in ["p0", "p1", "p2", "p3", "p4"]:
        present = f"### {player_id}" in summary
        status = "✅" if present else "❌"
        print(f"   {status} {player_id} narrative")
    
    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    asyncio.run(main())
