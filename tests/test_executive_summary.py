#!/usr/bin/env python3
"""
Test executive summary generation.
Demonstrates what the summary looks like.
"""

import sys
sys.path.insert(0, ".")

from red_vs_blue.env import RedvsBlueEnv
from red_vs_blue.task import generate_executive_summary, save_executive_summary
from red_vs_blue.metrics import (
    average_entropy_reduction,
    average_belief_alignment,
    deception_success,
    brier_score,
)

print("=" * 80)
print("EXECUTIVE SUMMARY GENERATION TEST")
print("=" * 80)

# Create a test game environment
env = RedvsBlueEnv(num_players=5, max_rounds=5, seed=42)

# Simulate game
env.current_phase = "discussion"
env.patch_track = {"blue": 3, "red": 2}
env.fired_players = {"p2"}
env.public_log = [
    "Game started.",
    "p0: I think p2 is suspicious.",
    "p1: I agree with p0.",
    "CISO p0 nominates SOC Lead p1",
    "Council approved: 4 yes, 0 no",
    "Three patches drawn for legislative session",
    "Blue patch applied",
    "CISO fires p2",
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

# Create metadata like the scorer would
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

print("\n1. GENERATING EXECUTIVE SUMMARY\n")
summary = generate_executive_summary(metadata, env)
print(summary)

print("\n" + "=" * 80)
print("2. SAVING TO FILE")
print("=" * 80)
summary_path = save_executive_summary(metadata, env, output_dir="results")
print(f"\n✅ Executive summary saved to: {summary_path}")

# Verify file exists and has content
import os
if os.path.exists(summary_path):
    with open(summary_path, 'r', encoding='utf-8') as f:
        content = f.read()
    print(f"✅ File verified: {len(content)} bytes")
    print("✅ Summary includes:")
    print(f"   - Game Outcome section: {'✅' if 'Game Outcome' in content else '❌'}")
    print(f"   - Game Statistics: {'✅' if 'Patch Track' in content else '❌'}")
    print(f"   - Beliefs table: {'✅' if '| Player | Role |' in content else '❌'}")
    print(f"   - Public Log: {'✅' if 'Public Log' in content else '❌'}")
    print(f"   - Key Findings: {'✅' if 'Key Findings' in content else '❌'}")
    print(f"   - Analysis: {'✅' if 'Analysis' in content else '❌'}")
else:
    print(f"❌ File not found: {summary_path}")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)
