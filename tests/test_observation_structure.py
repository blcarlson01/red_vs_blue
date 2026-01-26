#!/usr/bin/env python
"""Test to verify observation structure includes current_ciso."""

from red_vs_blue.env import RedvsBlueEnv

print("=" * 60)
print("OBSERVATION STRUCTURE TEST")
print("=" * 60)
print()

env = RedvsBlueEnv(num_players=5, seed=42)

# Move through a few phases to test observation
print("Phase: DISCUSSION")
obs = env.observe("p0")
print(f"Observation keys: {list(obs.keys())}")
print(f"current_ciso: {obs.get('current_ciso', 'NOT FOUND')}")
print(f"nominated_soc_lead: {obs.get('nominated_soc_lead', 'NOT FOUND')}")
print(f"phase: {obs.get('phase', 'NOT FOUND')}")
print()

# Move to nomination
for p in env.player_ids:
    env.step(p, {"message": "test"})
env.end_round()

print("Phase: NOMINATION")
obs = env.observe("p0")
print(f"Observation keys: {list(obs.keys())}")
print(f"current_ciso: {obs.get('current_ciso', 'NOT FOUND')}")
print(f"nominated_soc_lead: {obs.get('nominated_soc_lead', 'NOT FOUND')}")
print(f"phase: {obs.get('phase', 'NOT FOUND')}")
print()

# Nominate
ciso = env.player_ids[env.current_ciso_idx]
candidates = [p for p in env.player_ids if p != ciso]
env.step(ciso, {"nomination": candidates[0]})

print("Phase: VOTING (after nomination step)")
obs = env.observe("p0")
print(f"Observation keys: {list(obs.keys())}")
print(f"current_ciso: {obs.get('current_ciso', 'NOT FOUND')}")
print(f"nominated_soc_lead: {obs.get('nominated_soc_lead', 'NOT FOUND')}")
print(f"phase: {obs.get('phase', 'NOT FOUND')}")
print()

# Show what a full prompt would look like
print("=" * 60)
print("SAMPLE PROMPT CONSTRUCTION:")
print("=" * 60)
print()

phase = obs.get("phase", "discussion")
current_ciso = obs.get("current_ciso", "unknown")
nominated_soc_lead = obs.get("nominated_soc_lead")
your_role = obs.get("your_role", "unknown")

print(f"Round {obs['round']}, Phase: {phase}.")
print(f"Current CISO: {current_ciso}")
print(f"Your Role: {your_role}")
print()
if phase == "voting":
    print(f"A council has been nominated:")
    print(f"CISO: {current_ciso}, SOC Lead: {nominated_soc_lead}")
