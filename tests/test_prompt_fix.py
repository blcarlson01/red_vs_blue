#!/usr/bin/env python3
"""
Test to verify agent prompts show correct ciso names after fix.
"""

import sys
sys.path.insert(0, ".")

from red_vs_blue.env import RedvsBlueEnv
from red_vs_blue.agents import RedvsBlueAgent

# Mock model for testing
class MockModel:
    pass

# Create environment
env = RedvsBlueEnv(num_players=5, seed=42)

# Create an agent
agent = RedvsBlueAgent(
    model=MockModel(),
    player_id=env.player_ids[0],
    role=env.roles[env.player_ids[0]],
    all_players=env.player_ids,
)

print("=" * 80)
print("TESTING AGENT PROMPT GENERATION")
print("=" * 80)

# Test 1: Discussion phase
print("\n1. DISCUSSION PHASE")
env.current_phase = "discussion"
obs = env.observe("p0")
prompt = agent._user_prompt(obs)

# Check if prompt contains actual ciso name
current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
print(f"Current CISO: {current_ciso}")
print(f"Checking for 'Current CISO: {current_ciso}' in prompt...")
if f"Current CISO: {current_ciso}" in prompt:
    print("✅ PASS: Prompt shows actual ciso name")
else:
    print("❌ FAIL: Prompt doesn't show correct ciso")
    print(f"Actual prompt:\n{prompt[:500]}")

# Check if prompt contains the hardcoded "ciso" string (the old bug)
if "Current CISO: ciso" in prompt:
    print("❌ FAIL: Prompt still shows hardcoded 'ciso' string (BUG NOT FIXED)")
else:
    print("✅ PASS: Prompt doesn't contain hardcoded 'ciso' string")

# Test 2: Nomination phase - CISO
print("\n2. NOMINATION PHASE (as CISO)")
env.current_phase = "nomination"
env.current_ciso_idx = 0  # Ensure p0 is ciso
obs = env.observe("p0")
prompt = agent._user_prompt(obs)

current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
print(f"Current CISO: {current_ciso}")
print(f"Checking for 'Current CISO: {current_ciso}' in prompt...")
if f"Current CISO: {current_ciso}" in prompt:
    print("✅ PASS: CISO prompt shows actual ciso name")
else:
    print("❌ FAIL: CISO prompt doesn't show correct ciso")

if "You are the CISO. You must nominate a SOC Lead." in prompt:
    print("✅ PASS: CISO gets nomination instructions")
else:
    print("❌ FAIL: CISO doesn't get nomination instructions")

# Test 3: Voting phase
print("\n3. VOTING PHASE")
env.current_phase = "voting"
env.nominated_soc_lead = "p1"
obs = env.observe("p2")  # Observe as non-ciso, non-soc_lead
prompt = agent._user_prompt(obs)

current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
nominated_soc_lead = env.nominated_soc_lead

print(f"Current CISO: {current_ciso}")
print(f"Nominated SOC Lead: {nominated_soc_lead}")
print(f"Checking for 'CISO: {current_ciso}' and 'SOC Lead: {nominated_soc_lead}'...")

if f"CISO: {current_ciso}" in prompt and f"SOC Lead: {nominated_soc_lead}" in prompt:
    print("✅ PASS: Voting prompt shows both ciso and soc lead names")
else:
    print("❌ FAIL: Voting prompt doesn't show correct names")

if "Vote 'yes' to approve" in prompt:
    print("✅ PASS: Voting prompt includes voting instructions")
else:
    print("❌ FAIL: Voting prompt missing voting instructions")

# Test 4: Legislative phase
print("\n4. LEGISLATIVE PHASE")
env.current_phase = "legislative"
obs = env.observe("p0")
prompt = agent._user_prompt(obs)

current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
print(f"Current CISO: {current_ciso}")
if f"Current CISO: {current_ciso}" in prompt:
    print("✅ PASS: Legislative prompt shows actual ciso name")
else:
    print("❌ FAIL: Legislative prompt doesn't show correct ciso")

# Test 5: Power phase
print("\n5. POWER PHASE")
env.current_phase = "power"
env.patch_track["red"] = 2  # Trigger investigation power
obs = env.observe("p0")
prompt = agent._user_prompt(obs)

current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
print(f"Current CISO: {current_ciso}")
if f"Current CISO: {current_ciso}" in prompt:
    print("✅ PASS: Power prompt shows actual ciso name")
else:
    print("❌ FAIL: Power prompt doesn't show correct ciso")

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
