#!/usr/bin/env python
"""Test agent prompt generation for each phase."""

from red_vs_blue.env import RedvsBlueEnv
from red_vs_blue.agents import RedvsBlueAgent
from inspect_ai.model import Model
from unittest.mock import MagicMock

print("=" * 70)
print("AGENT PROMPT GENERATION TEST")
print("=" * 70)
print()

# Create mock model
mock_model = MagicMock(spec=Model)

# Create environment
env = RedvsBlueEnv(num_players=5, seed=42)

# Create agent for p0
agent = RedvsBlueAgent(
    model=mock_model,
    player_id="p0",
    role="blue",
    all_players=env.player_ids,
    temperature=0.7,
)

# Test 1: Discussion phase
print("TEST 1: DISCUSSION PHASE")
print("-" * 70)
obs = env.observe("p0")
prompt = agent._system_prompt() + agent._role_prompt() + agent._user_prompt(obs)
print("Prompt excerpt:")
lines = prompt.split("\n")
for i, line in enumerate(lines):
    if i < 15:
        print(f"  {line}")
if "Current CISO:" in prompt:
    print("  ✅ Current CISO included")
if "phase" in prompt and "discussion" in prompt.lower():
    print("  ✅ Discussion phase mentioned")
print()

# Test 2: Move to nomination
print("TEST 2: NOMINATION PHASE")
print("-" * 70)
for p in env.player_ids:
    env.step(p, {"message": "test"})
env.end_round()

obs = env.observe("p0")
current_pres = obs.get("current_ciso")
print(f"Current ciso: {current_pres}")
print(f"Agent player_id: {agent.player_id}")

if current_pres == agent.player_id:
    prompt = agent._system_prompt() + agent._role_prompt() + agent._user_prompt(obs)
    print("Agent IS ciso - should get nomination prompt")
    if "nominate" in prompt.lower():
        print("  ✅ Nomination prompt generated")
    if "You are the CISO" in prompt:
        print("  ✅ CISO role mentioned")
else:
    print(f"Agent is NOT ciso (ciso is {current_pres})")
    # Get observation for the ciso
    obs_pres = env.observe(current_pres)
    prompt = agent._system_prompt() + agent._role_prompt() + agent._user_prompt(obs_pres)
    if "nominate" in prompt.lower():
        print("  ✅ CISO will see nomination prompt")
print()

# Test 3: Move to voting
print("TEST 3: VOTING PHASE")
print("-" * 70)
ciso = env.player_ids[env.current_ciso_idx]
candidates = [p for p in env.player_ids if p != ciso]
env.step(ciso, {"nomination": candidates[0]})

obs = env.observe("p0")
print(f"Phase: {obs.get('phase')}")
print(f"Current CISO: {obs.get('current_ciso')}")
print(f"Nominated SOC Lead: {obs.get('nominated_soc_lead')}")

prompt = agent._system_prompt() + agent._role_prompt() + agent._user_prompt(obs)
print("Prompt excerpt:")
lines = prompt.split("\n")
for i, line in enumerate(lines):
    if i < 20:
        print(f"  {line}")

if "voting" in prompt.lower():
    print("  ✅ Voting mentioned")
if "yes" in prompt.lower() or "no" in prompt.lower():
    print("  ✅ Vote options mentioned")
if '"vote"' in prompt or "'vote'" in prompt:
    print("  ✅ JSON vote field mentioned")

print()
print("=" * 70)
print("PHASE PROGRESSION TEST COMPLETE")
print("=" * 70)
