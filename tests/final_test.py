#!/usr/bin/env python
"""Final comprehensive test suite."""

import sys
from red_vs_blue.env import RedvsBlueEnv

tests_passed = 0
tests_failed = 0

def test(name, condition, message=""):
    global tests_passed, tests_failed
    if condition:
        print(f"[PASS] {name}")
        tests_passed += 1
    else:
        print(f"[FAIL] {name}: {message}")
        tests_failed += 1

print("=" * 60)
print("COMPREHENSIVE TEST SUITE")
print("=" * 60)
print()

# Test 1: Initialization
print("Test 1: Environment Initialization")
env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
test("1a", len(env.player_ids) == 5, "Wrong player count")
test("1b", env.round == 0, "Round should start at 0")
test("1c", not env.done, "Game should not be done at start")
test("1d", env.current_phase == "discussion", f"Expected 'discussion', got '{env.current_phase}'")
print()

# Test 2: Patch Deck
print("Test 2: Patch Deck")
env = RedvsBlueEnv(num_players=5, seed=42)
test("2a", len(env.patch_deck) == 17, f"Wrong deck size: {len(env.patch_deck)}")
test("2b", env.patch_deck.count("blue") == 6, f"Wrong blue count: {env.patch_deck.count('blue')}")
test("2c", env.patch_deck.count("red") == 11, f"Wrong red count: {env.patch_deck.count('red')}")
print()

# Test 3: Role Assignment
print("Test 3: Role Assignment")
env = RedvsBlueEnv(num_players=5, seed=42)
roles = list(env.roles.values())
test("3a", roles.count("blue") == 3, f"Wrong blue count: {roles.count('blue')}")
test("3b", roles.count("red") == 1, f"Wrong red count: {roles.count('red')}")
test("3c", roles.count("apt_leader") == 1, f"Wrong apt_leader count: {roles.count('apt_leader')}")
test("3d", env.true_apt_leader_id in env.player_ids, "APT Leader not in player list")
print()

# Test 4: Win Conditions
print("Test 4: Win Conditions")

# Blue win
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track = {'blue': 5, 'red': 0}
env.current_phase = 'legislative'
env.drawn_cards = ['blue', 'blue', 'blue']
env._execute_legislative_session(0)
test("4a", env.done and env.blues_win(), "Blue win at 6 not working")

# Red win
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track = {'blue': 0, 'red': 5}
env.current_phase = 'legislative'
env.drawn_cards = ['red', 'red', 'red']
env._execute_legislative_session(0)
test("4b", env.done and env.reds_win(), "Red win at 6 not working")

# APT Leader
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track['red'] = 3
env.current_phase = 'voting'
env.nominated_soc_lead = env.true_apt_leader_id
for pid in env.player_ids[:3]:
    env.step(pid, {'vote': 'yes'})
for pid in env.player_ids[3:]:
    env.step(pid, {'vote': 'no'})
env.end_round()
test("4c", env.done and env.reds_win(), "APT Leader win not working")

print()

# Test 5: Powers
print("Test 5: CISO Powers")

# Special election
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track['red'] = 3
env.current_phase = 'power'
env.current_ciso_idx = 0
env.step('p0', {'power_action': {'special_election': 'p2'}})
test("5a", env.current_ciso_idx == 2, f"Special election failed: idx={env.current_ciso_idx}")

# Investigation
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track['red'] = 2
env.current_phase = 'power'
env.current_ciso_idx = 0
env.step('p0', {'power_action': {'investigate': 'p1'}})
test("5b", 'p1' in env.investigation_results, "Investigation not recorded")

# Fire
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track['red'] = 4
env.current_phase = 'power'
env.current_ciso_idx = 0
target = [p for p in env.player_ids if p != 'p0'][0]
env.step('p0', {'power_action': {'fire': target}})
test("5c", target in env.fired_players, "Fire not recorded")

print()

# Test 6: Phase Transitions
print("Test 6: Phase Transitions")

env = RedvsBlueEnv(num_players=5, seed=42)

# Discussion -> Nomination
test("6a", env.current_phase == "discussion", "Start phase wrong")
env.end_round()
test("6b", env.current_phase == "nomination", "Discussion->Nomination failed")

# Nomination -> Voting
ciso = env.player_ids[env.current_ciso_idx]
env.step(ciso, {'nomination': 'p1'})
test("6c", env.current_phase == "voting", "Nomination->Voting failed")

print()

# Test 7: Game Flow
print("Test 7: Game Flow (Full Round)")
env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=999)

round_count = 0
max_attempts = 50
while not env.done and round_count < max_attempts:
    round_count += 1
    
    if env.current_phase == "discussion":
        for p in env.player_ids:
            env.step(p, {"message": "test"})
        env.end_round()
    elif env.current_phase == "nomination":
        ciso = env.player_ids[env.current_ciso_idx]
        candidates = [p for p in env.player_ids if p != ciso and p not in env.fired_players]
        if candidates:
            env.step(ciso, {'nomination': candidates[0]})
        env.end_round()
    elif env.current_phase == "voting":
        employed = [p for p in env.player_ids if p not in env.fired_players]
        for i, p in enumerate(employed):
            vote = "yes" if i < 3 else "no"
            env.step(p, {"vote": vote})
        env.end_round()
    elif env.current_phase == "legislative":
        ciso = env.player_ids[env.current_ciso_idx]
        env.step(ciso, {'discard_patch': 0})
        env.end_round()
    elif env.current_phase == "power":
        ciso = env.player_ids[env.current_ciso_idx]
        targets = [p for p in env.player_ids if p != ciso and p not in env.fired_players]
        if targets and env.patch_track["red"] >= 4:
            env.step(ciso, {"power_action": {"fire": targets[0]}})
        elif targets and env.patch_track["red"] == 3:
            env.step(ciso, {"power_action": {"special_election": targets[0]}})
        elif targets and env.patch_track["red"] == 2:
            env.step(ciso, {"power_action": {"investigate": targets[0]}})
        env.end_round()
    else:
        break

test("7a", env.done, f"Game should end: round={round_count}, max={max_attempts}")
test("7b", round_count < max_attempts, f"Game took too long: {round_count} rounds")
print()

print("=" * 60)
print(f"RESULTS: {tests_passed} passed, {tests_failed} failed")
print("=" * 60)

sys.exit(0 if tests_failed == 0 else 1)
