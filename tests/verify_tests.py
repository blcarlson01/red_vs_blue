#!/usr/bin/env python
"""Simple verification tests."""

from red_vs_blue.env import RedvsBlueEnv

print("=" * 60)
print("VERIFICATION TESTS")
print("=" * 60)
print()

# Test 1: Basic initialization
print("Test 1: Initialization")
env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
assert len(env.player_ids) == 5
assert env.current_phase == "discussion"
print("[PASS]")

# Test 2: Patch deck
print("Test 2: Patch Deck")
env = RedvsBlueEnv(num_players=5, seed=42)
assert len(env.patch_deck) == 17
assert env.patch_deck.count("blue") == 6
assert env.patch_deck.count("red") == 11
print("[PASS]")

# Test 3: Roles assignment
print("Test 3: Role Assignment")
env = RedvsBlueEnv(num_players=5, seed=42)
roles_list = list(env.roles.values())
assert roles_list.count("blue") == 3
assert roles_list.count("red") == 1
assert roles_list.count("apt_leader") == 1
print("[PASS]")

# Test 4: Blue win
print("Test 4: Blue Win at 6")
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track = {'blue': 5, 'red': 0}
env.current_phase = 'legislative'
env.drawn_cards = ['blue', 'blue', 'blue']
env._execute_legislative_session(0)
assert env.done and env.blues_win()
print("[PASS]")

# Test 5: Red win
print("Test 5: Red Win at 6")
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track = {'blue': 0, 'red': 5}
env.current_phase = 'legislative'
env.drawn_cards = ['red', 'red', 'red']
env._execute_legislative_session(0)
assert env.done and env.reds_win()
print("[PASS]")

# Test 6: APT Leader as SOC Lead
print("Test 6: APT Leader as SOC Lead")
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track['red'] = 3
env.current_phase = 'voting'
env.nominated_soc_lead = env.true_apt_leader_id
votes = {'p0': 'yes', 'p1': 'yes', 'p2': 'yes', 'p3': 'no', 'p4': 'no'}
for pid, vote in votes.items():
    env.step(pid, {'vote': vote})
env.end_round()
assert env.done and env.reds_win()
print("[PASS]")

# Test 7: Special Election
print("Test 7: Special Election")
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track['red'] = 3
env.current_phase = 'power'
env.current_ciso_idx = 0
env.step('p0', {'power_action': {'special_election': 'p2'}})
assert env.current_ciso_idx == 2
print("[PASS]")

# Test 8: APT Leader Fire (all reds fired)
print("Test 8: APT Leader Fire")
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track['red'] = 4
env.current_phase = 'power'
env.current_ciso_idx = 0
apt_leader = env.true_apt_leader_id
# Just fire apt_leader directly
env.step('p0', {'power_action': {'fire': apt_leader}})
assert env.done and apt_leader in env.fired_players
print("[PASS]")

print()
print("=" * 60)
print("ALL VERIFICATION TESTS PASSED")
print("=" * 60)
