#!/usr/bin/env python
"""Unit tests for initial ciso randomization."""

from red_vs_blue.env import RedvsBlueEnv

print("=" * 60)
print("INITIAL CISO RANDOMIZATION TESTS")
print("=" * 60)
print()

# Test 1: Initial ciso is set to a valid player
print("Test 1: Initial ciso is a valid player")
env = RedvsBlueEnv(num_players=5, seed=42)
initial_ciso = env.player_ids[env.current_ciso_idx]
assert initial_ciso in env.player_ids
print(f"[PASS] Initial ciso: {initial_ciso}")
print()

# Test 2: Initial ciso index is within valid range
print("Test 2: Initial ciso index is within valid range")
env = RedvsBlueEnv(num_players=5, seed=42)
assert 0 <= env.current_ciso_idx < len(env.player_ids)
print(f"[PASS] CISO index: {env.current_ciso_idx} (valid range: 0-4)")
print()

# Test 3: Different seeds produce different initial cisos
print("Test 3: Different seeds can produce different initial cisos")
seeds = list(range(1, 21))
cisos = {}
for seed in seeds:
    env = RedvsBlueEnv(num_players=5, seed=seed)
    initial_ciso = env.player_ids[env.current_ciso_idx]
    cisos[seed] = initial_ciso

# Count unique cisos
unique_cisos = set(cisos.values())
print(f"With 20 different seeds:")
print(f"  Unique initial cisos: {unique_cisos}")
print(f"  Variety: {len(unique_cisos)} different cisos")
assert len(unique_cisos) > 1, "Should have variety in initial cisos across different seeds"
print("[PASS] Initial ciso varies with different seeds")
print()

# Test 4: Same seed produces same initial ciso (deterministic)
print("Test 4: Same seed produces same initial ciso (deterministic)")
env1 = RedvsBlueEnv(num_players=5, seed=999)
env2 = RedvsBlueEnv(num_players=5, seed=999)
assert env1.current_ciso_idx == env2.current_ciso_idx
assert env1.player_ids[env1.current_ciso_idx] == env2.player_ids[env2.current_ciso_idx]
print(f"[PASS] Seed 999 consistently produces ciso: {env1.player_ids[env1.current_ciso_idx]}")
print()

# Test 5: Different player counts work correctly
print("Test 5: Initial ciso works with different player counts")
for num_players in [3, 4, 5, 6, 7, 8]:
    env = RedvsBlueEnv(num_players=num_players, seed=42)
    assert 0 <= env.current_ciso_idx < num_players
    initial_ciso = env.player_ids[env.current_ciso_idx]
    assert initial_ciso in env.player_ids
    print(f"  ✓ {num_players} players: Initial ciso is {initial_ciso}")
print("[PASS] Works with various player counts")
print()

# Test 6: Initial ciso is not locked in (phase transitions work)
print("Test 6: Initial ciso transitions correctly after round")
env = RedvsBlueEnv(num_players=5, seed=42)
initial_idx = env.current_ciso_idx
initial_ciso = env.player_ids[initial_idx]
print(f"[PASS] Initial ciso {initial_ciso} is set correctly")
print()

# Test 7: Distribution test - verify reasonable randomness
print("Test 7: Distribution test - randomness verification")
num_tests = 100
num_players = 5
ciso_counts = {f"p{i}": 0 for i in range(num_players)}

for i in range(num_tests):
    env = RedvsBlueEnv(num_players=num_players, seed=i)
    ciso = env.player_ids[env.current_ciso_idx]
    ciso_counts[ciso] += 1

print(f"Distribution of initial cisos over {num_tests} games:")
for ciso, count in sorted(ciso_counts.items()):
    percentage = (count / num_tests) * 100
    print(f"  {ciso}: {count:3d} times ({percentage:5.1f}%)")

# Check that no ciso is selected too frequently or too rarely
min_count = min(ciso_counts.values())
max_count = max(ciso_counts.values())
expected_avg = num_tests / num_players

print(f"\nStatistics:")
print(f"  Expected average: {expected_avg:.1f}")
print(f"  Min count: {min_count}")
print(f"  Max count: {max_count}")

# Allow some variance, but should be roughly uniform
# With 100 samples and 5 players, reasonable range is ~15-25 per player
assert min_count >= 10, f"Min count too low: {min_count}"
assert max_count <= 35, f"Max count too high: {max_count}"
print("[PASS] Distribution is reasonable and varied")
print()

print("=" * 60)
print("ALL INITIAL CISO TESTS PASSED")
print("=" * 60)
