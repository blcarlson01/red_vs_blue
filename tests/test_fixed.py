from red_vs_blue.env import RedvsBlueEnv

# Test 1: Red win by ensuring we enact red
print('=== Test 1: Red Win at 6 ===')
env = RedvsBlueEnv(num_players=5, seed=42)
env.patch_track = {'blue': 0, 'red': 5}
env.current_phase = 'legislative'
# Set drawn cards so that no matter what discards happen, red is applied
env.drawn_cards = ['red', 'red', 'red']

print(f'Before: red={env.patch_track["red"]}, done={env.done}')
env._execute_legislative_session(0)  # CISO discards first red
print(f'After: red={env.patch_track["red"]}, done={env.done}')
assert env.done, "Game should be done when red reaches 6"
assert env.reds_win(), "Reds should have won"
print('[PASS] Red Win at 6')

# Test 2: Blue win
print('\n=== Test 2: Blue Win at 6 ===')
env2 = RedvsBlueEnv(num_players=5, seed=42)
env2.patch_track = {'blue': 5, 'red': 0}
env2.current_phase = 'legislative'
# All blue cards to ensure blue is applied
env2.drawn_cards = ['blue', 'blue', 'blue']

print(f'Before: blue={env2.patch_track["blue"]}, done={env2.done}')
env2._execute_legislative_session(0)
print(f'After: blue={env2.patch_track["blue"]}, done={env2.done}')
assert env2.done, "Game should be done when blue reaches 6"
assert env2.blues_win(), "Blues should have won"
print('[PASS] Blue Win at 6')

# Test 3: APT Leader with 3+ red patches
print('\n=== Test 3: APT Leader Red Win ===')
env3 = RedvsBlueEnv(num_players=5, seed=42)
env3.patch_track['red'] = 3
env3.current_phase = 'voting'
env3.nominated_soc_lead = env3.true_apt_leader_id

print(f'APT Leader: {env3.true_apt_leader_id}')
print(f'Nominated: {env3.nominated_soc_lead}')
print(f'Before: done={env3.done}')

# Vote to approve
votes = {'p0': 'yes', 'p1': 'yes', 'p2': 'yes', 'p3': 'no', 'p4': 'no'}
for pid, vote in votes.items():
    env3.step(pid, {'vote': vote})

# End round to process votes
env3.end_round()

print(f'After: done={env3.done}')
assert env3.done, "Game should be done when APT Leader becomes SOC Lead with 3+ red patches"
assert env3.reds_win(), "Reds should have won"
print('[PASS] APT Leader Red Win')

print('\n=== All tests passed! ===')
