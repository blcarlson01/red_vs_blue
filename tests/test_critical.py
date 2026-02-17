"""Critical gameplay behavior tests."""

from red_vs_blue.env import RedvsBlueEnv


def test_blue_win():
    """Test blue win at 6 patches."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.patch_track = {'blue': 5, 'red': 0}
    env.current_phase = 'legislative'
    env.drawn_cards = ['blue', 'blue', 'red']
    
    print(f'Before: blue={env.patch_track["blue"]}, done={env.done}')
    env._execute_legislative_session(2)
    print(f'After: blue={env.patch_track["blue"]}, red={env.patch_track["red"]}, done={env.done}')
    assert env.done and env.blues_win()


def test_red_win():
    """Test red win at 6 patches."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.patch_track = {'blue': 0, 'red': 5}
    env.current_phase = 'legislative'
    env.drawn_cards = ['red', 'red', 'blue']
    
    print(f'Before: red={env.patch_track["red"]}, done={env.done}')
    env._execute_legislative_session(2)
    print(f'After: red={env.patch_track["red"]}, done={env.done}')
    assert env.done and env.reds_win()


def test_apt_leader_fire():
    """Test apt_leader fire triggers blue win."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.patch_track['red'] = 4
    env.current_phase = 'power'
    env.current_ciso_idx = 0
    apt_leader = env.true_apt_leader_id
    
    print(f'APT Leader: {apt_leader}')
    print(f'Before: done={env.done}')
    env.step('p0', {'power_action': {'fire': apt_leader}})
    print(f'After: done={env.done}, blues_win={env.blues_win()}')
    assert env.done and env.blues_win()


def test_special_election():
    """Test special election power."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.patch_track['red'] = 3
    env.current_phase = 'power'
    env.current_ciso_idx = 0
    
    print(f'Before: ciso_idx={env.current_ciso_idx}')
    env.step('p0', {'power_action': {'special_election': 'p2'}})
    print(f'After: ciso_idx={env.current_ciso_idx}')
    assert env.current_ciso_idx == 2


def test_apt_leader_soc_lead():
    """Test APT Leader as SOC Lead with 3+ patches triggers red win."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.patch_track['red'] = 3
    env.current_phase = 'voting'
    env.nominated_soc_lead = env.true_apt_leader_id
    
    print(f'APT Leader: {env.true_apt_leader_id}')
    print(f'Nominated: {env.nominated_soc_lead}')
    print(f'Before: done={env.done}')
    
    # Vote to approve
    votes = {'p0': 'yes', 'p1': 'yes', 'p2': 'yes', 'p3': 'no', 'p4': 'no'}
    for pid, vote in votes.items():
        env.step(pid, {'vote': vote})
    env.end_round()
    
    print(f'After: done={env.done}')
    assert env.done and env.reds_win()


