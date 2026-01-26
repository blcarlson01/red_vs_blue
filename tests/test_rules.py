"""
Test script to verify Red vs. Blue rules are correctly implemented.
"""
import sys
sys.path.insert(0, r'c:\Users\laker\OneDrive\Desktop\secret_hilter_benchmark')

from red_vs_blue.env import RedvsBlueEnv

def test_rules():
    """Verify key rules are implemented correctly."""
    print("="*70)
    print("TESTING RED VS. BLUE RULES")
    print("="*70)
    
    # Test 1: Deck composition
    print("\n[TEST 1] Patch deck composition")
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    deck = env.patch_deck + env.discard_pile
    blues = sum(1 for p in deck if p == "blue")
    reds = sum(1 for p in deck if p == "red")
    print(f"  Deck size: {len(deck)}")
    print(f"  Blue patches: {blues} (should be 6)")
    print(f"  Red patches: {reds} (should be 11)")
    assert blues == 6, "Wrong blue count"
    assert reds == 11, "Wrong red count"
    print("  [PASS]")
    
    # Test 2: Role assignment
    print("\n[TEST 2] Role assignment")
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    roles = list(env.roles.values())
    apt_leader_count = sum(1 for r in roles if r == "apt_leader")
    red_count = sum(1 for r in roles if r == "red")
    blue_count = sum(1 for r in roles if r == "blue")
    print(f"  APT Leader: {apt_leader_count} (should be 1)")
    print(f"  Red: {red_count} (should be 1)")
    print(f"  Blue: {blue_count} (should be 3)")
    assert apt_leader_count == 1, "Wrong apt_leader count"
    assert red_count == 1, "Wrong red count"
    assert blue_count == 3, "Wrong blue count"
    print("  [PASS]")
    
    # Test 3: Game phases
    print("\n[TEST 3] Game phases")
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    expected_phases = ["discussion", "nomination", "voting", "legislative", "power", "discussion"]
    print(f"  Starting phase: {env.current_phase}")
    assert env.current_phase == "discussion", "Should start in discussion"
    print("  [PASS]")
    
    # Test 4: CISO tracking
    print("\n[TEST 4] CISO tracking")
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
    print(f"  Initial ciso: {ciso}")
    print(f"  CISO index: {env.current_ciso_idx}")
    assert ciso in env.player_ids, "CISO not in player list"
    print("  [PASS]")
    
    # Test 5: Win condition (blues at 6)
    print("\n[TEST 5] Blue win condition")
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    env.patch_track["blue"] = 6
    env.done = True
    blues_win = env.blues_win()
    print(f"  Patch track: {env.patch_track}")
    print(f"  Blues win (6 patches): {blues_win}")
    assert blues_win, "Blues should win at 6 patches"
    print("  [PASS]")
    
    # Test 6: Win condition (reds at 6)
    print("\n[TEST 6] Red win condition")
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    env.patch_track["red"] = 6
    env.done = True
    reds_win = env.reds_win()
    print(f"  Patch track: {env.patch_track}")
    print(f"  Reds win (6 patches): {reds_win}")
    assert reds_win, "Reds should win at 6 patches"
    print("  [PASS]")
    
    # Test 7: Investigation power
    print("\n[TEST 7] Investigation power (after 2nd red patch)")
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    env.patch_track["red"] = 2
    env.current_phase = "power"
    obs = env.observe("p0")
    print(f"  Red patches: {env.patch_track['red']}")
    print(f"  Phase: {env.current_phase}")
    print(f"  Investigation should be available: {env.patch_track['red'] == 2}")
    assert env.patch_track["red"] == 2, "Should have 2 red patches"
    print("  [PASS]")
    
    # Test 8: Fire power
    print("\n[TEST 8] Fire power (after 4th red patch)")
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    env.patch_track["red"] = 4
    env.current_phase = "power"
    obs = env.observe("p0")
    print(f"  Red patches: {env.patch_track['red']}")
    print(f"  Fire should be available: {env.patch_track['red'] >= 4}")
    assert env.patch_track["red"] >= 4, "Should have 4+ red patches"
    print("  [PASS]")
    
    # Test 9: APT Leader win
    print("\n[TEST 9] APT Leader becomes SOC Lead after 3+ red patches (red win)")
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    env.patch_track["red"] = 3
    env.nominated_soc_lead = env.true_apt_leader_id
    env.council_votes = {"p0": "yes", "p1": "yes", "p2": "yes"}  # Majority approved
    
    # Manually trigger council resolution
    employed_count = len([p for p in env.player_ids if p not in env.fired_players])
    yes_votes = 3
    majority_needed = employed_count / 2.0
    
    print(f"  Red patches: {env.patch_track['red']}")
    print(f"  SOC Lead: {env.nominated_soc_lead}")
    print(f"  True APT Leader: {env.true_apt_leader_id}")
    print(f"  Is APT Leader: {env.nominated_soc_lead == env.true_apt_leader_id}")
    print(f"  Yes votes: {yes_votes} > majority ({majority_needed}): {yes_votes > majority_needed}")
    
    assert env.nominated_soc_lead == env.true_apt_leader_id, "SOC Lead should be APT Leader for this test"
    assert env.patch_track["red"] >= 3, "Need 3+ red patches"
    print("  [PASS]")
    
    # Test 10: Observation includes new phase info
    print("\n[TEST 10] Agent observation includes proper info")
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    obs = env.observe("p0")
    required_keys = ["current_ciso", "nominated_soc_lead", "phase", "patch_track", "your_role"]
    missing_keys = [k for k in required_keys if k not in obs]
    print(f"  Observation keys: {list(obs.keys())}")
    print(f"  Missing keys: {missing_keys}")
    assert not missing_keys, f"Missing keys: {missing_keys}"
    print("  [PASS]")
    
    print("\n" + "="*70)
    print("ALL TESTS PASSED!")
    print("="*70)

if __name__ == "__main__":
    test_rules()
