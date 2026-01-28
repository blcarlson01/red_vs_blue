"""
test_patch_visibility.py

Unit tests to verify that CISO and SOC Lead can view patches during the legislative session.

Rules:
- CISO draws 3 patches, sees them all, and discards 1
- SOC Lead receives the remaining 2 patches, sees them both, and discards 1
- The final patch is enacted
"""

import sys
from pathlib import Path

# Add parent directory to path
parent = Path(__file__).resolve().parent.parent
if str(parent) not in sys.path:
    sys.path.insert(0, str(parent))

from red_vs_blue.env import RedvsBlueEnv


def test_ciso_sees_three_patches():
    """Test that CISO can see all 3 patches in observation during legislative phase."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    # Set up legislative_ciso phase
    env.current_phase = "legislative_ciso"
    env.drawn_cards = ["blue", "red", "blue"]
    
    # Get CISO observation
    ciso_id = env.player_ids[env.current_ciso_idx]
    observation = env.observe(ciso_id)
    
    # CISO should see the drawn cards in their observation
    assert "drawn_cards" in observation, "CISO observation missing 'drawn_cards'"
    assert observation["drawn_cards"] == ["blue", "red", "blue"], \
        f"CISO should see all 3 patches, got: {observation.get('drawn_cards')}"
    assert len(observation["drawn_cards"]) == 3, \
        f"CISO should see exactly 3 patches, got {len(observation.get('drawn_cards', []))}"
    
    print("✓ CISO can see all 3 patches")


def test_ciso_sees_patches_in_correct_order():
    """Test that CISO sees patches in the order they were drawn."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    # Set up legislative_ciso phase with specific patch order
    env.current_phase = "legislative_ciso"
    expected_patches = ["red", "blue", "red"]
    env.drawn_cards = expected_patches.copy()
    
    # Get CISO observation
    ciso_id = env.player_ids[env.current_ciso_idx]
    observation = env.observe(ciso_id)
    
    # Verify order is preserved
    assert observation.get("drawn_cards") == expected_patches, \
        f"Patch order mismatch. Expected {expected_patches}, got {observation.get('drawn_cards')}"
    
    print("✓ CISO sees patches in correct order")


def test_non_ciso_cannot_see_patches_during_legislative():
    """Test that non-CISO players cannot see patches during legislative phase."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    # Set up legislative_ciso phase
    env.current_phase = "legislative_ciso"
    env.drawn_cards = ["blue", "red", "blue"]
    
    ciso_id = env.player_ids[env.current_ciso_idx]
    
    # Check observations for non-CISO players
    for player_id in env.player_ids:
        if player_id != ciso_id:
            observation = env.observe(player_id)
            # Non-CISO players should either not have drawn_cards or it should be empty
            assert observation.get("drawn_cards") is None or observation.get("drawn_cards") == [], \
                f"Non-CISO player {player_id} should not see drawn cards during legislative phase"
    
    print("✓ Non-CISO players cannot see patches during legislative phase")


def test_soc_lead_sees_two_remaining_patches():
    """Test that SOC Lead can see the 2 remaining patches after CISO discards."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    # Set up council
    env.nominated_soc_lead = env.player_ids[1]
    
    # Set up legislative_soc phase with 2 remaining cards
    env.current_phase = "legislative_soc"
    env.drawn_cards = ["red", "blue"]  # After CISO discarded one
    
    # SOC Lead should see 2 remaining cards
    soc_lead_id = env.nominated_soc_lead
    observation = env.observe(soc_lead_id)
    
    # SOC Lead should see the remaining 2 cards
    assert "drawn_cards" in observation, "SOC Lead observation missing 'drawn_cards'"
    assert len(observation.get("drawn_cards", [])) == 2, \
        f"SOC Lead should see exactly 2 patches, got {len(observation.get('drawn_cards', []))}"
    assert observation["drawn_cards"] == ["red", "blue"], \
        f"SOC Lead should see remaining patches, got: {observation.get('drawn_cards')}"
    
    print("✓ SOC Lead can see the 2 remaining patches")


def test_full_legislative_session_patch_flow():
    """Test complete flow: 3 patches → CISO sees 3 → discards 1 → SOC Lead sees 2 → discards 1 → 1 enacted."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    # Set up council
    ciso_id = env.player_ids[0]
    soc_lead_id = env.player_ids[1]
    env.current_ciso_idx = 0
    env.nominated_soc_lead = soc_lead_id
    
    # Step 1: CISO phase - Set up legislative_ciso phase with known patches
    env.current_phase = "legislative_ciso"
    initial_patches = ["blue", "red", "blue"]
    env.drawn_cards = initial_patches.copy()
    
    # CISO sees all 3 patches
    ciso_obs = env.observe(ciso_id)
    assert ciso_obs.get("drawn_cards") == initial_patches, \
        f"CISO should see all 3 patches: {initial_patches}"
    
    # Step 2: CISO discards patch at index 1 (the "red" patch)
    ciso_discard_idx = 1
    action = {"discard_patch": ciso_discard_idx}
    env.step(ciso_id, action)
    
    # After CISO discard, should transition to legislative_soc phase
    assert env.current_phase == "legislative_soc", \
        f"Should transition to legislative_soc, but got {env.current_phase}"
    assert len(env.drawn_cards) == 2, \
        f"Should have 2 cards remaining after CISO discard, got {len(env.drawn_cards)}"
    expected_after_ciso = ["blue", "blue"]  # After discarding index 1
    assert env.drawn_cards == expected_after_ciso, \
        f"After CISO discard, should have {expected_after_ciso}, got {env.drawn_cards}"
    
    # Step 3: SOC Lead sees remaining 2 patches
    soc_obs = env.observe(soc_lead_id)
    assert soc_obs.get("drawn_cards") == expected_after_ciso, \
        f"SOC Lead should see {expected_after_ciso}, got {soc_obs.get('drawn_cards')}"
    
    # Step 4: SOC Lead discards 1 patch, final patch is enacted
    soc_discard_idx = 0
    action = {"discard_patch": soc_discard_idx}
    env.step(soc_lead_id, action)
    
    # After SOC Lead discard, 1 blue patch should be enacted
    assert env.patch_track["blue"] == 1, \
        f"Should have 1 blue patch enacted, got {env.patch_track['blue']}"
    assert len(env.drawn_cards) == 0, "Should have no drawn cards remaining"
    
    print("✓ Full legislative session patch flow works correctly")


def test_patch_visibility_only_during_legislative_phase():
    """Test that patches are only visible during legislative phase, not other phases."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    ciso_id = env.player_ids[env.current_ciso_idx]
    
    # Test during different phases
    phases_without_patches = ["discussion", "nomination", "voting", "power"]
    
    for phase in phases_without_patches:
        env.current_phase = phase
        env.drawn_cards = ["blue", "red", "blue"]  # Set cards but wrong phase
        
        observation = env.observe(ciso_id)
        
        # Patches should not be visible outside legislative phases
        assert observation.get("drawn_cards") is None or observation.get("drawn_cards") == [], \
            f"Patches should not be visible during {phase} phase"
    
    print("✓ Patches only visible during legislative phase")


def test_patch_counts_are_correct():
    """Test that patch counts match what players see."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    # Set up legislative_ciso phase
    env.current_phase = "legislative_ciso"
    env.drawn_cards = ["blue", "blue", "red"]
    
    ciso_id = env.player_ids[env.current_ciso_idx]
    observation = env.observe(ciso_id)
    
    # Count patches
    drawn = observation.get("drawn_cards", [])
    blue_count = sum(1 for p in drawn if p == "blue")
    red_count = sum(1 for p in drawn if p == "red")
    
    assert blue_count == 2, f"Expected 2 blue patches, got {blue_count}"
    assert red_count == 1, f"Expected 1 red patch, got {red_count}"
    
    print("✓ Patch counts are correct")


def test_ciso_decision_based_on_visible_patches():
    """Test that CISO can make informed decision based on visible patches."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    # Scenario: CISO is blue and sees 2 red, 1 blue
    # CISO should be able to strategically discard a red patch
    env.current_phase = "legislative_ciso"
    env.drawn_cards = ["red", "red", "blue"]
    env.nominated_soc_lead = env.player_ids[1]  # Set SOC Lead for transition
    
    ciso_id = env.player_ids[env.current_ciso_idx]
    env.roles[ciso_id] = "blue"  # Make CISO blue
    
    observation = env.observe(ciso_id)
    
    # CISO can see all patches
    assert observation.get("drawn_cards") == ["red", "red", "blue"], \
        "CISO should see the patches to make strategic decision"
    
    # CISO (blue) would want to discard a red patch (index 0 or 1)
    # Let's say CISO discards index 0 (first red)
    action = {"discard_patch": 0}
    env.step(ciso_id, action)
    
    # After CISO's action, there should be 2 cards remaining
    assert len(env.drawn_cards) == 2, \
        f"Should have 2 cards after CISO discard, got {len(env.drawn_cards)}"
    assert env.drawn_cards == ["red", "blue"], \
        f"Expected ['red', 'blue'] remaining, got {env.drawn_cards}"
    
    print("✓ CISO can make strategic decision based on visible patches")


def run_all_tests():
    """Run all patch visibility tests."""
    print("\n" + "=" * 70)
    print("PATCH VISIBILITY TESTS")
    print("=" * 70 + "\n")
    
    tests = [
        test_ciso_sees_three_patches,
        test_ciso_sees_patches_in_correct_order,
        test_non_ciso_cannot_see_patches_during_legislative,
        test_soc_lead_sees_two_remaining_patches,
        test_full_legislative_session_patch_flow,
        test_patch_visibility_only_during_legislative_phase,
        test_patch_counts_are_correct,
        test_ciso_decision_based_on_visible_patches,
    ]
    
    failed = []
    passed = 0
    
    for test in tests:
        try:
            print(f"\nRunning: {test.__name__}")
            print(f"Description: {test.__doc__.strip()}")
            test()
            passed += 1
        except AssertionError as e:
            print(f"✗ FAILED: {e}")
            failed.append((test.__name__, str(e)))
        except Exception as e:
            print(f"✗ ERROR: {e}")
            failed.append((test.__name__, f"Error: {e}"))
    
    # Summary
    print("\n" + "=" * 70)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed:
        print("\nFailed tests:")
        for name, error in failed:
            print(f"  - {name}: {error}")
    else:
        print("\n🎉 All tests passed!")
    print("=" * 70 + "\n")
    
    return len(failed) == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
