#!/usr/bin/env python3
"""Test that beliefs are correctly updated when players die."""

import sys
sys.path.insert(0, str(__file__))

from red_vs_blue.env import RedvsBlueEnv

def test_belief_with_fired_players():
    """Verify beliefs exclude fired players and renormalize probabilities."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    print("Test 1: Initial state - all players employed")
    p0_obs = env.observe("p0")
    p0_belief = p0_obs.get("belief", {})
    assert len(p0_belief) == 4, f"Expected 4 players in belief, got {len(p0_belief)}"
    assert "p4" in p0_belief, "p4 should be in p0's belief"
    assert abs(p0_belief["p4"] - 0.25) < 0.01, f"Expected prob ~0.25, got {p0_belief['p4']}"
    print("  ✓ p0 has beliefs for 4 other players (p1, p2, p3, p4)")
    print(f"    Belief: {p0_belief}")
    
    print("\nTest 2: One player dies - belief should exclude fired player")
    env.fired_players.add("p4")
    
    p0_obs = env.observe("p0")
    p0_belief = p0_obs.get("belief", {})
    assert len(p0_belief) == 3, f"Expected 3 players in belief after p4 dies, got {len(p0_belief)}"
    assert "p4" not in p0_belief, "p4 should NOT be in p0's belief after being fired"
    assert "p1" in p0_belief, "p1 should still be in p0's belief"
    assert "p2" in p0_belief, "p2 should still be in p0's belief"
    assert "p3" in p0_belief, "p3 should still be in p0's belief"
    
    # Check probabilities are renormalized
    total_prob = sum(p0_belief.values())
    assert abs(total_prob - 1.0) < 0.01, f"Probabilities should sum to 1.0, got {total_prob}"
    expected_prob = 1.0 / 3  # Now 3 remaining players
    for player, prob in p0_belief.items():
        assert abs(prob - expected_prob) < 0.01, f"Expected prob ~{expected_prob}, got {prob} for {player}"
    
    print("  ✓ p0's belief correctly excludes p4")
    print(f"    Belief: {p0_belief}")
    print(f"    Probabilities sum to: {sum(p0_belief.values()):.4f}")
    
    print("\nTest 3: Another player dies")
    env.fired_players.add("p3")
    
    p0_obs = env.observe("p0")
    p0_belief = p0_obs.get("belief", {})
    assert len(p0_belief) == 2, f"Expected 2 players in belief after p3 dies, got {len(p0_belief)}"
    assert "p4" not in p0_belief, "p4 should NOT be in p0's belief"
    assert "p3" not in p0_belief, "p3 should NOT be in p0's belief"
    assert "p1" in p0_belief, "p1 should still be in p0's belief"
    assert "p2" in p0_belief, "p2 should still be in p0's belief"
    
    total_prob = sum(p0_belief.values())
    assert abs(total_prob - 1.0) < 0.01, f"Probabilities should sum to 1.0, got {total_prob}"
    expected_prob = 1.0 / 2  # Now 2 remaining players
    for player, prob in p0_belief.items():
        assert abs(prob - expected_prob) < 0.01, f"Expected prob ~{expected_prob}, got {prob} for {player}"
    
    print("  ✓ p0's belief correctly excludes p3 and p4")
    print(f"    Belief: {p0_belief}")
    print(f"    Probabilities sum to: {sum(p0_belief.values()):.4f}")
    
    print("\nTest 4: Check other players' beliefs are also correct")
    for player_id in ["p1", "p2"]:
        obs = env.observe(player_id)
        belief = obs.get("belief", {})
        assert len(belief) == 2, f"Expected 2 players in {player_id}'s belief, got {len(belief)}"
        assert "p3" not in belief, f"p3 should not be in {player_id}'s belief"
        assert "p4" not in belief, f"p4 should not be in {player_id}'s belief"
        print(f"  ✓ {player_id}'s belief is correct: {belief}")
    
    print("\n" + "="*60)
    print("✓ All belief tests passed!")
    print("✓ Beliefs correctly exclude fired players and renormalize probabilities")

if __name__ == "__main__":
    test_belief_with_fired_players()
