"""Test that belief tracking excludes self."""

from red_vs_blue.env import RedvsBlueEnv

def test_belief_excludes_self():
    """Verify that player beliefs don't include themselves."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    print("="*60)
    print("BELIEF TRACKING SELF-EXCLUSION TEST")
    print("="*60)
    
    player_ids = env.player_ids
    print(f"\n1. Player IDs: {player_ids}")
    
    # Check initial beliefs
    print(f"\n2. Initial Beliefs (uniform over OTHER players):")
    for pid in player_ids:
        initial_belief = env.belief_histories[pid][0]
        print(f"   {pid}:")
        print(f"     - Has belief about self: {pid in initial_belief}")
        print(f"     - Beliefs over others: {len(initial_belief)} players")
        print(f"     - Others: {list(initial_belief.keys())}")
        
        # Verify self is excluded
        assert pid not in initial_belief, f"Error: {pid} has belief about self"
        
        # Verify all other players are included
        expected_others = [p for p in player_ids if p != pid]
        assert set(initial_belief.keys()) == set(expected_others), f"Error: Missing other players"
        
        # Verify probabilities sum to 1
        total = sum(initial_belief.values())
        assert abs(total - 1.0) < 1e-6, f"Error: Probabilities don't sum to 1: {total}"
        
        print(f"     ✅ Valid: {len(initial_belief)} others, sum={total:.4f}")
    
    # Test belief update with self-exclusion
    print(f"\n3. Testing Belief Update with Self-Exclusion:")
    
    # Simulate agent providing belief that includes self
    belief_with_self = {player_ids[0]: 0.3, player_ids[1]: 0.2, player_ids[2]: 0.25, 
                        player_ids[3]: 0.15, player_ids[4]: 0.1}
    
    action = {
        "belief": belief_with_self,
        "message": "I think p3 might be suspicious"
    }
    
    # For player p0, update belief
    env.step(player_ids[0], action)
    
    # Check that self (p0) was removed from stored belief
    updated_beliefs = env.belief_histories[player_ids[0]]
    print(f"   Updated belief for {player_ids[0]}:")
    print(f"     - Length of history: {len(updated_beliefs)}")
    print(f"     - Latest belief: {updated_beliefs[-1]}")
    
    latest = updated_beliefs[-1]
    assert player_ids[0] not in latest, f"Error: {player_ids[0]} still has belief about self"
    assert len(latest) == 4, f"Error: Should have beliefs about 4 others, got {len(latest)}"
    
    # Verify probabilities were renormalized
    total = sum(latest.values())
    assert abs(total - 1.0) < 1e-6, f"Error: Updated probabilities don't sum to 1: {total}"
    print(f"     ✅ Self excluded, {len(latest)} others, sum={total:.4f}")
    
    # Test _latest_belief method
    print(f"\n4. Testing _latest_belief() method:")
    for pid in player_ids:
        latest = env._latest_belief(pid)
        print(f"   {pid}: {len(latest)} other players (has self: {pid in latest})")
        assert pid not in latest, f"Error: {pid} in their own belief distribution"
    
    print(f"\n" + "="*60)
    print("✅ BELIEF SELF-EXCLUSION TEST PASSED")
    print("="*60)
    print("\nKey Verifications:")
    print("  ✅ Initial beliefs exclude self")
    print("  ✅ Beliefs over only other players")
    print("  ✅ Probabilities normalize correctly")
    print("  ✅ Updated beliefs maintain invariant")
    print("  ✅ _latest_belief() excludes self")

if __name__ == "__main__":
    test_belief_excludes_self()
