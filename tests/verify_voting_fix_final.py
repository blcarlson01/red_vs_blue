#!/usr/bin/env python3
"""Final verification that voting_history has votes recorded correctly."""

import sys
sys.path.insert(0, str(__file__))

from red_vs_blue.env import RedvsBlueEnv

def test_voting_history_populated():
    """Verify voting_history has votes recorded."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    # Manual 1 voting round
    env.current_phase = "nomination"
    env.nominated_soc_lead = "p1"
    env.current_phase = "voting"
    
    # All players vote
    for player_id, vote in [("p0", "yes"), ("p1", "no"), ("p2", "yes"), ("p3", "no"), ("p4", "yes")]:
        env.step(player_id, {"vote": vote})
    
    # Resolve voting
    env.end_round()
    
    # Verify voting_history
    assert len(env.voting_history) > 0, "No voting history recorded!"
    last_vote = env.voting_history[-1]
    
    print("✓ Voting history recorded")
    print(f"  Round: {last_vote['round']}")
    print(f"  CISO: {last_vote['ciso']}")
    print(f"  SOC Lead: {last_vote['nominated_soc_lead']}")
    print(f"  Votes: {last_vote['votes']}")
    print(f"  Approved: {last_vote['approved']}")
    print(f"  Result: {last_vote['result']}")
    
    # Verify votes dict is populated
    assert last_vote['votes'], "Votes dict is empty!"
    assert len(last_vote['votes']) == 5, f"Expected 5 votes, got {len(last_vote['votes'])}"
    assert last_vote['yes_votes'] == 3, f"Expected 3 yes votes, got {last_vote['yes_votes']}"
    assert last_vote['no_votes'] == 2, f"Expected 2 no votes, got {last_vote['no_votes']}"
    assert last_vote['approved'] == True, "Should be approved with 3 yes votes"
    
    print("\n✓ All assertions passed!")
    print("✓ Voting history fix verified successfully!")

if __name__ == "__main__":
    test_voting_history_populated()
