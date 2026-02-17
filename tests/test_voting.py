"""Test voting mechanism to ensure all players vote and voting is tracked correctly."""

from red_vs_blue.env import RedvsBlueEnv


def test_all_employed_players_must_vote():
    """Test that all employed players are required to vote."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    # Move to voting phase
    env.current_phase = "nomination"
    env.nominated_soc_lead = env.player_ids[1]
    env.current_phase = "voting"
    
    # Get employed players count
    employed_players = [p for p in env.player_ids if p not in env.fired_players]
    assert len(employed_players) == 5, "All players should be employed"
    
    # Verify voting check works
    assert not env.have_all_employed_players_voted(), "No votes yet"
    
    # Add partial votes
    env.council_votes[env.player_ids[0]] = "yes"
    env.council_votes[env.player_ids[1]] = "no"
    assert not env.have_all_employed_players_voted(), "Not all votes received"
    
    # Add remaining votes
    env.council_votes[env.player_ids[2]] = "yes"
    env.council_votes[env.player_ids[3]] = "no"
    assert not env.have_all_employed_players_voted(), "Still missing one vote"
    
    env.council_votes[env.player_ids[4]] = "yes"
    assert env.have_all_employed_players_voted(), "All votes received"
    print("✓ test_all_employed_players_must_vote")


def test_fired_players_not_required_to_vote():
    """Test that fired/fired players are not required to vote."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    # Kill a player
    env.fired_players.add(env.player_ids[4])
    
    # Move to voting phase
    env.current_phase = "nomination"
    env.nominated_soc_lead = env.player_ids[1]
    env.current_phase = "voting"
    
    # Verify employed count is 4
    employed_players = [p for p in env.player_ids if p not in env.fired_players]
    assert len(employed_players) == 4, "Should have 4 employed players"
    
    # Add votes from only employed players
    env.council_votes[env.player_ids[0]] = "yes"
    env.council_votes[env.player_ids[1]] = "no"
    env.council_votes[env.player_ids[2]] = "yes"
    env.council_votes[env.player_ids[3]] = "no"
    
    # Should be complete without player_ids[4]'s vote
    assert env.have_all_employed_players_voted(), "All employed players voted"
    print("✓ test_fired_players_not_required_to_vote")


def test_invalid_votes_not_counted():
    """Test that invalid votes don't count as cast."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    env.current_phase = "voting"
    
    # Add valid votes
    env.council_votes[env.player_ids[0]] = "yes"
    env.council_votes[env.player_ids[1]] = "no"
    env.council_votes[env.player_ids[2]] = "yes"
    env.council_votes[env.player_ids[3]] = "no"
    
    # Add invalid vote (not in dict counts as missing)
    assert not env.have_all_employed_players_voted(), "player_ids[4] hasn't voted"
    
    # Invalid vote value wouldn't be added by step(), but verify logic
    env.council_votes[env.player_ids[4]] = "abstain"  # Invalid
    assert not env.have_all_employed_players_voted(), "Invalid vote shouldn't count"
    print("✓ test_invalid_votes_not_counted")


def test_majority_calculation_with_all_votes():
    """Test that majority is correctly calculated when all players have voted."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    env.current_phase = "voting"
    env.nominated_soc_lead = env.player_ids[1]
    
    # 3 yes, 2 no - should approve (need > 2.5)
    env.council_votes[env.player_ids[0]] = "yes"
    env.council_votes[env.player_ids[1]] = "yes"
    env.council_votes[env.player_ids[2]] = "yes"
    env.council_votes[env.player_ids[3]] = "no"
    env.council_votes[env.player_ids[4]] = "no"
    
    # Verify all voted
    assert env.have_all_employed_players_voted()
    
    # Resolve vote
    env._resolve_council_vote()
    assert env.current_phase == "legislative_ciso", "Council should be approved"
    print("✓ test_majority_calculation_with_all_votes")


def test_majority_calculation_rejection():
    """Test that councils are rejected when not majority."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    env.current_phase = "voting"
    env.nominated_soc_lead = env.player_ids[1]
    
    # 2 yes, 3 no - should reject
    env.council_votes[env.player_ids[0]] = "yes"
    env.council_votes[env.player_ids[1]] = "yes"
    env.council_votes[env.player_ids[2]] = "no"
    env.council_votes[env.player_ids[3]] = "no"
    env.council_votes[env.player_ids[4]] = "no"
    
    # Verify all voted
    assert env.have_all_employed_players_voted()
    
    # Resolve vote
    env._resolve_council_vote()
    assert env.current_phase != "legislative", "Council should be rejected"
    print("✓ test_majority_calculation_rejection")


def test_voting_history_recorded():
    """Test that voting history is properly recorded."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    # Set up voting
    env.current_ciso_idx = 0
    env.current_phase = "voting"
    env.nominated_soc_lead = env.player_ids[1]
    env.council_votes[env.player_ids[0]] = "yes"
    env.council_votes[env.player_ids[1]] = "yes"
    env.council_votes[env.player_ids[2]] = "yes"
    env.council_votes[env.player_ids[3]] = "no"
    env.council_votes[env.player_ids[4]] = "no"
    
    # Clear any existing history
    env.voting_history = []
    
    # Resolve vote
    env._resolve_council_vote()
    
    # Verify history recorded
    assert len(env.voting_history) == 1, "One vote recorded"
    vote_record = env.voting_history[0]
    assert vote_record["ciso"] == env.player_ids[0], "CISO recorded"
    assert vote_record["nominated_soc_lead"] == env.player_ids[1], "SOC Lead recorded"
    assert vote_record["approved"] == True, "Approved flag correct"
    assert vote_record["yes_votes"] == 3, "Yes votes correct"
    assert vote_record["no_votes"] == 2, "No votes correct"
    assert env.player_ids[0] in vote_record["votes"], "Player votes recorded"
    print("✓ test_voting_history_recorded")


def test_voting_history_rejection():
    """Test that rejected votes are also recorded in history."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    env.current_ciso_idx = 0
    env.current_phase = "voting"
    env.nominated_soc_lead = env.player_ids[1]
    env.council_votes[env.player_ids[0]] = "no"
    env.council_votes[env.player_ids[1]] = "no"
    env.council_votes[env.player_ids[2]] = "yes"
    env.council_votes[env.player_ids[3]] = "yes"
    env.council_votes[env.player_ids[4]] = "no"
    env.voting_history = []
    
    # Resolve vote
    env._resolve_council_vote()
    
    # Verify rejection recorded
    assert len(env.voting_history) == 1
    vote_record = env.voting_history[0]
    assert vote_record["approved"] == False, "Rejection recorded"
    assert vote_record["yes_votes"] == 2
    assert vote_record["no_votes"] == 3
    print("✓ test_voting_history_rejection")


def test_consecutive_rejections_tracked():
    """Test that consecutive rejections are tracked."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    env.consecutive_failed_councils = 0
    env.current_ciso_idx = 0
    
    # First rejection
    env.current_phase = "voting"
    env.nominated_soc_lead = env.player_ids[1]
    for p in env.player_ids:
        env.council_votes[p] = "no"
    env._resolve_council_vote()
    assert env.consecutive_failed_councils == 1
    
    # Second rejection (simulated)
    for p in env.player_ids:
        env.council_votes[p] = "no"
    env.nominated_soc_lead = env.player_ids[2]
    env._resolve_council_vote()
    assert env.consecutive_failed_councils == 2
    
    # Third rejection triggers forced patch
    for p in env.player_ids:
        env.council_votes[p] = "no"
    env.nominated_soc_lead = env.player_ids[3]
    initial_patches = env.patch_track["blue"] + env.patch_track["red"]
    env._resolve_council_vote()
    final_patches = env.patch_track["blue"] + env.patch_track["red"]
    assert final_patches > initial_patches, "Patch applied on third failure"
    print("✓ test_consecutive_rejections_tracked")


def test_voting_clears_after_resolution():
    """Test that votes are cleared after resolution."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    env.current_phase = "voting"
    env.nominated_soc_lead = env.player_ids[1]
    env.council_votes[env.player_ids[0]] = "yes"
    env.council_votes[env.player_ids[1]] = "yes"
    env.council_votes[env.player_ids[2]] = "yes"
    env.council_votes[env.player_ids[3]] = "no"
    env.council_votes[env.player_ids[4]] = "no"
    
    # Resolve
    env._resolve_council_vote()
    
    # Votes should be cleared
    assert len(env.council_votes) == 0, "Votes cleared after resolution"
    print("✓ test_voting_clears_after_resolution")


def test_tie_breaking_in_voting():
    """Test that ties result in council rejection (need > majority, not >=)."""
    env = RedvsBlueEnv(num_players=4, seed=42)
    env.reset()
    
    env.current_phase = "voting"
    env.nominated_soc_lead = env.player_ids[1]
    # 2 yes, 2 no - tie
    env.council_votes[env.player_ids[0]] = "yes"
    env.council_votes[env.player_ids[1]] = "yes"
    env.council_votes[env.player_ids[2]] = "no"
    env.council_votes[env.player_ids[3]] = "no"
    
    # Verify all voted
    assert env.have_all_employed_players_voted()
    
    # Resolve - should reject (need > 2)
    env._resolve_council_vote()
    assert env.consecutive_failed_councils == 1, "Tie should be treated as rejection"
    print("✓ test_tie_breaking_in_voting")


if __name__ == "__main__":
    try:
        test_all_employed_players_must_vote()
        test_fired_players_not_required_to_vote()
        test_invalid_votes_not_counted()
        test_majority_calculation_with_all_votes()
        test_majority_calculation_rejection()
        test_voting_history_recorded()
        test_voting_history_rejection()
        test_consecutive_rejections_tracked()
        test_voting_clears_after_resolution()
        test_tie_breaking_in_voting()
        print("\n✅ All voting tests passed! (10/10)")
    except AssertionError as e:
        print(f"\n❌ Test failed: {e}")
        raise
