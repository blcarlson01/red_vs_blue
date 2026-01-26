"""Test that voting always produces votes (not 0 yes/no)."""

from red_vs_blue.env import RedvsBlueEnv
from red_vs_blue.agents import RedvsBlueAgent, create_agents
from red_vs_blue.metrics import average_belief_alignment


def test_vote_defaults_when_parse_fails():
    """Test that even with bad JSON, votes are recorded."""
    print("\n" + "="*70)
    print("TEST: Vote Defaults When JSON Parsing Fails")
    print("="*70)
    
    # Create agent that we can manually set response for
    agent = RedvsBlueAgent(
        model=None,  # We'll mock this
        player_id="test_p0",
        role="blue",
        all_players=["test_p0", "p1", "p2", "p3", "p4"],
        temperature=0.7
    )
    
    # Simulate bad JSON response
    bad_json = "This is not valid JSON at all!!!"
    
    # Parse response - should use defaults
    action = agent._parse_json_response(
        bad_json,
        fallback_belief={"p1": 0.5, "p2": 0.5},
        phase="voting",
        employed_players=["p1", "p2"]
    )
    
    print(f"\nBad JSON: {bad_json}")
    print(f"Parsed action: {action}")
    print(f"✓ Vote field present: {'vote' in action}")
    print(f"✓ Vote value: {action.get('vote')}")
    assert "vote" in action, "Vote should be present in action dict"
    assert action["vote"] in ["yes", "no"], f"Vote should be 'yes' or 'no', got {action['vote']}"
    print("✓ PASS: Default vote provided when JSON parsing fails\n")


def test_vote_defaults_on_empty_response():
    """Test that even with empty response, votes are recorded."""
    print("="*70)
    print("TEST: Vote Defaults On Empty Response")
    print("="*70)
    
    agent = RedvsBlueAgent(
        model=None,
        player_id="test_p1",
        role="red",
        all_players=["p0", "test_p1", "p2", "p3", "p4"],
        temperature=0.7
    )
    
    # Empty response
    action = agent._parse_json_response(
        "",
        fallback_belief={"p0": 0.5, "p2": 0.5},
        phase="voting",
        employed_players=["p0", "p2"]
    )
    
    print(f"\nEmpty response")
    print(f"Parsed action: {action}")
    print(f"✓ Vote field present: {'vote' in action}")
    print(f"✓ Vote value: {action.get('vote')}")
    assert "vote" in action, "Vote should be present in action dict"
    assert action["vote"] in ["yes", "no"], f"Vote should be 'yes' or 'no', got {action['vote']}"
    print("✓ PASS: Default vote provided on empty response\n")


def test_environment_records_all_votes():
    """Test that environment correctly records all votes."""
    print("="*70)
    print("TEST: Environment Records All Votes")
    print("="*70)
    
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    env.reset()
    
    # Move to voting phase
    env.current_phase = "voting"
    env.nominated_soc_lead = "p1"
    
    print(f"\nInitial state:")
    print(f"  Phase: {env.current_phase}")
    print(f"  Nominated: {env.nominated_soc_lead}")
    print(f"  Votes before: {env.council_votes}")
    
    # Simulate votes from all players
    for player_id in env.player_ids:
        if player_id not in env.fired_players:
            vote = "yes" if player_id in ["p0", "p2"] else "no"
            action = {"vote": vote, "message": f"I vote {vote}"}
            env.step(player_id, action)
    
    print(f"\nVotes after all players acted:")
    print(f"  {env.council_votes}")
    
    # Count votes
    yes_votes = sum(1 for v in env.council_votes.values() if v == "yes")
    no_votes = sum(1 for v in env.council_votes.values() if v == "no")
    
    print(f"\nVote count:")
    print(f"  Yes votes: {yes_votes}")
    print(f"  No votes: {no_votes}")
    print(f"  Total: {len(env.council_votes)}")
    
    assert yes_votes == 2, f"Should have 2 yes votes, got {yes_votes}"
    assert no_votes == 3, f"Should have 3 no votes, got {no_votes}"
    assert len(env.council_votes) == 5, f"Should have 5 total votes, got {len(env.council_votes)}"
    print("✓ PASS: All votes recorded correctly\n")


def test_non_voting_phase_has_no_vote():
    """Test that non-voting phases don't get votes by default."""
    print("="*70)
    print("TEST: Non-Voting Phases Have No Vote")
    print("="*70)
    
    agent = RedvsBlueAgent(
        model=None,
        player_id="test_p2",
        role="blue",
        all_players=["p0", "p1", "test_p2", "p3", "p4"],
        temperature=0.7
    )
    
    # Discussion phase should not get vote field
    action = agent._parse_json_response(
        "This is a discussion response",
        fallback_belief={"p0": 0.33, "p1": 0.33, "p3": 0.34},
        phase="discussion",
        employed_players=["p0", "p1", "p3"]
    )
    
    print(f"\nDiscussion phase response")
    print(f"Parsed action: {action}")
    print(f"✓ Has message: {'message' in action}")
    print(f"✓ Has vote: {'vote' in action}")
    
    # Discussion should not have vote by default
    assert "vote" not in action or not action.get("vote"), "Discussion phase should not have vote"
    print("✓ PASS: Discussion phase correctly has no vote\n")


if __name__ == "__main__":
    test_vote_defaults_when_parse_fails()
    test_vote_defaults_on_empty_response()
    test_environment_records_all_votes()
    test_non_voting_phase_has_no_vote()
    
    print("="*70)
    print("✨ ALL TESTS PASSED")
    print("="*70)
    print("\n✅ Voting mechanism now correctly:")
    print("   - Provides default votes when JSON parsing fails")
    print("   - Provides default votes on empty responses")
    print("   - Records all votes in the environment")
    print("   - Doesn't add votes to non-voting phases\n")
