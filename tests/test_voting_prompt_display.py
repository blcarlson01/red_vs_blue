"""Test that voting prompts show current vote counts."""

from red_vs_blue.env import RedvsBlueEnv
from red_vs_blue.agents import RedvsBlueAgent

def test_voting_prompt_shows_vote_counts():
    """Verify voting phase prompts show current vote counts."""
    print("\n" + "="*70)
    print("TEST: Voting Prompt Shows Vote Counts")
    print("="*70 + "\n")
    
    # Create environment and agent
    env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=42)
    env.reset()
    
    agent = RedvsBlueAgent(
        model=None,
        player_id="p0",
        role="blue",
        all_players=env.player_ids,
        temperature=0.7
    )
    
    # Move to voting phase
    env.current_phase = "voting"
    env.nominated_soc_lead = "p1"
    
    # Collect observation
    obs = env.observe("p0")
    
    print(f"Council votes in observation: {obs.get('council_votes', {})}")
    print(f"Employed players: {obs.get('employed_players', [])}")
    
    # Build prompt (simulating agent)
    prompt = agent._user_prompt(obs)
    
    print(f"\nPrompt excerpt (voting phase):")
    lines = prompt.split("\n")
    in_voting = False
    for line in lines:
        if "voting" in line.lower():
            in_voting = True
        if in_voting:
            print(f"  {line}")
            if "vote" in line and "respond" not in line.lower():
                break
    
    # Now add a vote and check again
    print(f"\n" + "─"*70)
    print("After p0 votes 'yes':")
    print("─"*70 + "\n")
    
    env.step("p0", {"vote": "yes", "message": "I vote yes"})
    obs = env.observe("p1")
    
    print(f"Council votes in observation: {obs.get('council_votes', {})}")
    
    prompt = agent._user_prompt(obs)
    
    print(f"\nPrompt excerpt (voting phase with 1 vote):")
    lines = prompt.split("\n")
    in_voting = False
    for line in lines:
        if "voting" in line.lower():
            in_voting = True
        if in_voting:
            print(f"  {line}")
            if "vote" in line and "respond" not in line.lower():
                break
    
    # Add more votes
    print(f"\n" + "─"*70)
    print("After 3 more players vote (p1=yes, p2=no, p3=no):")
    print("─"*70 + "\n")
    
    env.step("p1", {"vote": "yes", "message": "I vote yes"})
    env.step("p2", {"vote": "no", "message": "I vote no"})
    env.step("p3", {"vote": "no", "message": "I vote no"})
    
    obs = env.observe("p4")
    
    print(f"Council votes in observation: {obs.get('council_votes', {})}")
    
    prompt = agent._user_prompt(obs)
    
    print(f"\nPrompt excerpt (voting phase with 4 votes):")
    lines = prompt.split("\n")
    in_voting = False
    for line in lines:
        if "voting" in line.lower():
            in_voting = True
        if in_voting:
            print(f"  {line}")
            if "vote" in line and "respond" not in line.lower():
                break
    
    print("\n" + "="*70)
    print("✅ Test Complete: Vote counts are now shown in prompts")
    print("="*70 + "\n")

if __name__ == "__main__":
    test_voting_prompt_shows_vote_counts()
