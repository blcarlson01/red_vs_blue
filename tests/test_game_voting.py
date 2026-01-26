#!/usr/bin/env python3
"""Run a simple game to debug voting with full output."""

import sys
sys.path.insert(0, str(__file__))

from red_vs_blue.env import RedvsBlueEnv

class MockAgent:
    """Simple agent that just votes/nominates in a round-robin way."""
    
    def __init__(self, player_id):
        self.player_id = player_id
        self.vote_cycle = ["yes", "no"]
        self.vote_idx = 0
    
    def act(self, observation):
        """Return action based on phase."""
        phase = observation.get("phase")
        
        if phase == "voting":
            vote = self.vote_cycle[self.vote_idx % 2]
            self.vote_idx += 1
            return {"vote": vote}
        elif phase == "nomination":
            # Nominate second player in order (not self)
            my_idx = int(self.player_id[1])  # p0 -> 0, p1 -> 1, etc.
            nom_idx = (my_idx + 1) % 5
            nom_player = f"p{nom_idx}"
            return {"nomination": nom_player}
        elif phase == "discussion":
            return {"statement": "I am blue"}
        elif phase == "legislative":
            return {"discard_patch": 0}
        elif phase == "power":
            return {"power_action": None}
        
        return {}

def test_game_voting():
    """Test voting in a real game."""
    env = RedvsBlueEnv(num_players=5, seed=42)
    env.reset()
    
    agents = [MockAgent(pid) for pid in env.player_ids]
    for agent in agents:
        agent.role = env.roles[agent.player_id]
    
    turn_count = 0
    max_turns = 200
    
    while not env.done and turn_count < max_turns:
        phase = env.current_phase
        
        if phase == "voting":
            print(f"\n[VOTING TURN {turn_count}] Round {env.round}, Phase: {phase}")
            print(f"  Council votes before: {env.council_votes}")
        
        # Get actions from agents
        if phase == "discussion":
            for agent in agents:
                if agent.player_id not in env.fired_players:
                    observation = env.observe(agent.player_id)
                    action = agent.act(observation)
                    env.step(agent.player_id, action)
            env.end_round()
        
        elif phase == "nomination":
            current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
            for agent in agents:
                if agent.player_id == current_ciso:
                    observation = env.observe(agent.player_id)
                    action = agent.act(observation)
                    print(f"  CISO {agent.player_id} nominates: {action.get('nomination')}")
                    env.step(agent.player_id, action)
                    break
            # Nomination phase changes to voting inside step() if successful nomination
            # Only call end_round if we're still in nomination (nomination failed)
            if env.current_phase == "nomination":
                env.end_round()
        
        elif phase == "voting":
            # Get votes from all employed players
            votes_before = len(env.council_votes)
            for agent in agents:
                if agent.player_id not in env.fired_players and agent.player_id not in env.council_votes:
                    observation = env.observe(agent.player_id)
                    action = agent.act(observation)
                    print(f"  Agent {agent.player_id} votes: {action.get('vote')}")
                    env.step(agent.player_id, action)
            
            print(f"  Council votes after: {env.council_votes}")
            print(f"  Calling end_round()...")
            env.end_round()
            print(f"  Council votes after end_round(): {env.council_votes}")
        
        elif phase == "legislative":
            current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
            for agent in agents:
                if agent.player_id == current_ciso:
                    observation = env.observe(agent.player_id)
                    action = agent.act(observation)
                    env.step(agent.player_id, action)
                    break
            env.end_round()
        
        elif phase == "power":
            current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
            for agent in agents:
                if agent.player_id == current_ciso:
                    observation = env.observe(agent.player_id)
                    action = agent.act(observation)
                    env.step(agent.player_id, action)
                    break
            env.end_round()
        
        else:
            print(f"[TURN {turn_count}] Unknown phase: {phase}")
            break
        
        turn_count += 1
    
    print("\n" + "="*60)
    print("GAME RESULTS - VOTING HISTORY:")
    print("="*60)
    
    if env.voting_history:
        for i, vote_record in enumerate(env.voting_history[:10]):
            print(f"\nVote Record {i}:")
            print(f"  Round: {vote_record.get('round')}")
            print(f"  CISO: {vote_record.get('ciso')}")
            print(f"  SOC Lead: {vote_record.get('nominated_soc_lead')}")
            print(f"  Votes: {vote_record.get('votes')}")
            print(f"  Approved: {vote_record.get('approved')}")
            print(f"  Result: {vote_record.get('result')}")
    else:
        print("NO VOTING HISTORY!")

if __name__ == "__main__":
    test_game_voting()
