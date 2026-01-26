#!/usr/bin/env python
"""Final integration test - run actual game."""

from red_vs_blue.env import RedvsBlueEnv

print("=" * 60)
print("INTEGRATION TEST - RUN ACTUAL GAME")
print("=" * 60)
print()

# Run a game with all the mechanics
env = RedvsBlueEnv(num_players=5, max_rounds=10, seed=123)

print(f"Game initialized:")
print(f"  Players: {env.player_ids}")
print(f"  Roles: {env.roles}")
print(f"  Patch deck: {len(env.patch_deck)} cards")
print()

round_count = 0
while not env.done and round_count < 20:
    round_count += 1
    
    if env.current_phase == "discussion":
        # Players discuss
        for player_id in env.player_ids:
            if player_id not in env.fired_players:
                env.step(player_id, {"message": f"{player_id} discusses"})
        env.end_round()
        
    elif env.current_phase == "nomination":
        # CISO nominates soc_lead
        ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
        candidates = [p for p in env.player_ids if p != ciso and p not in env.fired_players]
        if candidates:
            nominee = candidates[0]
            env.step(ciso, {"nomination": nominee})
        env.end_round()
        
    elif env.current_phase == "voting":
        # All players vote - mostly yes to move game forward
        employed = [p for p in env.player_ids if p not in env.fired_players]
        for i, player_id in enumerate(employed):
            vote = "yes" if i < len(employed) // 2 + 1 else "no"
            env.step(player_id, {"vote": vote})
        env.end_round()
        
    elif env.current_phase == "legislative":
        # CISO picks patch to discard (always discard first)
        ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
        env.step(ciso, {"discard_patch": 0})
        env.end_round()
        
    elif env.current_phase == "power":
        # CISO executes power
        ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
        red_count = env.patch_track["red"]
        
        if red_count >= 4:
            # Fire first target
            targets = [p for p in env.player_ids if p != ciso and p not in env.fired_players]
            if targets:
                env.step(ciso, {"power_action": {"fire": targets[0]}})
        elif red_count == 3:
            # Special election
            targets = [p for p in env.player_ids if p != ciso and p not in env.fired_players]
            if targets:
                env.step(ciso, {"power_action": {"special_election": targets[0]}})
        elif red_count == 2:
            # Investigate
            targets = [p for p in env.player_ids if p != ciso and p not in env.fired_players]
            if targets:
                env.step(ciso, {"power_action": {"investigate": targets[0]}})
        
        env.end_round()
        
    else:
        print(f"Unknown phase: {env.current_phase}")
        break
    
    print(f"Round {round_count}: {env.current_phase} -> blue={env.patch_track['blue']} red={env.patch_track['red']}")

print()
print(f"Game ended after {round_count} rounds")
print(f"Final state:")
print(f"  Done: {env.done}")
print(f"  Patch track: {env.patch_track}")
print(f"  Fired players: {env.fired_players}")
print()

if env.done:
    if env.blues_win():
        print("BLUES WIN!")
    elif env.reds_win():
        print("REDS WIN!")
    else:
        print("UNCLEAR WHO WON")
else:
    print("Game did not complete")

print()
print("=" * 60)
print("INTEGRATION TEST COMPLETE")
print("=" * 60)
