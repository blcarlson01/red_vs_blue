#!/usr/bin/env python3
"""
Test to diagnose what values are actually in the observation dict.
"""

import sys
sys.path.insert(0, ".")

from red_vs_blue.env import RedvsBlueEnv
from inspect_ai.model import Model

def main():
    # Create a simple environment
    env = RedvsBlueEnv(num_players=5)

    # Get observation at different phases
    print("=" * 80)
    print("TESTING OBSERVATION VALUES AT EACH PHASE")
    print("=" * 80)

    # Phase 1: Discussion
    print("\n1. DISCUSSION PHASE")
    obs = env.observe("p0")
    print(f"   current_ciso: {repr(obs.get('current_ciso'))}")
    print(f"   nominated_soc_lead: {repr(obs.get('nominated_soc_lead'))}")
    print(f"   phase: {repr(obs.get('phase'))}")
    print(f"   your_role: {repr(obs.get('your_role'))}")
    print(f"   Type of current_ciso: {type(obs.get('current_ciso'))}")

    # Manually set to nomination phase and check
    env.current_phase = "nomination"
    print("\n2. NOMINATION PHASE")
    obs = env.observe("p0")
    print(f"   current_ciso: {repr(obs.get('current_ciso'))}")
    print(f"   nominated_soc_lead: {repr(obs.get('nominated_soc_lead'))}")
    print(f"   phase: {repr(obs.get('phase'))}")
    print(f"   your_role: {repr(obs.get('your_role'))}")
    print(f"   current_ciso_idx: {env.current_ciso_idx}")
    print(f"   player_ids: {env.player_ids}")

    # Try voting phase
    env.current_phase = "voting"
    env.nominated_soc_lead = "p1"
    print("\n3. VOTING PHASE")
    obs = env.observe("p0")
    print(f"   current_ciso: {repr(obs.get('current_ciso'))}")
    print(f"   nominated_soc_lead: {repr(obs.get('nominated_soc_lead'))}")
    print(f"   phase: {repr(obs.get('phase'))}")
    print(f"   your_role: {repr(obs.get('your_role'))}")

    # Check all observation keys
    print("\n4. ALL OBSERVATION KEYS")
    for key, value in sorted(obs.items()):
        if key not in ['public_log', 'belief']:
            print(f"   {key}: {repr(value)}")

    print("\n" + "=" * 80)
    print("CHECKING FOR LITERAL 'ciso' STRING IN OBSERVATION")
    print("=" * 80)

    # Search for any instance of the string "ciso" (not "ciso" as player_id prefix)
    for key, value in obs.items():
        if isinstance(value, str) and "ciso" in value.lower():
            print(f"FOUND: {key} = {repr(value)}")
        elif isinstance(value, dict):
            for k2, v2 in value.items():
                if isinstance(v2, str) and "ciso" in v2.lower():
                    print(f"FOUND in {key}: {k2} = {repr(v2)}")
        elif isinstance(value, list):
            for i, item in enumerate(value):
                if isinstance(item, str) and "ciso" in item.lower():
                    print(f"FOUND in {key}[{i}]: {repr(item)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
