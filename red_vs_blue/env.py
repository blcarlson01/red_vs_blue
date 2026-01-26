# red_vs_blue/env.py

from __future__ import annotations
import time
import random
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

import numpy as np


@dataclass
class PlayerState:
    player_id: str
    role: str  # "blue", "red", "apt_leader"
    employed: bool = True


class RedvsBlueEnv:
    """
    Inspect-compatible environment for Red vs. Blue (proper rules).
    
    Rules Summary:
    - 5 players: 3 blues, 1 red, 1 APT Leader
    - Blue deck: 6 patches (blues win at 6)
    - Red deck: 11 patches (reds win at 6)
    - Council formation: CISO nominates SOC Lead, all vote
    - If approved: Draw 3, CISO discards 1, SOC Lead discards 1, enact top 2
    - CISO powers: After each red patch (investigate, special election, fire)
    - Reds win if: 6 red patches applied OR APT Leader elected SOC Lead (after 3+ patches)
    - Blues win if: 6 blue patches applied OR APT Leader fired OR all reds fired
    """

    def __init__(self, num_players: int = 5, max_rounds: int = 10, seed: int | None = None):
        self.num_players = num_players
        self.max_rounds = max_rounds
        self.seed = seed
        self.rng = random.Random(seed)

        # Player tracking
        self.player_ids: List[str] = [f"p{i}" for i in range(num_players)]
        self.roles: Dict[str, str] = {}
        self.players: List[PlayerState] = []
        self.fired_players: set[str] = set()
        self.true_apt_leader_id: str | None = None

        # Game state
        self.round = 0
        self.done = False
        self.patch_track = {"blue": 0, "red": 0}
        self.public_log: List[str] = []
        self.belief_histories: Dict[str, List[Dict[str, float]]] = {}
        self.voting_history: List[Dict] = []

        # Council formation state
        self.current_ciso_idx = 0  # Index into self.player_ids
        self.nominated_soc_lead: Optional[str] = None
        self.council_votes: Dict[str, str] = {}  # player_id -> "yes"/"no"
        self.votes_received_count = 0  # Track number of votes received
        self.consecutive_failed_councils = 0
        self.investigation_results: Dict[str, str] = {}  # player_id -> role (if investigated)

        # Game phases
        self.current_phase = "discussion"  # discussion → nomination → voting → legislative → power → resolution
        
        # Patch deck management
        self.patch_deck: List[str] = []
        self.discard_pile: List[str] = []
        self.drawn_cards: List[str] = []
        self._initialize_patch_deck()

        # Timing
        self.game_start_time = None
        self.game_end_time = None
        self.turn_times: List[float] = []
        self._turn_start_time = None

        # Initialize
        self.reset()

    def _initialize_patch_deck(self):
        """Create patch deck: 6 blue, 11 red."""
        self.patch_deck = ["blue"] * 6 + ["red"] * 11
        self.rng.shuffle(self.patch_deck)
        self.discard_pile = []

    def _reshuffle_deck_if_needed(self):
        """Reshuffle discard pile into deck if deck is empty."""
        if not self.patch_deck and self.discard_pile:
            self.patch_deck = self.discard_pile
            self.discard_pile = []
            self.rng.shuffle(self.patch_deck)

    # --------------------------------------------------
    # Lifecycle
    # --------------------------------------------------

    def reset(self):
        """Initialize a new game."""
        self._assign_roles()
        self.round = 0
        self.done = False
        self.current_phase = "discussion"
        self.patch_track = {"blue": 0, "red": 0}
        self.council_votes = {}
        self.votes_received_count = 0
        self.nominated_soc_lead = None
        self.consecutive_failed_councils = 0
        # Choose initial ciso randomly
        self.current_ciso_idx = self.rng.randint(0, len(self.player_ids) - 1)
        self.investigation_results = {}
        self._initialize_patch_deck()

        self.game_start_time = time.perf_counter()
        self.turn_times.clear()

        return self._observe_all()

    def _assign_roles(self):
        """Assign roles with red knowledge asymmetry."""
        roles = ["apt_leader"] + ["red"] + ["blue"] * (self.num_players - 2)
        self.rng.shuffle(roles)

        self.roles = {pid: role for pid, role in zip(self.player_ids, roles)}
        self.players = [PlayerState(pid, self.roles[pid]) for pid in self.player_ids]
        self.fired_players = set()

        # Find APT Leader
        for pid, role in self.roles.items():
            if role == "apt_leader":
                self.true_apt_leader_id = pid
                break

        # Initialize logs
        self.public_log = []
        self.belief_histories = {}
        n = len(self.player_ids)
        # Initialize uniform beliefs over OTHER players (excluding self, since players know their own role)
        for pid in self.player_ids:
            other_players = [p for p in self.player_ids if p != pid]
            uniform = {p: 1.0 / len(other_players) for p in other_players}
            self.belief_histories[pid] = [uniform.copy()]

    # --------------------------------------------------
    # Observations
    # --------------------------------------------------

    def observe(self, player_id: str) -> Dict:
        """Return role-aware observation for a specific agent."""
        employed_players = [p for p in self.player_ids if p not in self.fired_players]
        current_ciso = self.player_ids[self.current_ciso_idx % len(self.player_ids)]

        obs = {
            "round": self.round,
            "phase": self.current_phase,
            "public_log": list(self.public_log),
            "patch_track": self.patch_track.copy(),
            "fired_players": list(self.fired_players),
            "employed_players": employed_players,
            "current_ciso": current_ciso,
            "nominated_soc_lead": self.nominated_soc_lead,
            "belief": self._latest_belief(player_id),
            "consecutive_failed_councils": self.consecutive_failed_councils,
            "your_role": self.roles[player_id],
            "council_votes": self.council_votes.copy(),  # Include current votes
        }

        # Red asymmetry: Reds know each other
        if self.roles[player_id] in ["red", "apt_leader"]:
            obs["faction"] = "red"
            obs["known_reds"] = [
                pid for pid in employed_players
                if self.roles[pid] in ["red", "apt_leader"] and pid != player_id
            ]
            # Red ciso knows if nominated player is red
            if player_id == current_ciso and self.nominated_soc_lead:
                obs["soc_lead_is_red"] = self.roles[self.nominated_soc_lead] in ["red", "apt_leader"]
        else:
            obs["faction"] = "blue"
            obs["known_reds"] = []
            obs["investigation_results"] = self.investigation_results.copy()

        return obs

    def _observe_all(self) -> Dict:
        """Aggregate observation for all players."""
        return {
            "round": self.round,
            "phase": self.current_phase,
            "public_log": list(self.public_log),
            "patch_track": self.patch_track.copy(),
            "fired_players": list(self.fired_players),
        }

    def _latest_belief(self, player_id: str) -> Dict[str, float]:
        """Get latest belief for a player (excluding self and fired players)."""
        hist = self.belief_histories.get(player_id)
        latest = hist[-1] if hist else {}
        
        # Filter out fired players and self from the belief
        employed = [p for p in self.player_ids if p not in self.fired_players and p != player_id]
        
        if not latest:
            # No belief history, create uniform distribution over employed players
            return {pid: 1.0 / len(employed) if employed else 1.0 for pid in employed}
        
        # Filter existing belief to only include employed players, renormalize
        filtered = {pid: prob for pid, prob in latest.items() if pid in employed}
        total = sum(filtered.values())
        if total > 0:
            filtered = {pid: prob / total for pid, prob in filtered.items()}
        else:
            # If no probability mass remains, redistribute uniformly
            filtered = {pid: 1.0 / len(employed) if employed else 1.0 for pid in employed}
        
        return filtered

    # --------------------------------------------------
    # Step interface
    # --------------------------------------------------

    def step(self, player_id: str, action: Dict):
        """Process a single player action."""
        self._start_turn()

        if not isinstance(action, dict):
            action = {}

        # Record message
        msg = action.get("message")
        if msg:
            self.public_log.append(f"{player_id}: {msg}")

        # Record belief update (excluding self since player knows their own role)
        belief = action.get("belief")
        if belief:
            # Remove belief about self - player already knows their own role
            filtered_belief = {k: v for k, v in belief.items() if k != player_id}
            if filtered_belief:  # Only record if there are other players
                # Renormalize probabilities since we removed one player
                total = sum(filtered_belief.values())
                if total > 0:
                    filtered_belief = {k: v / total for k, v in filtered_belief.items()}
                self.belief_histories.setdefault(player_id, []).append(filtered_belief)

        # Handle action based on current phase
        if self.current_phase == "nomination":
            # CISO nominates SOC Lead
            if player_id == self.player_ids[self.current_ciso_idx % len(self.player_ids)]:
                nomination = action.get("nomination")
                if nomination and nomination in self.player_ids and nomination not in self.fired_players:
                    self.nominated_soc_lead = nomination
                    self.public_log.append(f"CISO {player_id} nominates SOC Lead {nomination}")
                    self.current_phase = "voting"

        elif self.current_phase == "voting":
            # All players vote on council
            vote = action.get("vote")
            if vote in ["yes", "no"]:
                self.council_votes[player_id] = vote

        elif self.current_phase == "legislative":
            # Legislative session: ciso picks patch
            if player_id == self.player_ids[self.current_ciso_idx % len(self.player_ids)]:
                discard = action.get("discard_patch")
                if discard is not None and discard in [0, 1, 2]:
                    self._execute_legislative_session(discard)

        elif self.current_phase == "power":
            # Execute ciso power (investigate, special election, fire)
            power = action.get("power_action")
            if power:
                self._execute_ciso_power(player_id, power)

        self._end_turn()

        obs = self._observe_all()
        reward = 0.0
        info = {}

        if self.done:
            self.game_end_time = time.perf_counter()

        return obs, reward, self.done, info

    # --------------------------------------------------
    # Round management
    # --------------------------------------------------

    def have_all_employed_players_voted(self) -> bool:
        """Check if all employed players have submitted a vote."""
        employed_players = [p for p in self.player_ids if p not in self.fired_players]
        return len(self.council_votes) == len(employed_players) and all(
            vote in ["yes", "no"] for vote in self.council_votes.values()
        )

    def end_round(self):
        """Called after all players have acted to advance to next phase."""
        # Only process if we're actually in a phase that needs processing
        if self.current_phase == "voting":
            # Count and resolve votes
            self._resolve_council_vote()

        elif self.current_phase == "legislative":
            # If we're still in legislative (no action was taken), we need to handle it
            # This can happen if the ciso never submitted a discard action
            # For now, just force a resolution with default discard
            if self.drawn_cards and len(self.drawn_cards) >= 3:
                # Auto-resolve with ciso discarding first card
                self._execute_legislative_session(0)
            # After legislative session, phase will be changed to power or discussion

        elif self.current_phase == "power":
            # Power phase has been handled in step(), just move to next round
            self._advance_to_next_round()

        elif self.current_phase == "discussion":
            # Move to nomination phase
            self.current_phase = "nomination"

        # Check win conditions after phase change
        self._check_win_conditions()

    def _resolve_council_vote(self):
        """Count votes on council nomination."""
        # If there's no nominated soc_lead, treat as automatic rejection
        if not self.nominated_soc_lead:
            self.public_log.append("No soc lead nominated - council rejected")
            self.consecutive_failed_councils += 1
            
            # Record voting history
            self.voting_history.append({
                "round": self.round,
                "ciso": self.player_ids[self.current_ciso_idx % len(self.player_ids)],
                "nominated_soc_lead": None,
                "votes": {},
                "approved": False,
                "result": "No soc_lead nominated"
            })
            
            if self.consecutive_failed_councils >= 3:
                # Enact a random patch
                patch = self.patch_deck.pop() if self.patch_deck else "blue"
                self.public_log.append(f"Three councils rejected: Random {patch} patch applied")
                if patch == "blue":
                    self.patch_track["blue"] += 1
                else:
                    self.patch_track["red"] += 1
                self.consecutive_failed_councils = 0
            
            self.council_votes.clear()
            self.nominated_soc_lead = None
            self._advance_to_next_round()
            return
        
        employed_count = len([p for p in self.player_ids if p not in self.fired_players])
        yes_votes = sum(1 for v in self.council_votes.values() if v == "yes")
        no_votes = sum(1 for v in self.council_votes.values() if v == "no")

        majority_needed = employed_count / 2.0

        if yes_votes > majority_needed:
            # Council approved
            self.public_log.append(f"Council approved: {yes_votes} yes, {no_votes} no")
            self.consecutive_failed_councils = 0
            
            # Record voting history
            self.voting_history.append({
                "round": self.round,
                "ciso": self.player_ids[self.current_ciso_idx % len(self.player_ids)],
                "nominated_soc_lead": self.nominated_soc_lead,
                "votes": self.council_votes.copy(),
                "approved": True,
                "yes_votes": yes_votes,
                "no_votes": no_votes,
                "result": f"{yes_votes} yes, {no_votes} no"
            })

            # Check if APT Leader becomes SOC Lead (red win)
            if self.nominated_soc_lead == self.true_apt_leader_id and self.patch_track["red"] >= 3:
                self.public_log.append(f"Reds win: APT Leader {self.nominated_soc_lead} became SOC Lead!")
                self.done = True
                self.council_votes.clear()
                self.nominated_soc_lead = None
                return

            self.current_phase = "legislative"
            self.council_votes.clear()
            self._draw_patch_cards()
        else:
            # Council rejected
            self.public_log.append(f"Council rejected: {yes_votes} yes, {no_votes} no")
            self.consecutive_failed_councils += 1
            
            # Record voting history
            self.voting_history.append({
                "round": self.round,
                "ciso": self.player_ids[self.current_ciso_idx % len(self.player_ids)],
                "nominated_soc_lead": self.nominated_soc_lead,
                "votes": self.council_votes.copy(),
                "approved": False,
                "yes_votes": yes_votes,
                "no_votes": no_votes,
                "result": f"{yes_votes} yes, {no_votes} no"
            })

            if self.consecutive_failed_councils >= 3:
                # Enact a random patch
                patch = self.patch_deck.pop() if self.patch_deck else "blue"
                self.public_log.append(f"Three councils rejected: Random {patch} patch applied")
                if patch == "blue":
                    self.patch_track["blue"] += 1
                else:
                    self.patch_track["red"] += 1
                self.consecutive_failed_councils = 0

            self.council_votes.clear()
            self.nominated_soc_lead = None
            self._advance_to_next_round()

    def _draw_patch_cards(self):
        """Draw 3 patch cards from deck for legislative session."""
        self._reshuffle_deck_if_needed()
        self.drawn_cards = [self.patch_deck.pop() for _ in range(3)]
        self.public_log.append("Three patches drawn for legislative session")

    def _execute_legislative_session(self, ciso_discard_idx: int):
        """CISO picks which patch to discard."""
        if not hasattr(self, 'drawn_cards') or len(self.drawn_cards) < 3:
            return

        # CISO discards one of 3
        if ciso_discard_idx not in [0, 1, 2]:
            ciso_discard_idx = 0  # Default if invalid
        discarded = self.drawn_cards.pop(ciso_discard_idx)
        self.discard_pile.append(discarded)

        # SOC Lead discards one of remaining 2 (random for now)
        # TODO: Get from SOC Lead action via separate phase
        soc_lead_discard_idx = self.rng.randint(0, 1)
        discarded = self.drawn_cards.pop(soc_lead_discard_idx)
        self.discard_pile.append(discarded)

        # Enact remaining patch
        patch = self.drawn_cards[0]
        if patch == "blue":
            self.patch_track["blue"] += 1
            self.public_log.append("Blue patch applied")
        else:
            self.patch_track["red"] += 1
            self.public_log.append("Red patch applied")

        self.drawn_cards = []

        # Check win conditions BEFORE power phase
        if self.patch_track["blue"] >= 6:
            self.done = True
            self.public_log.append("Blues win: 6 blue patches applied!")
            return
        
        if self.patch_track["red"] >= 6:
            self.done = True
            self.public_log.append("Reds win: 6 red patches applied!")
            return

        # Check if red power should trigger (after 2-5 red patches)
        # Note: At 6, the game ends due to red win, so no power is needed
        if self.patch_track["red"] in [2, 3, 4, 5]:
            self.current_phase = "power"
        else:
            self._advance_to_next_round()

    def _execute_ciso_power(self, player_id: str, power_action: Dict):
        """Execute ciso's special power."""
        current_ciso = self.player_ids[self.current_ciso_idx % len(self.player_ids)]
        if player_id != current_ciso:
            return

        red_count = self.patch_track["red"]
        special_election_used = False

        if red_count == 1:
            # No power
            pass
        elif red_count == 2:
            # Investigate: see one player's role
            target = power_action.get("investigate")
            if target and target in self.player_ids and target not in self.fired_players:
                self.investigation_results[target] = self.roles[target]
                # CISO can lie about result
                message = power_action.get("investigation_claim")
                if message:
                    self.public_log.append(f"CISO investigated {target}: claimed {message}")
        elif red_count == 3:
            # Special election: pick next ciso
            target = power_action.get("special_election")
            if target and target in self.player_ids and target not in self.fired_players:
                idx = self.player_ids.index(target)
                self.current_ciso_idx = idx
                self.public_log.append(f"Special election: {target} is new CISO")
                special_election_used = True
        elif red_count >= 4:
            # Fire: fire a player (4 = first shot, 5 = second, 6 = third)
            target = power_action.get("fire")
            if target and target in self.player_ids and target not in self.fired_players:
                self.fired_players.add(target)
                self.public_log.append(f"CISO fires {target}")

                if target == self.true_apt_leader_id:
                    self.public_log.append("Blues win: APT Leader was fired!")
                    self.done = True
                    return

        self._advance_to_next_round(skip_ciso_increment=special_election_used)

    def _advance_to_next_round(self, skip_ciso_increment=False):
        """Move to next council formation cycle."""
        self.round += 1

        # Advance ciso (unless special election was used)
        if not skip_ciso_increment:
            employed_players = [p for p in self.player_ids if p not in self.fired_players]
            current_ciso = self.player_ids[self.current_ciso_idx % len(self.player_ids)]
            
            # Find next employed ciso
            while current_ciso in self.fired_players:
                self.current_ciso_idx = (self.current_ciso_idx + 1) % len(self.player_ids)
                current_ciso = self.player_ids[self.current_ciso_idx]

            self.current_ciso_idx = (self.current_ciso_idx + 1) % len(self.player_ids)
            while self.player_ids[self.current_ciso_idx % len(self.player_ids)] in self.fired_players:
                self.current_ciso_idx = (self.current_ciso_idx + 1) % len(self.player_ids)

        self.nominated_soc_lead = None
        self.council_votes.clear()
        self.current_phase = "discussion"

        if self.round >= self.max_rounds:
            self.done = True
            self.public_log.append(f"Max rounds ({self.max_rounds}) reached")

    def _check_win_conditions(self):
        """Check if game has ended and who won."""
        # Already checked in _resolve_council_vote and _execute_ciso_power
        # Additional check: all reds fired
        if not self.done:
            reds_employed = [
                p for p in self.player_ids
                if self.roles[p] in ["red", "apt_leader"] and p not in self.fired_players
            ]
            if len(reds_employed) == 0:
                self.public_log.append("Blues win: All reds fired!")
                self.done = True

    # --------------------------------------------------
    # Timing
    # --------------------------------------------------

    def _start_turn(self):
        self._turn_start_time = time.perf_counter()

    def _end_turn(self):
        if self._turn_start_time is not None:
            self.turn_times.append(time.perf_counter() - self._turn_start_time)

    @property
    def total_game_time(self) -> float:
        if self.game_end_time is None:
            return 0.0
        return self.game_end_time - self.game_start_time

    @property
    def avg_turn_time(self) -> float:
        if not self.turn_times:
            return 0.0
        return float(np.mean(self.turn_times))

    # --------------------------------------------------
    # Outcome helpers
    # --------------------------------------------------

    def blues_win(self) -> bool:
        """Check if blues won."""
        if not self.done:
            return False

        # Blues win if:
        # 1. 6 blue patches applied
        # 2. APT Leader was fired
        # 3. All reds fired
        
        if self.patch_track["blue"] >= 6:
            return True

        # Check if APT Leader was fired
        if any("APT Leader was fired" in msg for msg in self.public_log):
            return True

        reds_employed = [
            p for p in self.player_ids
            if self.roles[p] in ["red", "apt_leader"] and p not in self.fired_players
        ]
        if len(reds_employed) == 0:
            return True

        return False

    def reds_win(self) -> bool:
        """Check if reds won."""
        if not self.done:
            return False

        # Reds win if:
        # 1. 6 red patches applied
        # 2. APT Leader became SOC Lead with 3+ patches applied
        
        if self.patch_track["red"] >= 6:
            return True
        
        # Check if APT Leader became SOC Lead (check public log)
        if any("APT Leader" in msg and "SOC Lead" in msg for msg in self.public_log):
            return True
        
        return False

    def get_public_state(self):
        """Return public game state."""
        return {
            "round": self.round,
            "public_log": list(self.public_log),
            "patch_track": self.patch_track.copy(),
            "fired_players": list(self.fired_players),
        }
