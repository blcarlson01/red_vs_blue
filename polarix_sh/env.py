# polarix_sh/env.py

from dataclasses import dataclass
import math
import random
from red_vs_blue.env import RedvsBlueEnv
from polarix_sh.reward import compute_sh_reward


@dataclass
class SHObservation:
    public_state: dict
    belief_state: dict
    transcript: list


class PolarixRedvsBlueEnv:
    """
    Polarix-compatible wrapper around Red vs Blue.
    """

    def __init__(self, num_players=5, seed=None, max_rounds=10):
        self.num_players = num_players
        self.seed = seed
        self.max_rounds = max_rounds
        self.rng = random.Random(seed)

        self.env = RedvsBlueEnv(num_players=num_players, max_rounds=max_rounds, seed=seed)
        self.round = 0
        self.winner = None
        self._last_blue_policies = 0
        self._last_red_policies = 0
        self._last_entropy_reduction = 0.0
        self._last_correct_accusations = 0
        self._last_incorrect_accusations = 0

    def reset(self):
        self.env.reset()
        self.round = 0
        self.winner = None
        self._last_blue_policies = 0
        self._last_red_policies = 0
        self._last_entropy_reduction = 0.0
        self._last_correct_accusations = 0
        self._last_incorrect_accusations = 0
        return self._observe()

    def _belief_entropy(self) -> float:
        entropies = []
        for pid in self.env.player_ids:
            belief = self.env._latest_belief(pid)
            if not belief:
                continue
            entropy = 0.0
            for p in belief.values():
                if p > 0:
                    entropy -= p * math.log(p)
            entropies.append(entropy)
        return float(sum(entropies) / len(entropies)) if entropies else 0.0

    def get_beliefs(self):
        return {pid: self.env._latest_belief(pid) for pid in self.env.player_ids}

    def _observe(self):
        return SHObservation(
            public_state=self.env.get_public_state(),
            belief_state=self.get_beliefs(),
            transcript=list(self.env.public_log),
        )

    def _build_phase_action(self, phase: str, actions: dict):
        if phase == "discussion":
            for pid in self.env.player_ids:
                if pid in self.env.fired_players:
                    continue
                act = actions.get(pid, {}) or {}
                message = act.get("message", "")
                self.env.step(pid, {"message": message})
            return

        if phase == "nomination":
            ciso = self.env.player_ids[self.env.current_ciso_idx % len(self.env.player_ids)]
            act = actions.get(ciso, {}) or {}
            nomination = act.get("nomination")
            if nomination is None:
                candidates = [p for p in self.env.player_ids if p != ciso and p not in self.env.fired_players]
                if candidates:
                    nomination = candidates[0]
            payload = {"nomination": nomination} if nomination is not None else {}
            self.env.step(ciso, payload)
            return

        if phase == "voting":
            for pid in self.env.player_ids:
                if pid in self.env.fired_players:
                    continue
                act = actions.get(pid, {}) or {}
                vote = act.get("vote", "yes")
                if isinstance(vote, bool):
                    vote = "yes" if vote else "no"
                if vote not in ["yes", "no"]:
                    vote = "yes"
                self.env.step(pid, {"vote": vote})
            return

        if phase in ["legislative", "legislative_ciso"]:
            ciso = self.env.player_ids[self.env.current_ciso_idx % len(self.env.player_ids)]
            act = actions.get(ciso, {}) or {}
            discard = act.get("discard_patch", 0)
            if not isinstance(discard, int):
                discard = 0
            self.env.step(ciso, {"discard_patch": discard})
            return

        if phase == "legislative_soc":
            soc = self.env.nominated_soc_lead
            if soc is None:
                return
            act = actions.get(soc, {}) or {}
            discard = act.get("discard_patch", 0)
            if not isinstance(discard, int):
                discard = 0
            self.env.step(soc, {"discard_patch": discard})
            return

        if phase == "power":
            ciso = self.env.player_ids[self.env.current_ciso_idx % len(self.env.player_ids)]
            act = actions.get(ciso, {}) or {}
            power_action = act.get("power_action")
            if power_action is None:
                targets = [p for p in self.env.player_ids if p != ciso and p not in self.env.fired_players]
                red_count = self.env.patch_track["red"]
                if targets and red_count >= 4:
                    power_action = {"fire": targets[0]}
                elif targets and red_count == 3:
                    power_action = {"special_election": targets[0]}
                elif targets and red_count == 2:
                    power_action = {"investigate": targets[0]}
                else:
                    power_action = {}
            self.env.step(ciso, {"power_action": power_action})

    def step(self, actions: dict):
        """
        actions = {
            player_id: {
                "message": str,
                "vote": bool,
                "accuse": int | None
            }
        }
        """

        self.round += 1

        pre_blue = self.env.patch_track["blue"]
        pre_red = self.env.patch_track["red"]
        pre_entropy = self._belief_entropy()

        phase = self.env.current_phase
        self._build_phase_action(phase, actions)
        self.env.end_round()

        post_blue = self.env.patch_track["blue"]
        post_red = self.env.patch_track["red"]
        post_entropy = self._belief_entropy()

        self._last_blue_policies = max(0, post_blue - pre_blue)
        self._last_red_policies = max(0, post_red - pre_red)
        self._last_entropy_reduction = max(0.0, pre_entropy - post_entropy)
        self._last_correct_accusations = 0
        self._last_incorrect_accusations = 0

        done = self.env.done or self.round >= self.max_rounds

        if self.env.blues_win():
            self.winner = "blue"
        elif self.env.reds_win():
            self.winner = "red"
        elif done:
            self.winner = "draw"
        else:
            self.winner = None

        reward, role_rewards = compute_sh_reward(self, done)

        obs = self._observe()

        info = {
            "round": self.round,
            "winner": self.winner,
            "entropy": post_entropy,
            **role_rewards,
        }

        return obs, reward, done, info

    def num_blue_policies_enacted_this_step(self):
        return self._last_blue_policies

    def num_red_policies_enacted_this_step(self):
        return self._last_red_policies

    def entropy_reduction(self):
        return self._last_entropy_reduction

    def correct_accusations(self):
        return self._last_correct_accusations

    def incorrect_accusations(self):
        return self._last_incorrect_accusations