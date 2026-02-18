from __future__ import annotations

from datetime import datetime, timedelta
from types import SimpleNamespace

import numpy as np

from polarix_sh.reward import compute_sh_reward
from red_vs_blue.env import RedvsBlueEnv
from red_vs_blue.metrics import (
    average_belief_alignment,
    average_entropy_reduction,
    brier_score,
    cohens_d,
    compute_adjusted_win,
    deception_success,
    entropy,
    holm_bonferroni,
    permutation_test,
    tokens_per_entropy_reduction,
    tokens_per_game,
    wall_clock_time,
)
from red_vs_blue.scoring import APTLeaderOnlyScorer, BlueOnlyScorer, RedvsBlueScorer


class DummyUsage:
    def __init__(self, total_tokens: int):
        self.total_tokens = total_tokens


class DummyResponse:
    def __init__(self, tokens: int | None):
        self.usage = DummyUsage(tokens) if tokens is not None else None


class DummyEnv:
    def __init__(self):
        self.true_apt_leader_id = "p0"
        self.round = 3
        self.num_players = 3
        self.roles = {"p0": "apt_leader", "p1": "blue", "p2": "blue"}
        self.belief_histories = {
            "p0": [
                {"p1": 0.6, "p2": 0.4},
                {"p1": 0.8, "p2": 0.2},
            ],
            "p1": [
                {"p0": 0.2, "p2": 0.8},
                {"p0": 0.7, "p2": 0.3},
            ],
            "p2": [
                {"p0": 0.3, "p1": 0.7},
                {"p0": 0.6, "p1": 0.4},
            ],
        }

    def blues_win(self):
        return True


def test_metrics_edge_cases_and_stats_helpers():
    assert entropy({}) == 0.0
    assert brier_score({}, "p0") == 0.0
    assert average_entropy_reduction([{"p0": 1.0}]) == 0.0
    assert average_belief_alignment([{"p0": 1.0}], "p0") == 0.0
    assert deception_success([{"p0": 1.0}], "p0") == 0.0

    np.random.seed(123)
    p = permutation_test(np.array([1.0, 2.0]), np.array([3.0, 4.0]), n_perm=200)
    assert 0.0 <= p <= 1.0

    d = cohens_d(np.array([1.0, 2.0, 3.0]), np.array([2.0, 3.0, 4.0]))
    assert d < 0

    corrected = holm_bonferroni({"h1": 0.001, "h2": 0.04, "h3": 0.2}, alpha=0.05)
    assert corrected["h1"]["reject_null"] is True
    assert corrected["h2"]["reject_null"] is False
    assert corrected["h3"]["threshold"] is None


def test_metrics_resource_and_time_helpers():
    start = datetime(2026, 2, 17, 10, 0, 0)
    end = start + timedelta(seconds=12)
    assert wall_clock_time(start, end) == 12

    responses = [DummyResponse(10), DummyResponse(None), DummyResponse(5)]
    assert tokens_per_game(responses) == 15

    assert tokens_per_entropy_reduction(100, 0) == float("inf")
    assert tokens_per_entropy_reduction(100, -1) == float("inf")
    assert tokens_per_entropy_reduction(100, 2.0) == 50

    assert compute_adjusted_win(1.0, 0) == 1.0
    assert compute_adjusted_win(2.0, 4) == 0.5


def test_reward_branches_and_role_rewards():
    env = SimpleNamespace(
        winner="blue",
        num_blue_policies_enacted_this_step=lambda: 2,
        num_red_policies_enacted_this_step=lambda: 1,
        entropy_reduction=lambda: 0.5,
        correct_accusations=lambda: 1,
        incorrect_accusations=lambda: 2,
    )
    reward, role_rewards = compute_sh_reward(env, done=True)

    assert reward > 0
    assert "blue_role_reward" in role_rewards
    assert "red_role_reward" in role_rewards

    env.winner = "red"
    reward_red, _ = compute_sh_reward(env, done=True)
    assert reward_red < reward


def test_scorers_produce_expected_metadata():
    state = SimpleNamespace(env=DummyEnv())

    main_score = RedvsBlueScorer().score(state)
    assert main_score.value == 1
    for key in [
        "blues_win",
        "rounds_played",
        "num_players",
        "avg_entropy_reduction",
        "avg_belief_alignment",
        "avg_brier",
        "apt_leader_deception",
    ]:
        assert key in main_score.metadata

    blue_score = BlueOnlyScorer().score(state)
    assert blue_score.metadata["role"] == "blue"

    apt_score = APTLeaderOnlyScorer().score(state)
    assert apt_score.metadata["role"] == "apt_leader"


def test_env_observe_legislative_visibility_and_belief_fallbacks():
    env = RedvsBlueEnv(num_players=5, seed=11)

    current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
    env.current_phase = "legislative_ciso"
    env.drawn_cards = ["red", "blue", "red"]

    ciso_obs = env.observe(current_ciso)
    assert ciso_obs["drawn_cards"] == ["red", "blue", "red"]

    non_ciso = next(pid for pid in env.player_ids if pid != current_ciso)
    non_ciso_obs = env.observe(non_ciso)
    assert "drawn_cards" not in non_ciso_obs

    env.nominated_soc_lead = non_ciso
    env.current_phase = "legislative_soc"
    env.drawn_cards = ["red", "blue"]
    soc_obs = env.observe(non_ciso)
    assert soc_obs["drawn_cards"] == ["red", "blue"]

    env.fired_players = {"p1", "p2"}
    env.belief_histories["p0"] = []
    belief = env._latest_belief("p0")
    assert set(belief.keys()).issubset(set(env.player_ids) - {"p0", "p1", "p2"})

    env.belief_histories["p3"] = [{"p0": 0.0, "p4": 0.0}]
    renorm = env._latest_belief("p3")
    assert abs(sum(renorm.values()) - 1.0) < 1e-9


def test_env_end_round_auto_resolves_legislative_stages():
    env = RedvsBlueEnv(num_players=5, seed=9)

    env.current_phase = "legislative_ciso"
    env.drawn_cards = ["red", "blue", "blue"]
    env.end_round()
    assert env.current_phase == "legislative_soc"

    env.current_phase = "legislative_soc"
    env.drawn_cards = ["blue", "blue"]
    prev_round = env.round
    env.end_round()
    assert env.round == prev_round + 1


def test_env_draw_and_power_and_win_helpers():
    env = RedvsBlueEnv(num_players=5, seed=5)

    env.patch_deck = ["red", "blue"]
    env.discard_pile = []
    env._draw_patch_cards()
    assert len(env.drawn_cards) == 2
    assert any("Only 2 patches available" in msg for msg in env.public_log)

    current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
    non_ciso = next(pid for pid in env.player_ids if pid != current_ciso)
    before_fired = set(env.fired_players)
    env.patch_track["red"] = 4
    env._execute_ciso_power(non_ciso, {"fire": env.player_ids[0]})
    assert env.fired_players == before_fired

    env.done = True
    env.patch_track["red"] = 0
    env.public_log.append("Reds win: APT Leader p0 became SOC Lead!")
    assert env.reds_win() is True


def test_env_resolve_vote_without_nomination_and_three_failures():
    env = RedvsBlueEnv(num_players=5, seed=13)
    env.current_phase = "voting"
    env.nominated_soc_lead = None
    env.consecutive_failed_councils = 2
    env.patch_deck = ["red"]

    env.end_round()

    assert env.consecutive_failed_councils == 0
    assert env.patch_track["red"] == 1
    assert env.current_phase == "discussion"


def test_env_power_investigation_and_special_election_paths():
    env = RedvsBlueEnv(num_players=5, seed=21)
    current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]

    env.patch_track["red"] = 2
    target = next(pid for pid in env.player_ids if pid != current_ciso)
    env._execute_ciso_power(current_ciso, {"investigate": target, "investigation_claim": "blue"})
    assert target in env.investigation_results
    assert any("claimed blue" in msg for msg in env.public_log)

    current_ciso = env.player_ids[env.current_ciso_idx % len(env.player_ids)]
    env.patch_track["red"] = 3
    next_target = next(pid for pid in env.player_ids if pid != target)
    env._execute_ciso_power(current_ciso, {"special_election": next_target})
    assert env.player_ids[env.current_ciso_idx] == next_target


def test_env_backward_compat_legislative_and_power_round_advance():
    env = RedvsBlueEnv(num_players=5, seed=17)
    env.current_phase = "legislative"
    env.drawn_cards = ["blue", "red", "blue"]
    env.end_round()
    assert env.current_phase in {"discussion", "power"}

    env.current_phase = "power"
    prev_round = env.round
    env.end_round()
    assert env.round == prev_round + 1
