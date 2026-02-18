from __future__ import annotations

import asyncio
from types import SimpleNamespace

from inspect_ai.scorer import Score

import red_vs_blue.task as task_module
from red_vs_blue.agents import RedvsBlueAgent, create_agents, normalize, softmax
from red_vs_blue.task import (
    _create_dataset,
    _format_public_log_for_llm,
    _run_game_loop,
    avg_rounds_played_metric,
    blue_win_metric,
    generate_executive_summary_with_llm,
    red_vs_blue_task,
    red_win_metric,
    save_executive_summary,
)


class _AsyncGenModel:
    def __init__(self, text: str = "{}"):
        self._text = text

    async def generate(self, input, config):
        completion = self._text if self._text else None
        return SimpleNamespace(completion=completion, choices=[])


class _RaisingModel:
    async def generate(self, input, config):
        raise RuntimeError("model failure")


class _ChoicesOnlyModel:
    async def generate(self, input, config):
        choice = SimpleNamespace(message=SimpleNamespace(text="from choices"))
        return SimpleNamespace(completion=None, choices=[choice])


class _PhaseAgent:
    def __init__(self, player_id: str, async_mode: bool = False):
        self.player_id = player_id
        self.role = "blue"
        self.async_mode = async_mode

    def act(self, observation):
        phase = observation.get("phase")
        if phase == "voting":
            action = {"vote": "yes", "belief": observation.get("belief", {})}
        elif phase == "nomination":
            action = {
                "nomination": observation.get("employed_players", [self.player_id])[-1],
                "belief": observation.get("belief", {}),
            }
        elif phase in {"legislative_ciso", "legislative_soc"}:
            action = {"discard_patch": 0, "belief": observation.get("belief", {})}
        elif phase == "power":
            action = {"power_action": {"investigate": "p1"}, "belief": observation.get("belief", {})}
        else:
            action = {"message": "ok", "belief": observation.get("belief", {})}

        if self.async_mode:
            async def _inner():
                return action
            return _inner()

        return action


class _FakeEnv:
    def __init__(self):
        self.player_ids = ["p0", "p1"]
        self.fired_players = set()
        self.patch_track = {"blue": 1, "red": 1}
        self.current_ciso_idx = 0
        self.nominated_soc_lead = "p1"
        self.council_votes = {}
        self.public_log = []
        self.round = 0
        self.done = False
        self.voting_history = []
        self.roles = {"p0": "blue", "p1": "apt_leader"}
        self.true_apt_leader_id = "p1"
        self.belief_histories = {
            "p0": [{"p1": 1.0}],
            "p1": [{"p0": 1.0}],
        }
        self._phases = [
            "discussion",
            "nomination",
            "voting",
            "legislative_ciso",
            "legislative_soc",
            "power",
            "unknown",
        ]
        self._phase_idx = 0
        self.current_phase = self._phases[self._phase_idx]

    def observe(self, player_id):
        other = [p for p in self.player_ids if p != player_id]
        return {
            "round": self.round,
            "phase": self.current_phase,
            "public_log": list(self.public_log),
            "patch_track": dict(self.patch_track),
            "fired_players": list(self.fired_players),
            "employed_players": [p for p in self.player_ids if p not in self.fired_players],
            "current_ciso": self.player_ids[self.current_ciso_idx],
            "nominated_soc_lead": self.nominated_soc_lead,
            "belief": {other[0]: 1.0} if other else {},
            "council_votes": dict(self.council_votes),
        }

    def step(self, player_id, action):
        if self.current_phase == "voting":
            if player_id == "p0":
                self.council_votes[player_id] = action.get("vote", "no")
        elif self.current_phase == "nomination":
            self.nominated_soc_lead = action.get("nomination", self.nominated_soc_lead)
            self.current_phase = "voting"
        self.public_log.append(f"{player_id}:{self.current_phase}")

    def have_all_employed_players_voted(self):
        return False

    def end_round(self):
        self.round += 1
        self._phase_idx += 1
        if self._phase_idx >= len(self._phases):
            self.done = True
            return
        self.current_phase = self._phases[self._phase_idx]

    def blues_win(self):
        return True

    def reds_win(self):
        return False


class _FakeEnvAllVotes(_FakeEnv):
    def step(self, player_id, action):
        if self.current_phase == "voting":
            self.council_votes[player_id] = action.get("vote", "yes")
        elif self.current_phase == "nomination":
            self.nominated_soc_lead = action.get("nomination", self.nominated_soc_lead)
        self.public_log.append(f"{player_id}:{self.current_phase}")

    def have_all_employed_players_voted(self):
        return len(self.council_votes) == len(self.player_ids)


def test_agents_math_and_factory_paths():
    assert normalize({"a": 0.0, "b": 0.0}) == {"a": 0.5, "b": 0.5}

    probs = softmax({"a": 1.0, "b": 2.0}, temperature=0.7)
    assert abs(sum(probs.values()) - 1.0) < 1e-9
    assert probs["b"] > probs["a"]

    agents = create_agents(model=SimpleNamespace(), player_ids=["p0", "p1", "p2"])
    assert len(agents) == 3
    assert all(agent.role is None for agent in agents)


def test_agent_prompt_parse_and_fallback_paths():
    agent = RedvsBlueAgent(
        model=SimpleNamespace(),
        player_id="p0",
        role="blue",
        all_players=["p0", "p1", "p2"],
    )

    base_obs = {
        "round": 1,
        "phase": "discussion",
        "public_log": ["p1: hi"],
        "patch_track": {"blue": 1, "red": 2},
        "fired_players": [],
        "employed_players": ["p0", "p1", "p2"],
        "current_ciso": "p0",
        "nominated_soc_lead": "p1",
        "belief": {"p1": 0.5, "p2": 0.5},
        "known_reds": ["p2"],
        "council_votes": {"p0": "yes"},
    }

    assert "You are a BLUE" in agent._role_prompt(base_obs)
    agent.role = "red"
    assert "You are a RED" in agent._role_prompt(base_obs)
    agent.role = "apt_leader"
    assert "You are APT LEADER" in agent._role_prompt(base_obs)

    for phase in [
        "discussion",
        "nomination",
        "voting",
        "legislative_ciso",
        "legislative_soc",
        "power",
    ]:
        obs = dict(base_obs)
        obs["phase"] = phase
        if phase == "legislative_ciso":
            obs["drawn_cards"] = ["red", "blue", "red"]
        if phase == "legislative_soc":
            obs["drawn_cards"] = ["blue", "red"]
        agent._user_prompt(obs)

    with_nom = """{"message":"x","nomination":"p1","belief":{"p1":0.6,"p2":0.4}}"""
    parsed_nom = agent._parse_json_response(with_nom, {"p1": 0.5, "p2": 0.5}, "nomination", ["p1", "p2"])
    assert parsed_nom["nomination"] == "p1"

    with_vote = "prefix {\"vote\":\"yes\",\"belief\":{\"p1\":1}} suffix"
    parsed_vote = agent._parse_json_response(with_vote, {"p1": 1.0}, "voting", ["p1"])
    assert parsed_vote["vote"] == "yes"

    with_discussion = """{"message":"hi","accusation":"p2","belief":{"p1":0.2,"p2":0.8}}"""
    parsed_disc = agent._parse_json_response(with_discussion, {"p1": 0.5, "p2": 0.5}, "discussion", ["p1", "p2"])
    assert parsed_disc["accusation"] == "p2"

    with_leg = """{"discard_patch":2,"belief":{"p1":1}}"""
    parsed_leg = agent._parse_json_response(with_leg, {"p1": 1.0}, "legislative", ["p1"])
    assert parsed_leg["discard_patch"] == 2

    with_power = """{"power_action":{"investigate":"p1"}}"""
    parsed_pow = agent._parse_json_response(with_power, {"p1": 1.0}, "power", ["p1"])
    assert parsed_pow["power_action"]["investigate"] == "p1"

    bad = agent._parse_json_response("not-json", {"p1": 1.0}, "voting", ["p1"])
    assert bad["vote"] == "no"

    no_markers = agent._parse_json_response("plain text only", {"p1": 1.0}, "discussion", ["p1"])
    assert no_markers["message"] == ""

    fallback = agent._generate_fallback_action({"phase": "discussion"}, {"p1": 1.0}, ["p1"])
    assert fallback["message"]
    fallback_vote = agent._generate_fallback_action({"phase": "voting"}, {"p1": 1.0}, ["p1"])
    assert fallback_vote["vote"] == "no"

    agent.role = "unknown"
    try:
        agent._role_prompt(base_obs)
        assert False, "Expected unknown role to raise"
    except ValueError:
        pass
    agent.role = "blue"

    msg, belief, accusation, vote = agent._parse_response(
        "Public message: hello\naccuse p1\np1: 0.9, p2: 0.1",
        {"p1": 0.5, "p2": 0.5},
        phase="discussion",
        employed_players=["p1", "p2"],
    )
    assert msg
    assert abs(sum(belief.values()) - 1.0) < 1e-9
    assert accusation == "p1"
    assert vote is None

    msg2, _, _, vote2 = agent._parse_response(
        "Vote: yes",
        {"p1": 1.0},
        phase="vote",
        employed_players=["p1"],
    )
    assert msg2
    assert vote2 == "yes"


def test_agent_prompt_waiting_and_power_variants():
    agent = RedvsBlueAgent(
        model=SimpleNamespace(),
        player_id="p9",
        role="blue",
        all_players=["p0", "p1", "p9"],
    )
    obs = {
        "round": 2,
        "phase": "legislative_ciso",
        "public_log": [],
        "patch_track": {"blue": 1, "red": 1},
        "fired_players": [],
        "employed_players": ["p0", "p1", "p9"],
        "current_ciso": "p0",
        "nominated_soc_lead": "p1",
        "belief": {"p0": 0.5, "p1": 0.5},
        "council_votes": {},
    }
    assert "Waiting for CISO" in agent._user_prompt(obs)

    obs["phase"] = "legislative_soc"
    assert "Waiting for SOC Lead" in agent._user_prompt(obs)

    obs["phase"] = "power"
    assert "Waiting for CISO" in agent._user_prompt(obs)

    ciso_agent = RedvsBlueAgent(
        model=SimpleNamespace(),
        player_id="p0",
        role="blue",
        all_players=["p0", "p1", "p2"],
    )
    for red_count, expected in [(1, "No ciso power"), (2, "INVESTIGATION POWER"), (3, "SPECIAL ELECTION POWER"), (4, "FIRE POWER")]:
        power_obs = dict(obs)
        power_obs["current_ciso"] = "p0"
        power_obs["patch_track"] = {"blue": 1, "red": red_count}
        assert expected in ciso_agent._user_prompt(power_obs)

    ciso_leg_obs = dict(obs)
    ciso_leg_obs["phase"] = "legislative_ciso"
    ciso_leg_obs["current_ciso"] = "p0"
    assert "You have been dealt 3 patches" in ciso_agent._user_prompt(ciso_leg_obs)

    soc_agent = RedvsBlueAgent(
        model=SimpleNamespace(),
        player_id="p1",
        role="blue",
        all_players=["p0", "p1", "p2"],
    )
    soc_obs = dict(obs)
    soc_obs["phase"] = "legislative_soc"
    soc_obs["nominated_soc_lead"] = "p1"
    assert "The CISO has given you 2 patches" in soc_agent._user_prompt(soc_obs)


def test_agent_json_parse_error_and_value_coercion_paths():
    agent = RedvsBlueAgent(
        model=SimpleNamespace(),
        player_id="p0",
        role="blue",
        all_players=["p0", "p1", "p2"],
    )

    parse_error = agent._parse_json_response("{bad json}", {"p1": 1.0}, "discussion", ["p1"])
    assert parse_error["belief"] == {"p1": 1.0}

    invalid_belief = agent._parse_json_response(
        '{"belief": {"p1": "oops", "p2": 2}}',
        {"p1": 0.5, "p2": 0.5},
        "discussion",
        ["p1", "p2"],
    )
    assert abs(sum(invalid_belief["belief"].values()) - 1.0) < 1e-9

    vote_no = agent._parse_response("vote no", {"p1": 1.0}, phase="vote", employed_players=["p1"])
    assert vote_no[3] == "no"

    vote_yes_prioritized = agent._parse_response("yes and no appear, but vote yes", {"p1": 1.0}, phase="vote", employed_players=["p1"])
    assert vote_yes_prioritized[3] == "yes"


def test_agent_act_success_and_empty_response():
    agent = RedvsBlueAgent(
        model=_AsyncGenModel('{"message":"ok","belief":{"p1":1.0}}'),
        player_id="p0",
        role="blue",
        all_players=["p0", "p1"],
    )
    obs = {
        "round": 1,
        "phase": "discussion",
        "public_log": [],
        "patch_track": {"blue": 0, "red": 0},
        "fired_players": [],
        "employed_players": ["p0", "p1"],
        "current_ciso": "p0",
        "nominated_soc_lead": None,
        "belief": {"p1": 1.0},
        "council_votes": {},
    }
    action = asyncio.run(agent.act(obs))
    assert action["message"] == "ok"

    empty_agent = RedvsBlueAgent(
        model=_AsyncGenModel(""),
        player_id="p0",
        role="blue",
        all_players=["p0", "p1"],
    )
    fallback_action = asyncio.run(empty_agent.act(obs))
    assert fallback_action["message"] == "I'm observing the situation."


def test_task_metrics_and_helpers_paths():
    ss = [
        SimpleNamespace(score=SimpleNamespace(value=1, metadata={"rounds_played": 2})),
        SimpleNamespace(score=SimpleNamespace(value=0, metadata={"rounds_played": 4})),
        SimpleNamespace(score=None),
    ]
    assert blue_win_metric(ss) == 1 / 3
    assert red_win_metric(ss) == 1 / 3
    assert avg_rounds_played_metric(ss) == 2.0
    assert blue_win_metric([]) == 0.0

    long_log = ["event"] * 1000
    formatted = _format_public_log_for_llm(long_log, max_length=120)
    assert "earlier events" in formatted
    assert _format_public_log_for_llm([]) == "No public events recorded."

    ds = _create_dataset(num_games=2, num_players=5, max_rounds=7, seed_base=10)
    assert len(ds) == 2
    assert ds[1].metadata["seed"] == 11


def test_run_game_loop_branch_coverage():
    env = _FakeEnv()
    agents = [_PhaseAgent("p0", async_mode=True), _PhaseAgent("p1", async_mode=False)]

    result = asyncio.run(_run_game_loop(env, agents))
    assert result["blues_win"] is True
    assert result["rounds"] >= 1
    assert "belief_histories" in result


def test_run_game_loop_discussion_progress_and_all_votes_path():
    env = _FakeEnvAllVotes()
    env.round = 1
    agents = [_PhaseAgent("p0", async_mode=False), _PhaseAgent("p1", async_mode=True)]

    result = asyncio.run(_run_game_loop(env, agents))
    assert result["reds_win"] is False
    assert all("Warning: Missing votes" not in line for line in env.public_log)


async def _build_summary(model, metadata):
    env = SimpleNamespace()
    return await generate_executive_summary_with_llm(model, metadata, env)


def test_llm_summary_success_and_error_paths(tmp_path):
    metadata = {
        "num_players": 3,
        "rounds_played": 4,
        "blues_win": True,
        "reds_win": False,
        "avg_entropy_reduction": 0.2,
        "avg_belief_alignment": 0.3,
        "avg_brier": 0.6,
        "apt_leader_deception": 0.1,
        "public_log": ["p0: hello", "p1: vote yes"],
        "patch_track": {"blue": 3, "red": 2},
        "fired_players": ["p2"],
        "roles": {"p0": "blue", "p1": "red", "p2": "apt_leader"},
        "true_apt_leader": "p2",
        "belief_histories": {"p0": {"p1": 0.7, "p2": 0.3}},
    }

    summary = asyncio.run(_build_summary(_AsyncGenModel("Narrative text"), metadata))
    assert "Red vs. Blue Executive Summary" in summary

    failed_summary = asyncio.run(_build_summary(_RaisingModel(), metadata))
    assert "LLM narrative generation failed" in failed_summary

    path = asyncio.run(save_executive_summary(_AsyncGenModel("Narrative text"), metadata, SimpleNamespace(), output_dir=str(tmp_path)))
    assert path.endswith("executive_summary.md")


def test_llm_summary_draw_and_red_dominance_branches():
    metadata = {
        "num_players": 5,
        "rounds_played": 10,
        "blues_win": False,
        "reds_win": False,
        "avg_entropy_reduction": 0.1,
        "avg_belief_alignment": 0.1,
        "avg_brier": 0.9,
        "apt_leader_deception": 0.8,
        "public_log": ["x"] * 400,
        "patch_track": {"blue": 0, "red": 6},
        "fired_players": [],
        "roles": {"p0": "blue", "p1": "red", "p2": "blue", "p3": "red", "p4": "apt_leader"},
        "true_apt_leader": "p4",
        "belief_histories": {},
    }
    summary = asyncio.run(_build_summary(_AsyncGenModel("Narrative text"), metadata))
    assert "GAME TIMEOUT" in summary
    assert "Red Patch Dominance" in summary
    assert "No Eliminations" in summary


def test_llm_summary_choices_path_and_blue_dominance_branch():
    metadata = {
        "num_players": 5,
        "rounds_played": 3,
        "blues_win": True,
        "reds_win": False,
        "avg_entropy_reduction": 0.7,
        "avg_belief_alignment": 0.25,
        "avg_brier": 0.2,
        "apt_leader_deception": 0.2,
        "public_log": ["p0: x"],
        "patch_track": {"blue": 6, "red": 1},
        "fired_players": ["p4"],
        "roles": {"p0": "blue", "p1": "red", "p2": "blue", "p3": "red", "p4": "apt_leader"},
        "true_apt_leader": "p4",
        "belief_histories": {"p0": {"p1": 0.6, "p4": 0.4}},
    }
    summary = asyncio.run(_build_summary(_ChoicesOnlyModel(), metadata))
    assert "Blue Patch Dominance" in summary
    assert "Moderate Belief Tracking" in summary


def test_task_solver_and_scorer_paths(monkeypatch, tmp_path):
    class DummyModel:
        async def generate(self, input, config):
            return SimpleNamespace(completion="ok")

    async def fake_save_summary(model, metadata, env, output_dir="results"):
        output = tmp_path / "executive_summary.md"
        output.write_text("ok", encoding="utf-8")
        return str(output)

    async def fake_run_game(env, agents):
        return {
            "blues_win": True,
            "reds_win": False,
            "true_apt_leader": "p1",
            "rounds": 2,
            "avg_entropy": 0.1,
            "avg_alignment": 0.2,
            "avg_brier": 0.3,
            "apt_leader_deception": 0.4,
            "public_log": ["x"],
            "voting_history": [],
            "roles": {"p0": "blue", "p1": "apt_leader"},
            "fired_players": [],
            "patch_track": {"blue": 2, "red": 1},
            "belief_histories": {"p0": {"p1": 1.0}},
        }

    monkeypatch.setattr(task_module, "get_model", lambda _x: DummyModel())
    monkeypatch.setattr(task_module, "save_executive_summary", fake_save_summary)
    monkeypatch.setattr(task_module, "_run_game_loop", fake_run_game)

    def fake_create_agents(model, player_ids):
        return [SimpleNamespace(player_id=pid, role=None) for pid in player_ids]

    monkeypatch.setattr(task_module, "create_agents", fake_create_agents)

    task_obj = red_vs_blue_task(num_games=1, num_players=5, max_rounds=3, seed=77)

    state = SimpleNamespace(metadata=task_obj.dataset[0].metadata, output=None)
    solved = asyncio.run(task_obj.solver(state, None))
    assert "game_results" in solved.output

    score_fn = task_obj.scorer[0]
    score = asyncio.run(score_fn(solved, ""))
    assert isinstance(score, Score)
    assert score.value == 1


def test_task_scorer_save_summary_failure_path(monkeypatch):
    class DummyModel:
        async def generate(self, input, config):
            return SimpleNamespace(completion="ok")

    async def fake_run_game(env, agents):
        return {"blues_win": False, "reds_win": True, "rounds": 1}

    async def failing_save_summary(model, metadata, env, output_dir="results"):
        raise RuntimeError("save failed")

    monkeypatch.setattr(task_module, "get_model", lambda _x: DummyModel())
    monkeypatch.setattr(task_module, "_run_game_loop", fake_run_game)
    monkeypatch.setattr(task_module, "save_executive_summary", failing_save_summary)
    monkeypatch.setattr(task_module, "create_agents", lambda model, player_ids: [SimpleNamespace(player_id=pid, role=None) for pid in player_ids])

    task_obj = red_vs_blue_task(num_games=1, num_players=5, max_rounds=2, seed=5)
    state = SimpleNamespace(metadata=task_obj.dataset[0].metadata, output=None)
    solved = asyncio.run(task_obj.solver(state, None))
    score = asyncio.run(task_obj.scorer[0](solved, ""))
    assert score.value == 0
