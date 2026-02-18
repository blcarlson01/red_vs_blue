from __future__ import annotations

import asyncio
import json
import zipfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

import red_vs_blue.analysis.aggregate_results as ar
import red_vs_blue.analysis.convert_inspect_to_polarix as c2p
import red_vs_blue.analysis.cross_analysis_findings as caf


def _run(coro):
    return asyncio.run(coro)


def _make_eval(path: Path, sample_payloads: list[str]):
    with zipfile.ZipFile(path, "w") as zf:
        for i, payload in enumerate(sample_payloads, 1):
            zf.writestr(f"samples/{i:03d}.json", payload)
        zf.writestr("other/ignored.json", json.dumps({"ignored": True}))


def test_aggregate_results_loading_collect_and_aggregate(tmp_path):
    eval_path = tmp_path / "one.eval"
    _make_eval(
        eval_path,
        [
            json.dumps(
                {
                    "scores": {
                        "red_vs_blue_scorer": {
                            "value": 1,
                            "metadata": {
                                "model": "m1",
                                "num_players": 5,
                                "avg_entropy_reduction": 0.1,
                                "avg_belief_alignment": 0.2,
                                "avg_brier": 0.3,
                                "apt_leader_deception": 0.4,
                                "rounds_played": 6,
                                "public_log": ["x"],
                            },
                        }
                    },
                    "model_usage": {"foo": {"total_tokens": 100, "input_tokens": 60, "output_tokens": 40}},
                }
            ),
            "not-json",
        ],
    )

    loaded = ar.load_eval_file(eval_path)
    assert len(loaded) == 1

    df = ar.collect_results(tmp_path)
    assert not df.empty
    assert "tokens_used" in df.columns
    assert df.iloc[0]["model"] == "m1"

    agg = ar.aggregate(df)
    assert "win_rate" in agg.columns
    assert "avg_tokens_used" in agg.columns


def test_aggregate_results_branches_and_main(tmp_path, capsys):
    # no eval files -> error branch
    try:
        ar.collect_results(tmp_path)
        assert False, "expected FileNotFoundError"
    except FileNotFoundError:
        pass

    # aggregate default branches without model/num_players
    df = pd.DataFrame(
        [
            {
                "value": 1,
                "avg_entropy_reduction": 0.1,
                "avg_belief_alignment": 0.2,
                "avg_brier": 0.3,
                "apt_leader_deception": 0.4,
                "rounds_played": 5,
            },
            {
                "value": 0,
                "avg_entropy_reduction": 0.2,
                "avg_belief_alignment": 0.1,
                "avg_brier": 0.4,
                "apt_leader_deception": 0.3,
                "rounds_played": 6,
            },
        ]
    )
    agg = ar.aggregate(df)
    assert agg.iloc[0]["model"] == "default_model"
    assert agg.iloc[0]["num_players"] == 5

    # main full path
    eval_path = tmp_path / "run.eval"
    _make_eval(
        eval_path,
        [
            json.dumps(
                {
                    "scores": {
                        "red_vs_blue_scorer": {
                            "value": 1,
                            "metadata": {
                                "num_players": 5,
                                "avg_entropy_reduction": 0.1,
                                "avg_belief_alignment": 0.2,
                                "avg_brier": 0.3,
                                "apt_leader_deception": 0.4,
                                "rounds_played": 6,
                            },
                        }
                    }
                }
            )
        ],
    )
    ar.main(str(tmp_path))
    out = capsys.readouterr().out
    assert "Aggregated Results" in out
    assert (tmp_path / "aggregated" / "summary.csv").exists()


def test_convert_helpers_and_winner_logic():
    assert c2p._to_jsonable({"a": np.array([1, 2])}) == {"a": [1.0, 2.0]}
    assert c2p._to_jsonable((1, "x", None)) == [1.0, "x", None]

    assert c2p._winner_from_metadata({"blues_win": True, "reds_win": False}, None) == "blue"
    assert c2p._winner_from_metadata({"blues_win": False, "reds_win": True}, None) == "red"
    assert c2p._winner_from_metadata({}, 1) == "blue"
    assert c2p._winner_from_metadata({}, 0) == "red"
    assert c2p._winner_from_metadata({}, None) == "draw"


def test_convert_rollout_matrix_and_convert_flow(monkeypatch, tmp_path):
    sample = {
        "scores": {
            "red_vs_blue_scorer": {
                "value": 1,
                "metadata": {
                    "roles": {"p0": "blue", "p1": "red", "p2": "apt_leader"},
                    "patch_track": {"blue": 6, "red": 2},
                    "rounds_played": 7,
                    "avg_belief_alignment": 0.25,
                },
            }
        }
    }
    rollout = c2p._rollout_from_sample(sample, 3)
    assert rollout["winner"] == "blue"
    assert rollout["per_player_scores"]["p0"] == 1.0
    assert rollout["per_player_scores"]["p1"] == 0.0

    matrix, tasks = c2p._build_score_matrix([rollout], ["p0", "p1", "p2"])
    assert matrix.shape == (3, 1)
    assert tasks == ["inspect_rollout_3"]

    monkeypatch.setattr(c2p, "_collect_samples", lambda _path: [sample, sample])
    monkeypatch.setattr(
        c2p,
        "_solve_polarix",
        lambda agent_labels, task_labels, score_matrix, normalizer: {
            "solver": "ce_maxent",
            "normalizer": normalizer,
            "agent_ratings": {a: 1.0 for a in agent_labels},
            "agent_equilibrium_prob": {a: 0.5 for a in agent_labels},
            "summary": {"ok": True},
        },
    )

    out_path = tmp_path / "out" / "summary.json"
    written = c2p.convert(str(tmp_path), str(out_path), normalizer="winrate", model_name="demo")
    assert written.endswith("summary.json")
    payload = json.loads(out_path.read_text(encoding="utf-8"))
    assert payload["rollouts"] == 2
    assert payload["wins"]["blue"] == 2


def test_convert_main_and_solve_polarix_mock(monkeypatch):
    # test _solve_polarix with mocked polarix API
    monkeypatch.setattr(c2p.plx, "agent_vs_task_game", lambda **kwargs: {"game": kwargs})
    fake_result = SimpleNamespace(
        ratings=[None, [1.2, 0.8]],
        joint=np.array([[[0.25], [0.75]]]),
        summary={"eps": np.array([1, 2])},
    )
    monkeypatch.setattr(c2p.plx, "solve", lambda game, solver, disable_progress_bar=True: fake_result)
    solved = c2p._solve_polarix(["a", "b"], ["t"], np.array([[1.0], [0.0]]), "winrate")
    assert solved["agent_ratings"]["a"] == 1.2
    assert "eps" in solved["summary"]

    calls = {}

    def _fake_convert(results_dir, output_json, normalizer="winrate", policy="inspect-converted", model_name="inspect-derived"):
        calls["args"] = (results_dir, output_json, normalizer, policy, model_name)
        return output_json

    monkeypatch.setattr(c2p, "convert", _fake_convert)
    c2p.main(results_dir="R", output_json="O")
    assert calls["args"][0] == "R"


def test_cross_findings_helpers_and_reports(monkeypatch, tmp_path):
    eval_path = tmp_path / "cross.eval"
    _make_eval(eval_path, [json.dumps(_sample := {"scores": {"red_vs_blue_scorer": {"metadata": {"rounds_played": 3, "blues_win": True, "roles": {"p0": "blue"}, "patch_track": {"blue": 6, "red": 1}, "fired_players": [], "public_log": ["p0: hi"], "voting_history": []}}}}), "bad-json"])

    loaded = caf.load_eval_file(eval_path)
    assert len(loaded) == 1

    ctx = caf.extract_game_context(_sample)
    assert ctx["blues_win"] is True

    async def _good(*args, **kwargs):
        return {
            "overall_game_quality": "high",
            "cross_analysis_findings": ["f1"],
            "high_impact_players": ["p0"],
            "systemic_patterns": ["s1"],
            "recommended_interventions": [{"area": "rules", "recommendation": "r", "rationale": "why"}],
        }

    async def _none(*args, **kwargs):
        return None

    monkeypatch.setattr(caf, "generate_json_with_retries", _good)
    out = _run(caf.analyze_cross_findings(None, 1, ctx))
    assert out["overall_game_quality"] == "high"

    monkeypatch.setattr(caf, "generate_json_with_retries", _none)
    out2 = _run(caf.analyze_cross_findings(None, 1, ctx))
    assert out2["overall_game_quality"] == "unknown"

    report = caf.generate_cross_findings_markdown_report([
        {"game_num": 1, "game_context": ctx, "cross_findings": out}
    ])
    assert "Cross-Analysis Findings Report" in report
    assert "Recommended Interventions" in report


def test_cross_findings_main(monkeypatch, tmp_path):
    eval_path = tmp_path / "main.eval"
    _make_eval(eval_path, [json.dumps({"scores": {"red_vs_blue_scorer": {"metadata": {"rounds_played": 2, "blues_win": False, "roles": {"p0": "red"}, "patch_track": {"blue": 1, "red": 6}, "fired_players": ["p1"], "public_log": [], "voting_history": []}}}})])

    monkeypatch.setattr(caf, "get_model", lambda name: {"model": name})

    async def _fake_analyze(model, game_num, game_context):
        return {
            "overall_game_quality": "medium",
            "cross_analysis_findings": ["x"],
            "high_impact_players": ["p0"],
            "systemic_patterns": [],
            "recommended_interventions": [],
        }

    monkeypatch.setattr(caf, "analyze_cross_findings", _fake_analyze)

    _run(caf.main(str(eval_path), "demo/model"))
    assert (tmp_path / f"cross_analysis_findings_{eval_path.stem}.json").exists()
    assert (tmp_path / f"cross_analysis_findings_{eval_path.stem}.md").exists()
