from __future__ import annotations

import json
import os
import types
import zipfile
from pathlib import Path

import pandas as pd
import pytest

import red_vs_blue.analysis.confusion_analysis as confusion
import red_vs_blue.analysis.llm_client as llm_client
import red_vs_blue.analysis.statistics as stats
import red_vs_blue.analysis.strategy_analysis as strategy


class _AsyncResponseModel:
    def __init__(self, responses):
        self._responses = list(responses)

    async def generate(self, input, config):
        if not self._responses:
            raise RuntimeError("no response")
        item = self._responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def _sample_data():
    return {
        "scores": {
            "red_vs_blue_scorer": {
                "metadata": {
                    "num_players": 3,
                    "rounds_played": 4,
                    "blues_win": True,
                    "patch_track": {"blue": 6, "red": 2},
                    "roles": {"p0": "blue", "p1": "red", "p2": "apt_leader"},
                    "true_apt_leader": "p2",
                    "fired_players": ["p1"],
                    "public_log": ["p0: hi", "p1: no", "p2: maybe"],
                    "voting_history": [{"round": 1, "votes": {"p0": "yes", "p1": "no"}}],
                }
            }
        }
    }


def test_confusion_and_strategy_load_eval_file(tmp_path):
    eval_path = tmp_path / "game.eval"
    with zipfile.ZipFile(eval_path, "w") as zf:
        zf.writestr("samples/001.json", json.dumps({"id": 1}))
        zf.writestr("samples/002.json", "not-json")
        zf.writestr("other/ignored.json", json.dumps({"id": 2}))

    c_results = confusion.load_eval_file(eval_path)
    s_results = strategy.load_eval_file(eval_path)

    assert len(c_results) == 1
    assert c_results[0]["id"] == 1
    assert len(s_results) == 1


def test_extract_helpers_defaults_and_reasoning():
    sample = _sample_data()

    c_reasoning = confusion.extract_player_reasoning(sample)
    assert c_reasoning["p0"] == ["hi"]
    assert c_reasoning["p1"] == ["no"]

    s_reasoning = strategy.extract_player_reasoning(sample)
    assert s_reasoning["p2"] == ["maybe"]

    c_ctx = confusion.extract_game_context(sample)
    assert c_ctx["reds_win"] is False

    s_ctx = strategy.extract_game_context(sample)
    assert isinstance(s_ctx["voting_history"], list)

    defaults = confusion.extract_game_context({})
    assert defaults["num_players"] == 5
    assert defaults["patch_track"] == {"blue": 0, "red": 0}


def test_confusion_analysis_functions_with_mocked_llm(monkeypatch):
    async def _good(*args, **kwargs):
        return {"confused": True, "confusion_types": ["state"], "explanation": "x", "evidence": ["e"], "improvement_suggestions": ["i"]}

    async def _none(*args, **kwargs):
        return None

    sample = _sample_data()
    ctx = confusion.extract_game_context(sample)

    monkeypatch.setattr(confusion, "generate_json_with_retries", _good)
    out = pytest.run(async_fn=confusion.analyze_player_confusion, model=None, player_id="p0", role="blue", reasoning=["a"], game_context=ctx)
    assert out["confused"] is True

    monkeypatch.setattr(confusion, "generate_json_with_retries", _none)
    out2 = pytest.run(async_fn=confusion.analyze_player_confusion, model=None, player_id="p0", role="blue", reasoning=["a"], game_context=ctx)
    assert out2["explanation"] == "Analysis failed"


def test_confusion_improvements_and_analyze_game(monkeypatch):
    async def _improvements(*args, **kwargs):
        return {"overall_confusion_level": "low", "key_insights": ["k"], "improvement_suggestions": [{"category": "Rules", "suggestion": "s", "rationale": "r", "implementation": "i"}]}

    async def _player_analysis(model, player_id, role, reasoning, game_context):
        return {"confused": player_id == "p1", "confusion_types": ["logic"] if player_id == "p1" else [], "explanation": "ok", "evidence": ["q"] if player_id == "p1" else [], "improvement_suggestions": []}

    sample = _sample_data()
    ctx = confusion.extract_game_context(sample)

    monkeypatch.setattr(confusion, "generate_json_with_retries", _improvements)
    improved = pytest.run(async_fn=confusion.analyze_game_improvements, model=None, game_context=ctx, all_player_analysis={"p0": {"confused": False}})
    assert improved["overall_confusion_level"] == "low"

    monkeypatch.setattr(confusion, "analyze_player_confusion", _player_analysis)
    monkeypatch.setattr(confusion, "analyze_game_improvements", _improvements)
    result = pytest.run(async_fn=confusion.analyze_game, game_num=1, sample_data=sample, model=None)
    assert result["confused_count"] == 1


def test_confusion_markdown_report_branches():
    all_results = [
        {
            "game_num": 1,
            "game_context": {"roles": {"p0": "blue"}, "blues_win": True, "rounds_played": 3, "patch_track": {"blue": 6, "red": 1}},
            "player_analysis": {"p0": {"confused": False, "confusion_types": [], "explanation": "ok", "evidence": []}},
            "confused_count": 0,
            "improvements": {"overall_confusion_level": "low", "improvement_suggestions": []},
        },
        {
            "game_num": 2,
            "game_context": {"roles": {"p0": "blue", "p1": "red"}, "blues_win": False, "rounds_played": 5, "patch_track": {"blue": 2, "red": 6}},
            "player_analysis": {"p0": {"confused": True, "confusion_types": ["state"], "explanation": "x", "evidence": ["quote"]}, "p1": {"confused": True, "confusion_types": ["state"], "explanation": "y", "evidence": []}},
            "confused_count": 2,
            "improvements": {"overall_confusion_level": "high", "improvement_suggestions": [{"category": "Rules", "suggestion": "clarify", "rationale": "why"}]},
        },
    ]

    report = confusion.generate_confusion_markdown_report(all_results)
    assert "Confusion Analysis Report" in report
    assert "Confusion Types Summary" in report
    assert "state" in report


def test_strategy_analysis_functions_with_mocked_llm(monkeypatch):
    async def _good(*args, **kwargs):
        return {"has_strategy": True, "strategy_name": "Test", "strategy_description": "desc", "consistency_score": 0.8, "strategy_effectiveness": "high", "key_behaviors": ["k"], "contradictions": [], "role_alignment": "well-aligned"}

    async def _none(*args, **kwargs):
        return None

    sample = _sample_data()
    ctx = strategy.extract_game_context(sample)

    monkeypatch.setattr(strategy, "generate_json_with_retries", _good)
    out = pytest.run(async_fn=strategy.analyze_player_strategy, model=None, player_id="p0", role="blue", reasoning=["a"], game_context=ctx)
    assert out["has_strategy"] is True

    ctx_bad = dict(ctx)
    ctx_bad["voting_history"] = "bad"
    monkeypatch.setattr(strategy, "generate_json_with_retries", _none)
    out2 = pytest.run(async_fn=strategy.analyze_player_strategy, model=None, player_id="p0", role="blue", reasoning=["a"], game_context=ctx_bad)
    assert out2["strategy_description"] == "Analysis failed"


def test_strategy_analyze_game_and_report(monkeypatch):
    async def _analysis(model, player_id, role, reasoning, game_context):
        return {
            "has_strategy": player_id == "p0",
            "strategy_name": "Coordination" if player_id == "p0" else None,
            "strategy_description": "desc",
            "consistency_score": 0.9 if player_id == "p0" else 0.1,
            "strategy_effectiveness": "high" if player_id == "p0" else "low",
            "key_behaviors": ["vote"],
            "contradictions": [] if player_id == "p0" else ["flip"],
            "role_alignment": "well-aligned",
        }

    sample = _sample_data()
    monkeypatch.setattr(strategy, "analyze_player_strategy", _analysis)
    game = pytest.run(async_fn=strategy.analyze_game, game_num=1, sample_data=sample, model=None)
    assert game["strategic_players"] == 1

    report = strategy.generate_strategy_markdown_report([game])
    assert "Strategy Analysis Report" in report
    assert "Most Common Strategies" in report


def test_llm_client_json_extract_and_retry_paths(monkeypatch):
    assert llm_client._resolve_model_name("ollama/gpt-oss") == "gpt-oss"
    assert llm_client._resolve_model_name("gpt-oss") == "gpt-oss"

    monkeypatch.setenv("MODEL_BASE_URL", "http://base")
    assert llm_client._resolve_base_url() == "http://base"

    assert llm_client.extract_json_object('{"a":1}') == {"a": 1}
    assert llm_client.extract_json_object("prefix {\"a\": 2} suffix") == {"a": 2}
    assert llm_client.extract_json_object("no json") is None

    model = _AsyncResponseModel([
        types.SimpleNamespace(completion="not-json"),
        types.SimpleNamespace(completion='{"ok": true}'),
    ])
    parsed = pytest.run(async_fn=llm_client.generate_json_with_retries, model=model, prompt="p", max_tokens=50, retries=2)
    assert parsed == {"ok": True}

    choice_resp = types.SimpleNamespace(choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"x":1}'))])
    parsed2 = pytest.run(async_fn=llm_client.generate_json_with_retries, model=_AsyncResponseModel([choice_resp]), prompt="p", max_tokens=50, retries=1)
    assert parsed2 == {"x": 1}


def test_llm_client_get_model_branches(monkeypatch):
    fake_inspect = types.ModuleType("inspect_ai.model")
    fake_inspect.get_model = lambda name: {"name": name}

    import sys
    monkeypatch.setitem(sys.modules, "inspect_ai.model", fake_inspect)
    primary = llm_client.get_model("x")
    assert primary == {"name": "x"}

    def _raising_import(name, *args, **kwargs):
        if name == "inspect_ai.model":
            raise ImportError("boom")
        return __import__(name, *args, **kwargs)

    monkeypatch.setattr(llm_client, "_FallbackModel", lambda model_name: {"fallback": model_name})
    monkeypatch.setattr("builtins.__import__", _raising_import)
    fallback = llm_client.get_model("y")
    assert fallback == {"fallback": "y"}


def test_statistics_helpers_and_compare(monkeypatch, tmp_path, capsys):
    with pytest.raises(FileNotFoundError):
        stats.load_results(tmp_path)

    agg = tmp_path / "aggregated"
    agg.mkdir()
    df = pd.DataFrame(
        [
            {"model": "A", "value": 1, "avg_belief_alignment": 0.1, "avg_entropy_reduction": 0.2, "avg_brier": 0.3, "apt_leader_deception": 0.4, "avg_rounds": 5},
            {"model": "A", "value": 0, "avg_belief_alignment": 0.2, "avg_entropy_reduction": 0.3, "avg_brier": 0.4, "apt_leader_deception": 0.5, "avg_rounds": 6},
            {"model": "B", "value": 1, "avg_belief_alignment": 0.1, "avg_entropy_reduction": 0.2, "avg_brier": 0.3, "apt_leader_deception": 0.4, "avg_rounds": 5},
            {"model": "B", "value": 0, "avg_belief_alignment": 0.2, "avg_entropy_reduction": 0.3, "avg_brier": 0.4, "apt_leader_deception": 0.5, "avg_rounds": 6},
            {"model": "A", "value": 1, "avg_belief_alignment": 0.1, "avg_entropy_reduction": 0.2, "avg_brier": 0.3, "apt_leader_deception": 0.4, "avg_rounds": 5},
            {"model": "A", "value": 0, "avg_belief_alignment": 0.2, "avg_entropy_reduction": 0.3, "avg_brier": 0.4, "apt_leader_deception": 0.5, "avg_rounds": 6},
            {"model": "A", "value": 1, "avg_belief_alignment": 0.1, "avg_entropy_reduction": 0.2, "avg_brier": 0.3, "apt_leader_deception": 0.4, "avg_rounds": 5},
            {"model": "B", "value": 1, "avg_belief_alignment": 0.1, "avg_entropy_reduction": 0.2, "avg_brier": 0.3, "apt_leader_deception": 0.4, "avg_rounds": 5},
            {"model": "B", "value": 0, "avg_belief_alignment": 0.2, "avg_entropy_reduction": 0.3, "avg_brier": 0.4, "apt_leader_deception": 0.5, "avg_rounds": 6},
            {"model": "B", "value": 1, "avg_belief_alignment": 0.1, "avg_entropy_reduction": 0.2, "avg_brier": 0.3, "apt_leader_deception": 0.4, "avg_rounds": 5},
        ]
    )
    df.to_json(agg / "all_results.jsonl", orient="records", lines=True)

    loaded = stats.load_results(tmp_path)
    assert not loaded.empty

    monkeypatch.setattr(stats, "permutation_test", lambda x, y: 0.01)
    monkeypatch.setattr(stats, "cohens_d", lambda x, y: 0.5)
    compared = stats.compare_models(loaded, "A", "B")
    assert "win_rate" in compared

    monkeypatch.setattr(stats, "holm_bonferroni", lambda pvals, alpha: {k: {"reject_null": True} for k in pvals})
    stats.print_results("A", "B", {"win_rate": {"mean_a": 0.6, "mean_b": 0.4, "p_value": 0.01, "effect_size_d": 0.5}})
    captured = capsys.readouterr()
    assert "A vs B" in captured.out


# Small helper to run async functions without plugin dependency
class _PytestAsyncRunner:
    @staticmethod
    def run(async_fn, *args, **kwargs):
        import asyncio

        return asyncio.run(async_fn(*args, **kwargs))


pytest.run = _PytestAsyncRunner.run
