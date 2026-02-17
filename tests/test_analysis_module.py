import json
from types import SimpleNamespace

import pandas as pd

from analysis.aggregate_results import aggregate_role_policy_rewards
from analysis.merge_logs import load_json_logs
from analysis.plots import plot_role_policy_rewards
from analysis.model_profiles import plot_model_profile


def test_aggregate_role_policy_rewards():
    df = pd.DataFrame(
        [
            {"model": "A", "value": 1.0, "blue_role_reward": 0.1, "red_role_reward": -0.1},
            {"model": "A", "value": 2.0, "blue_role_reward": 0.2, "red_role_reward": -0.2},
            {"model": "B", "value": 3.0, "blue_role_reward": 0.0, "red_role_reward": 0.0},
        ]
    )

    out = aggregate_role_policy_rewards(df)

    a_row = out[out["model"] == "A"].iloc[0]
    assert abs(a_row.mean_reward - 1.5) < 1e-8
    assert abs(a_row.mean_blue_role_reward - 0.15) < 1e-8


def test_load_json_logs_and_plots(tmp_path):
    d = tmp_path / "logs"
    d.mkdir()

    data1 = {"model": "m1", "reward": 1.0, "tokens": 10, "time": 0.5, "blue_role_reward": 0.1, "red_role_reward": -0.1}
    data2 = {"model": "m2", "score": 2.0, "tokens": 20, "time": 1.0, "blue_role_reward": 0.2, "red_role_reward": -0.2}

    p1 = d / "a.json"
    p2 = d / "b.json"

    p1.write_text(json.dumps(data1))
    p2.write_text(json.dumps(data2))

    df = load_json_logs(str(d), "bench")

    assert set(df["model"]) == {"m1", "m2"}
    assert "value" in df.columns

    # test plotting utilities create files
    out_plot = tmp_path / "role_plot.png"
    plot_role_policy_rewards(pd.DataFrame({
        "model": ["m1", "m2"],
        "mean_blue_role_reward": [0.1, 0.2],
        "mean_red_role_reward": [-0.1, -0.2],
    }), out_path=str(out_plot))

    assert out_plot.exists()

    # test model profile plotting
    out_profile = tmp_path / "profile.png"
    row = SimpleNamespace(mean_reward=1.0, mean_blue_role_reward=0.1, mean_red_role_reward=-0.1)
    plot_model_profile(row, str(out_profile))
    assert out_profile.exists()
