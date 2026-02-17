import json
import pandas as pd
from pathlib import Path


def load_json_logs(dir_path, benchmark_name):
    rows = []
    for p in Path(dir_path).glob("*.json"):
        with p.open("r", encoding="utf-8") as fh:
            data = json.load(fh)

        rows.append({
            "model": data.get("model"),
            "benchmark": benchmark_name,
            "value": data.get("reward") or data.get("score"),
            "tokens": data.get("tokens"),
            "time": data.get("time"),
            "blue_role_reward": data.get("blue_role_reward"),
            "red_role_reward": data.get("red_role_reward"),
        })
    return pd.DataFrame(rows)


def main(inspect_dir, polarix_dir, out_csv):
    df_i = load_json_logs(inspect_dir, "red_vs_blue_inspect")
    df_p = load_json_logs(polarix_dir, "red_vs_blue_polarix")

    df = pd.concat([df_i, df_p], ignore_index=True)
    df.to_csv(out_csv, index=False)


if __name__ == "__main__":
    import sys
    main(sys.argv[1], sys.argv[2], sys.argv[3])