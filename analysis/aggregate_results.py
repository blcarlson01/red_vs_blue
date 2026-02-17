import pandas as pd

def aggregate_role_policy_rewards(df):
    return df.groupby("model").agg(
        mean_reward=("value", "mean"),
        mean_blue_role_reward=("blue_role_reward", "mean"),
        mean_red_role_reward=("red_role_reward", "mean"),
    ).reset_index()