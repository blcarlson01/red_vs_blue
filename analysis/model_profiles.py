import matplotlib.pyplot as plt


def plot_model_profile(row, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].bar(["Overall Reward"], [row.mean_reward])
    axes[0].set_title("Overall Performance")

    axes[1].bar(
        ["Blue Role", "Red Role"],
        [row.mean_blue_role_reward, row.mean_red_role_reward]
    )
    axes[1].set_title("Per-Role Policy Reward")

    plt.tight_layout()
    plt.savefig(out_path)