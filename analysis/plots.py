import matplotlib.pyplot as plt

def plot_role_policy_rewards(df, out_path="role_policy_rewards.png"):
    fig, ax = plt.subplots()

    for model in df.model.unique():
        sub = df[df.model == model]
        ax.scatter(
            sub.mean_blue_role_reward,
            sub.mean_red_role_reward,
            label=model,
        )

    ax.set_xlabel("Blue Role Policy Reward")
    ax.set_ylabel("Red Role Policy Reward")
    ax.set_title("Per-Role Policy Competence")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.savefig(out_path)