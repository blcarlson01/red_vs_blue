# polarix_sh/reward.py

def compute_sh_reward(env, done: bool):
    """
    Reward structure:
    - Terminal reward dominates
    - Policy rewards are shaping
    - Epistemic shaping included
    - Per-role policy rewards logged separately
    """

    reward = 0.0

    # ---- Terminal reward ----
    if done:
        if env.winner == "blue":
            reward += 1.0
        elif env.winner == "red":
            reward -= 1.0

    # ---- Policy shaping ----
    blue_policies = env.num_blue_policies_enacted_this_step()
    red_policies = env.num_red_policies_enacted_this_step()

    reward += 0.10 * blue_policies
    reward -= 0.10 * red_policies

    # ---- Epistemic shaping ----
    reward += 0.2 * env.entropy_reduction()
    reward += 0.1 * env.correct_accusations()
    reward -= 0.05 * env.incorrect_accusations()

    # ---- Per-role tracking (not added to main reward) ----
    role_rewards = {
        "blue_role_reward": 0.10 * blue_policies - 0.10 * red_policies,
        "red_role_reward": 0.10 * red_policies - 0.10 * blue_policies,
    }

    return reward, role_rewards