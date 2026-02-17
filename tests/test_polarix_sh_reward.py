from types import SimpleNamespace

from polarix_sh.reward import compute_sh_reward


class FakeEnv:
    def __init__(self, winner, blue_policies, red_policies, entropy_reduction,
                 correct_accusations, incorrect_accusations):
        self.winner = winner
        self._blue = blue_policies
        self._red = red_policies
        self._entropy = entropy_reduction
        self._correct = correct_accusations
        self._incorrect = incorrect_accusations

    def num_blue_policies_enacted_this_step(self):
        return self._blue

    def num_red_policies_enacted_this_step(self):
        return self._red

    def entropy_reduction(self):
        return self._entropy

    def correct_accusations(self):
        return self._correct

    def incorrect_accusations(self):
        return self._incorrect


def test_compute_sh_reward_terminal_and_shaping():
    env = FakeEnv(
        winner="blue",
        blue_policies=1,
        red_policies=0,
        entropy_reduction=0.5,
        correct_accusations=2,
        incorrect_accusations=1,
    )

    reward, role_rewards = compute_sh_reward(env, done=True)

    # manual expected calculation
    expected = 0.0
    expected += 1.0  # blue terminal
    expected += 0.10 * 1  # blue policy
    expected -= 0.10 * 0  # red policy
    expected += 0.2 * 0.5
    expected += 0.1 * 2
    expected -= 0.05 * 1

    assert abs(reward - expected) < 1e-8
    assert abs(role_rewards["blue_role_reward"] - 0.1) < 1e-8
    assert abs(role_rewards["red_role_reward"] + 0.1) < 1e-8


def test_compute_sh_reward_nonterminal():
    env = FakeEnv(
        winner=None,
        blue_policies=0,
        red_policies=2,
        entropy_reduction=0.0,
        correct_accusations=0,
        incorrect_accusations=3,
    )

    reward, role_rewards = compute_sh_reward(env, done=False)

    expected = 0.0
    expected += 0.10 * 0
    expected -= 0.10 * 2
    expected += 0.2 * 0.0
    expected += 0.1 * 0
    expected -= 0.05 * 3

    assert abs(reward - expected) < 1e-8
    assert abs(role_rewards["blue_role_reward"] - (-0.2)) < 1e-8
    assert abs(role_rewards["red_role_reward"] - 0.2) < 1e-8
