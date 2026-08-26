# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Sign contract for the embodied credit-assignment chain.

The chunk-level recipes (XR-1 on RoboCasa, and every other embodied GRPO config
that sets ``reward_type: chunk_level``) route one sparse terminal success
through ``calculate_adv_and_returns`` and then through the clipped policy loss.
Nothing in that path is checked by an assertion at runtime, so a sign error
would show up only as a training run that quietly fails to learn -- the most
expensive possible way to find a bug.

These tests pin the sign end to end on the offline half of the loop:

* the episode that succeeded must come out with a positive advantage, and its
  group peers that failed with a negative one;
* gradient descent on the resulting loss must *raise* the log-probability of the
  successful episode and lower it for the failures;
* negating the reward (``env.train.reward_coef: -1.0``, which the RoboCasa
  wrapper applies as ``reward = reward_coef * terminations``) must flip both,
  which is what makes a negated-reward run a valid negative control for the
  whole pipeline.
"""

import pytest
import torch

from rlinf.algorithms.registry import calculate_adv_and_returns, policy_loss

CHUNK_STEPS = 6  # policy calls per episode
NUM_ACTION_CHUNKS = 10  # XR-1 predicts 10 actions per call
ACTION_DIM = 7  # XR-1 drives a 7D end-effector action
GROUP_SIZE = 4  # episodes scored against each other by GRPO
WINNER = 0  # the only episode in the group that succeeds


def _rollout(reward_coef: float):
    """One GRPO group where episode ``WINNER`` succeeds on its last chunk step."""
    rewards = torch.zeros(CHUNK_STEPS, GROUP_SIZE, NUM_ACTION_CHUNKS)
    rewards[-1, WINNER, -1] = reward_coef  # env: reward_coef * terminations

    dones = torch.zeros(
        CHUNK_STEPS + 1, GROUP_SIZE, NUM_ACTION_CHUNKS, dtype=torch.bool
    )
    dones[-1, :, -1] = True  # every episode ends at the horizon

    return rewards, dones


def _advantages(reward_coef: float) -> torch.Tensor:
    rewards, dones = _rollout(reward_coef)
    res = calculate_adv_and_returns(
        task_type="embodied",
        adv_type="grpo",
        rewards=rewards,
        dones=dones,
        values=None,
        num_action_chunks=NUM_ACTION_CHUNKS,
        group_size=GROUP_SIZE,
        reward_type="chunk_level",
        loss_mask=torch.ones(CHUNK_STEPS, GROUP_SIZE, 1, dtype=torch.bool),
        loss_mask_sum=None,
    )
    return res["advantages"]


def _logprob_grad(advantages: torch.Tensor) -> torch.Tensor:
    """Gradient of the clipped policy loss w.r.t. every per-dimension log-prob.

    Mirrors the actor: the micro-batch is flattened over (chunk step, episode),
    and ``logprob_type: chunk_level`` sums the 10 x 7 per-dimension log-probs of
    each policy call into one number before the ratio is formed.
    """
    flat_batch = CHUNK_STEPS * GROUP_SIZE
    logprobs = torch.zeros(
        flat_batch, NUM_ACTION_CHUNKS, ACTION_DIM, requires_grad=True
    )
    loss, metrics = policy_loss(
        loss_type="actor",
        task_type="embodied",
        logprob_type="chunk_level",
        reward_type="chunk_level",
        single_action_dim=ACTION_DIM,
        logprobs=logprobs,
        old_logprobs=torch.zeros_like(logprobs),  # start on policy: ratio == 1
        advantages=advantages,
        clip_ratio_low=0.2,
        clip_ratio_high=0.28,
        loss_mask=torch.ones(CHUNK_STEPS, GROUP_SIZE, 1, dtype=torch.bool),
    )
    assert loss.isfinite(), "the policy loss must be finite on an on-policy batch"
    assert abs(metrics["actor/ratio"] - 1.0) < 1e-6, (
        "an unchanged policy must give an importance ratio of exactly 1"
    )
    loss.backward()
    return logprobs.grad.reshape(CHUNK_STEPS, GROUP_SIZE, NUM_ACTION_CHUNKS, ACTION_DIM)


def test_success_outscores_its_group():
    advantages = _advantages(reward_coef=1.0)

    assert advantages.shape == (CHUNK_STEPS, GROUP_SIZE, 1)
    assert (advantages[:, WINNER] > 0).all(), (
        "the episode that reached the goal must get a positive advantage"
    )
    peers = [i for i in range(GROUP_SIZE) if i != WINNER]
    assert (advantages[:, peers] < 0).all(), (
        "episodes that failed must get a negative advantage"
    )
    # GRPO normalizes within the group, so the group has no net bias.
    assert advantages.sum(dim=1).abs().max() < 1e-5


def test_descent_raises_the_logprob_of_the_successful_episode():
    grad = _logprob_grad(_advantages(reward_coef=1.0))

    # A descent step moves along -grad, so a negative gradient raises the logprob.
    assert (grad[:, WINNER] < 0).all(), (
        "gradient descent must make the successful episode more likely"
    )
    peers = [i for i in range(GROUP_SIZE) if i != WINNER]
    assert (grad[:, peers] > 0).all(), (
        "gradient descent must make the failed episodes less likely"
    )


@pytest.mark.parametrize("reward_coef", [-1.0, -2.5])
def test_negated_reward_flips_the_whole_chain(reward_coef):
    """A negated reward is a valid negative control for the full pipeline."""
    positive = _advantages(reward_coef=1.0)
    negated = _advantages(reward_coef=reward_coef)

    # GRPO divides by the group std, so the magnitude is scale-invariant and
    # only the sign flips -- a negative-reward run probes the same gradient
    # path at the same effective step size.
    torch.testing.assert_close(negated, -positive)
    torch.testing.assert_close(_logprob_grad(negated), -_logprob_grad(positive))
