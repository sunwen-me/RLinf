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

"""Degenerate-batch contract for ``filter_rewards``.

``algorithm.filter_rewards`` drops a GRPO group whose mean episodic return
falls outside ``[rewards_lower_bound, rewards_upper_bound]`` by zeroing its
``loss_mask``.  The bounds are expressed in *return* units, so a recipe whose
successful episode already returns ~1.0 (every RoboCasa config: ``reward_coef:
1.0`` differenced by ``use_rel_reward``) drops any group that is doing well --
and once every group in a batch is out of band, the mask is entirely False.

That case used to sail through: the masked means downstream divide by zero, the
step logs ``advantages_mean=nan`` with every ``actor/*`` metric at exactly
0.0, and no gradient is applied at all.  It cost a 100-step run its final
checkpoint when an external watchdog read the nan as divergence.  The filter now
falls back to the unfiltered batch, which is the only thing it can usefully say
about a batch it rejects wholesale.

The same guard is duplicated in ``EmbodiedFSDPActorWorker._preprocess_batch``,
which carries its own inline copy of this filter; these tests cover the shared
helper in ``rlinf.utils.utils``.
"""

import torch

from rlinf.utils.utils import preprocess_embodied_batch

N_CHUNK_STEP = 2
GROUP_SIZE = 2
LOWER, UPPER = 0.1, 0.9


def _batch(episode_returns: list[float]) -> dict[str, torch.Tensor]:
    """One batch whose per-episode return is exactly ``episode_returns[b]``."""
    rewards = torch.zeros(N_CHUNK_STEP, len(episode_returns), 1)
    rewards[0, :, 0] = torch.tensor(episode_returns)
    return {"rewards": rewards}


def _filter(episode_returns: list[float]) -> torch.Tensor:
    out = preprocess_embodied_batch(
        _batch(episode_returns),
        rollout_epoch=1,
        auto_reset=True,
        ignore_terminations=False,
        reward_type="chunk_level",
        filter_rewards=True,
        group_size=GROUP_SIZE,
        rewards_lower_bound=LOWER,
        rewards_upper_bound=UPPER,
    )
    return out["loss_mask"]


def test_only_groups_inside_the_band_survive():
    # Group means 0.0 (below), 0.5 (inside), 2.0 (above).
    mask = _filter([0.0, 0.0, 0.4, 0.6, 2.0, 2.0])
    assert mask.shape == (N_CHUNK_STEP, 6, 1)
    kept = mask[0, :, 0].tolist()
    assert kept == [False, False, True, True, False, False]


def test_a_wholly_out_of_band_batch_falls_back_to_unfiltered():
    # Every group mean is 1.2, i.e. above ``UPPER`` -- the regression case.
    mask = _filter([1.2] * 6)
    assert mask.all(), "a fully rejected batch must train unfiltered, not on nothing"


def test_the_fallback_keeps_masked_means_finite():
    mask = _filter([1.2] * 6)
    values = torch.arange(mask.numel(), dtype=torch.float32).reshape(mask.shape)
    masked_mean = (values * mask).sum() / mask.sum()
    assert torch.isfinite(masked_mean), "this is the nan that killed the run"


def test_filtering_off_leaves_the_batch_alone():
    out = preprocess_embodied_batch(
        _batch([1.2] * 6),
        rollout_epoch=1,
        auto_reset=True,
        ignore_terminations=False,
        reward_type="chunk_level",
        filter_rewards=False,
        group_size=GROUP_SIZE,
        rewards_lower_bound=LOWER,
        rewards_upper_bound=UPPER,
    )
    assert "loss_mask" not in out
