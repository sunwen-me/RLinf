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

"""Contract tests for how ``logprob_type`` amplifies rollout/training mismatch.

On the synchronous embodied path the actor takes ``old_logprobs`` straight from
the rollout worker (``embodied_fsdp_actor_worker.py``: ``old_logprobs =
micro_batch["prev_logprobs"]``), so the numerator and denominator of the
importance ratio come from two different forward passes and never agree
exactly. ``preprocess_loss_inputs`` decides how much that disagreement matters:
``chunk_level`` sums ``num_action_chunks * action_dim`` per-dimension logprobs
into a single number, so a per-dimension mismatch of ``sigma`` reaches the ratio
as ``sqrt(70) * sigma`` -- against a ``clip_ratio`` of 0.2.

This is not hypothetical. The ``critic_warmup_steps`` phase of an embodied PPO
run holds the actor frozen (``actor/lr=0``), so the true ratio is exactly 1 and
any reported deviation is pure mismatch. An XR-1 RoboCasa run measured
``actor/ratio_abs`` at 0.101-0.115 and ``actor/clip_fraction`` at 7.1-8.3% over
those frozen steps: the PPO clip was firing on numerical noise alone. These
tests pin the amplification, so that the cost of each granularity is explicit
rather than something to rediscover from a flat training curve.
"""

import math

import pytest
import torch
from rlinf.algorithms.utils import preprocess_loss_inputs

NUM_ACTION_CHUNKS = 10
ACTION_DIM = 7
BATCH = 4096
CLIP_RATIO = 0.2

# Per-dimension mismatch that reproduces the measured chunk-level floor:
# E|exp(D) - 1| ~= sqrt(2/pi) * sqrt(70) * SIGMA_PER_DIM = 0.108.
SIGMA_PER_DIM = 0.0162

DIMS_SUMMED = {
    "token_level": 1,
    "action_level": ACTION_DIM,
    "chunk_level": NUM_ACTION_CHUNKS * ACTION_DIM,
}


def _ratio_error(logprob_type: str, sigma: float, seed: int = 0) -> torch.Tensor:
    """Return ``|ratio - 1|`` when the two forwards differ by ``sigma`` per dim.

    The policy is unchanged -- only the numerical disagreement between the
    rollout and training forwards is present -- so the true ratio is exactly 1
    and everything returned here is error.
    """
    gen = torch.Generator().manual_seed(seed)
    old_logprobs = torch.zeros(BATCH, NUM_ACTION_CHUNKS, ACTION_DIM)
    noise = torch.randn(old_logprobs.shape, generator=gen) * sigma
    res = preprocess_loss_inputs(
        logprobs=old_logprobs + noise,
        old_logprobs=old_logprobs,
        advantages=torch.zeros(BATCH),
        logprob_type=logprob_type,
        reward_type="chunk_level",
        single_action_dim=ACTION_DIM,
    )
    ratio = (res["logprobs"] - res["old_logprobs"]).exp()
    return (ratio - 1.0).abs()


@pytest.mark.parametrize("logprob_type", list(DIMS_SUMMED))
def test_mismatch_grows_as_sqrt_of_the_dimensions_summed(logprob_type):
    measured = _ratio_error(logprob_type, SIGMA_PER_DIM).mean().item()
    expected = (
        math.sqrt(2.0 / math.pi)
        * math.sqrt(DIMS_SUMMED[logprob_type])
        * SIGMA_PER_DIM
    )
    assert measured == pytest.approx(expected, rel=0.05)


def test_chunk_level_puts_the_noise_floor_near_half_the_clip_range():
    chunk = _ratio_error("chunk_level", SIGMA_PER_DIM).mean().item()
    token = _ratio_error("token_level", SIGMA_PER_DIM).mean().item()
    # Brackets the 0.101-0.115 measured with a frozen actor.
    assert 0.09 < chunk < 0.13
    assert chunk > 0.4 * CLIP_RATIO
    # The very same per-dimension noise is negligible one level down.
    assert token < 0.1 * CLIP_RATIO


def test_chunk_level_mismatch_alone_clips_a_measurable_fraction():
    chunk = _ratio_error("chunk_level", SIGMA_PER_DIM)
    token = _ratio_error("token_level", SIGMA_PER_DIM)
    # PPO's min() only bites on the side the advantage points to, so roughly
    # half of the |ratio - 1| > clip_ratio mass reaches actor/clip_fraction.
    # The frozen-actor run measured 7.1-8.3%.
    one_sided = (chunk > CLIP_RATIO).float().mean().item() / 2.0
    assert 0.04 < one_sided < 0.12
    assert (token > CLIP_RATIO).float().mean().item() == 0.0
