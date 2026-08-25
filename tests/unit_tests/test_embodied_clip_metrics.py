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

"""Contract tests for ``actor/clip_fraction`` and ``actor/approx_kl`` normalization.

Under ``logprob_type: token_level`` the importance ratio keeps its trailing
action dimension -- ``ratio`` is ``[bsz, num_chunks, action_dim]`` -- while
``loss_mask`` stays ``[bsz, num_chunks, 1]``. In ``compute_ppo_actor_loss`` both
of these metrics divide a numerator that broadcasts over ``action_dim`` by
``loss_mask_count``, which counts only the unexpanded mask::

    loss_mask_count = loss_mask.count_nonzero()  # 1x
    clip_fraction = (clip_mask * loss_mask).sum() / loss_mask_count  # 7x / 1x
    approx_kl = -torch.sum(approx_kl) / loss_mask_count  # 7x / 1x

so both are reported ``action_dim`` times too large. The neighbouring metrics
are already normalized correctly against ``loss_mask.expand_as(ratio)``, which
is built a few lines further down -- ``ratio_abs`` is trustworthy, these two are
not.

The symptom is unmissable once you look: an XR-1 RoboCasa GRPO run at
``lr: 5.0e-5`` logged ``actor/clip_fraction=6.492``. A fraction cannot exceed 1.
The same run's steady state reads 0.638, i.e. a true 9.1%, and its ``lr: 5.0e-6``
control reads 0.015-0.038, i.e. a true 0.2-0.5%. Both were misread as
percentages an order of magnitude larger before these tests existed.

Every ``token_level`` recipe in the repo is affected, not just XR-1:
``libero_spatial_grpo_evo1.yaml`` (14x7 dims) and the lingbotvla configs
(50 chunks) log the same two metrics through the same path.
"""

import pytest
import torch

from rlinf.algorithms.losses import (
    compute_decoupled_ppo_actor_loss,
    compute_ppo_actor_loss,
)

NUM_CHUNKS = 10
ACTION_DIM = 7
BATCH = 64


def _actor_metrics(
    log_ratio_per_dim: float,
    advantage: float,
    action_dim: int = ACTION_DIM,
) -> dict:
    """Run the actor loss on token_level-shaped inputs and return its metrics.

    ``log_ratio_per_dim`` is applied to every dimension, so the true per-dim
    quantities are known exactly: the ratio is ``exp(log_ratio_per_dim)`` and the
    mean log-ratio is ``log_ratio_per_dim`` itself.
    """
    old_logprobs = torch.zeros(BATCH, NUM_CHUNKS, action_dim)
    logprobs = old_logprobs + log_ratio_per_dim
    _, metrics = compute_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        advantages=torch.full((BATCH, NUM_CHUNKS, 1), advantage),
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        loss_mask=torch.ones(BATCH, NUM_CHUNKS, 1).bool(),
    )
    return metrics


@pytest.mark.parametrize("advantage", [1.0, -1.0])
def test_clip_fraction_is_a_fraction(advantage: float) -> None:
    """``clip_fraction`` must lie in [0, 1] whatever the inputs.

    This needs no reference value: the metric is defined as a fraction of the
    batch, so a reported 6.492 is self-evidently wrong. A log-ratio of 1.0 puts
    every dimension far outside the 0.2 clip range, so the true answer is either
    exactly 0.0 (clipping does not bind) or exactly 1.0 (it binds everywhere).
    """
    metrics = _actor_metrics(log_ratio_per_dim=1.0, advantage=advantage)
    clip_fraction = metrics["actor/clip_fraction"].item()
    assert 0.0 <= clip_fraction <= 1.0, (
        f"clip_fraction={clip_fraction:.3f} is not a fraction; it is inflated by "
        f"action_dim={ACTION_DIM} because the numerator broadcasts over the "
        f"trailing dim while loss_mask_count does not"
    )


def test_clip_fraction_is_one_when_every_dim_clips() -> None:
    """With every dimension outside the clip range, the fraction is exactly 1."""
    metrics = _actor_metrics(log_ratio_per_dim=1.0, advantage=1.0)
    assert metrics["actor/clip_fraction"].item() == pytest.approx(1.0)


def test_approx_kl_is_the_per_dim_mean_log_ratio() -> None:
    """``approx_kl`` must not scale with ``action_dim``.

    Every dimension carries the same log-ratio, so the mean is that value. The
    sign follows the implementation (``-sum(logprobs - old_logprobs)``).
    """
    log_ratio = 0.01
    metrics = _actor_metrics(log_ratio_per_dim=log_ratio, advantage=1.0)
    assert metrics["actor/approx_kl"].item() == pytest.approx(-log_ratio, rel=1e-5)


@pytest.mark.parametrize("action_dim", [1, 7])
def test_metrics_do_not_depend_on_action_dim(action_dim: int) -> None:
    """The same per-dim mismatch must report the same metrics at any width.

    ``action_dim=1`` is the degenerate case where the buggy and correct
    normalizations coincide, so it pins the value the wider case must match.
    """
    metrics = _actor_metrics(
        log_ratio_per_dim=0.01, advantage=1.0, action_dim=action_dim
    )
    assert metrics["actor/approx_kl"].item() == pytest.approx(-0.01, rel=1e-5)
    assert 0.0 <= metrics["actor/clip_fraction"].item() <= 1.0


def _decoupled_metrics(
    log_ratio_per_dim: float,
    action_dim: int = ACTION_DIM,
    behave_weight_threshold: float | None = None,
) -> dict:
    """Same probe against the ``decoupled_actor_critic`` / ``opd`` loss path.

    ``maniskill_async_ppo_openpi.yaml`` ships ``loss_type:
    decoupled_actor_critic`` together with ``logprob_type: token_level``, so this
    path takes the same broadcast shapes and had the same inflated denominators.
    """
    old_logprobs = torch.zeros(BATCH, NUM_CHUNKS, action_dim)
    logprobs = old_logprobs + log_ratio_per_dim
    _, metrics = compute_decoupled_ppo_actor_loss(
        logprobs=logprobs,
        old_logprobs=old_logprobs,
        clip_ratio_low=0.2,
        clip_ratio_high=0.2,
        advantages=torch.full((BATCH, NUM_CHUNKS, 1), 1.0),
        loss_mask=torch.ones(BATCH, NUM_CHUNKS, 1).bool(),
        behave_weight_threshold=behave_weight_threshold,
    )
    return metrics


def test_decoupled_clip_fraction_is_a_fraction() -> None:
    """``clip_fraction`` on the decoupled path must also stay within [0, 1]."""
    metrics = _decoupled_metrics(log_ratio_per_dim=1.0)
    for key in ("actor/clip_fraction", "actor/dual_clip_fraction"):
        value = metrics[key].item() if torch.is_tensor(metrics[key]) else metrics[key]
        assert 0.0 <= value <= 1.0, f"{key}={value:.3f} is not a fraction"


def test_decoupled_proximal_approx_kl_is_the_per_dim_mean() -> None:
    """``proximal_approx_kl`` must not scale with ``action_dim``."""
    log_ratio = 0.01
    metrics = _decoupled_metrics(log_ratio_per_dim=log_ratio)
    assert metrics["actor/proximal_approx_kl"].item() == pytest.approx(
        -log_ratio, rel=1e-5
    )


@pytest.mark.parametrize("threshold", [None, 1e9])
def test_behav_clip_fraction_is_zero_when_nothing_is_dropped(
    threshold: float | None,
) -> None:
    """``behav_clip_fraction`` is 0 when the behaviour mask drops nothing.

    ``None`` reuses ``loss_mask`` directly (shape ``[bsz, chunks, 1]``) while a
    huge threshold builds a mask at the ratio's full width. Both keep every
    element, so both must report exactly 0 -- mixing the two widths in one
    quotient previously gave ``1 - 7 = -6``.
    """
    metrics = _decoupled_metrics(
        log_ratio_per_dim=0.01, behave_weight_threshold=threshold
    )
    value = metrics["actor/behav_clip_fraction"]
    value = value.item() if torch.is_tensor(value) else value
    assert value == pytest.approx(0.0, abs=1e-6)


def test_token_level_loss_scale_is_action_dim_times_the_per_dim_mean() -> None:
    """Pin the *loss* scale under token_level. This is characterization, not a fix.

    ``masked_mean`` is ``sum(values * mask) / sum(mask)``. When ``values`` is
    ``[bsz, chunks, action_dim]`` and ``mask`` is ``[bsz, chunks, 1]`` the
    numerator broadcasts over ``action_dim`` while the denominator does not, so
    the aggregated policy loss -- and every gradient through it -- is
    ``action_dim`` times a true per-dimension mean.

    This is deliberately left as is rather than "fixed":

    * it is a constant factor, and Adam's update is approximately invariant to a
      constant gradient scale, so the step size is governed by ``lr``;
    * ``clip_grad: 1.0`` is not reached either way (measured ``grad_norm`` is
      0.20-0.40 under token_level, so 7x smaller would still not clip);
    * every reference recipe in the repo -- ``libero_spatial_grpo_evo1.yaml``,
      the lingbotvla and openvlaoft configs -- had its ``lr`` chosen with this
      factor present, so removing it silently rescales all of them.

    The test exists so the factor is recorded and cannot change unnoticed.
    """
    old_logprobs = torch.zeros(BATCH, NUM_CHUNKS, ACTION_DIM)
    logprobs = old_logprobs + 0.01
    advantages = torch.full((BATCH, NUM_CHUNKS, 1), 1.0)
    kwargs = {
        "logprobs": logprobs,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
    }
    narrow, _ = compute_ppo_actor_loss(
        loss_mask=torch.ones(BATCH, NUM_CHUNKS, 1).bool(), **kwargs
    )
    wide, _ = compute_ppo_actor_loss(
        loss_mask=torch.ones(BATCH, NUM_CHUNKS, ACTION_DIM).bool(), **kwargs
    )
    assert narrow.item() == pytest.approx(ACTION_DIM * wide.item(), rel=1e-5)
