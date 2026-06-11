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

"""Test XR0 replay chain: _compute_denoise_mean_std + get_logprob_norm.

Verifies the core replay logic used by default_forward during training.
Does NOT require real model weights or network access.
"""

import math

import torch

from rlinf.models.embodiment.xr0.xr0_action_model import XR0ForRLActionPrediction


def _make_policy(action_dim=16, num_steps=5):
    """Create a minimal policy (no real model needed for these tests)."""
    # We only need the static/class methods, so a real model isn't required.
    # Create a minimal instance with just the attributes used by the tested methods.
    policy = object.__new__(XR0ForRLActionPrediction)
    policy.action_dim = action_dim
    policy.num_steps = num_steps
    policy.noise_level = 0.1
    policy.noise_method = "flow_sde"
    return policy


def test_denoise_mean_std_train_flow_sde():
    """_compute_denoise_mean_std produces non-zero std in train+flow_sde mode."""
    policy = _make_policy(action_dim=16, num_steps=5)

    timesteps = torch.linspace(1.0, 0.0, policy.num_steps + 1)
    x_t = torch.randn(2, 10, 16)
    v_t = torch.randn(2, 10, 16)

    x_t_mean, x_t_std = policy._compute_denoise_mean_std(
        x_t, v_t, timesteps, idx=2, mode="train"
    )

    # std should be positive (flow_sde noise)
    assert x_t_std.abs().sum().item() > 0, "x_t_std should be non-zero in train mode"
    # mean should differ from x_t (Euler step + drift correction)
    assert (x_t_mean - x_t).abs().sum().item() > 0, "x_t_mean should differ from x_t"
    print("✅ test_denoise_mean_std_train_flow_sde passed")


def test_denoise_mean_std_eval():
    """_compute_denoise_mean_std produces zero std in eval mode."""
    policy = _make_policy(action_dim=16, num_steps=5)

    timesteps = torch.linspace(1.0, 0.0, policy.num_steps + 1)
    x_t = torch.randn(2, 10, 16)
    v_t = torch.randn(2, 10, 16)

    x_t_mean, x_t_std = policy._compute_denoise_mean_std(
        x_t, v_t, timesteps, idx=2, mode="eval"
    )

    assert x_t_std.abs().sum().item() == 0, "x_t_std should be zero in eval mode"
    assert (x_t_mean - x_t).abs().sum().item() > 0, "x_t_mean should differ from x_t"
    print("✅ test_denoise_mean_std_eval passed")


def test_get_logprob_norm_nonzero():
    """get_logprob_norm produces non-zero logprobs when sigma > 0."""
    policy = _make_policy()

    sample = torch.randn(2, 10, 16)
    mu = torch.randn(2, 10, 16)
    sigma = torch.full((2, 10, 16), 0.1)

    logprobs = XR0ForRLActionPrediction.get_logprob_norm(sample, mu, sigma)

    assert logprobs.abs().sum().item() > 0, "Logprobs should be non-zero"
    assert logprobs.shape == sample.shape, "Logprobs shape should match sample"
    print("✅ test_get_logprob_norm_nonzero passed")


def test_get_logprob_norm_zero_sigma():
    """get_logprob_norm returns 0 when sigma == 0 (deterministic step)."""
    policy = _make_policy()

    sample = torch.randn(2, 10, 16)
    mu = torch.randn(2, 10, 16)
    sigma = torch.zeros(2, 10, 16)

    logprobs = XR0ForRLActionPrediction.get_logprob_norm(sample, mu, sigma)

    assert logprobs.abs().sum().item() == 0, "Logprobs should be zero when sigma=0"
    print("✅ test_get_logprob_norm_zero_sigma passed")


def test_replay_chain_gradient_flow():
    """Full replay chain: denoise → logprob → backward works."""
    policy = _make_policy(action_dim=16, num_steps=5)

    timesteps = torch.linspace(1.0, 0.0, policy.num_steps + 1)
    batch_size = 2
    action_chunks = 10
    action_dim = 16

    # Simulate chains from sample_actions
    x_t = torch.randn(batch_size, action_chunks, action_dim)
    x_next = torch.randn(batch_size, action_chunks, action_dim)
    v_t = torch.randn(batch_size, action_chunks, action_dim, requires_grad=True)

    # Replay: compute mean/std, then logprob
    x_t_mean, x_t_std = policy._compute_denoise_mean_std(
        x_t, v_t, timesteps, idx=2, mode="train"
    )
    logprobs = XR0ForRLActionPrediction.get_logprob_norm(x_next, x_t_mean, x_t_std)

    # Verify gradient flow
    assert logprobs.requires_grad, "Logprobs should require grad"
    loss = logprobs.sum()
    loss.backward()
    assert v_t.grad is not None, "v_t should receive gradients"
    assert v_t.grad.abs().sum().item() > 0, "Gradients should be non-zero"
    print("✅ test_replay_chain_gradient_flow passed")


def test_replay_chain_sigma_increases_at_higher_noise():
    """sigma should increase with noise_level."""
    timesteps = torch.linspace(1.0, 0.0, 6)
    x_t = torch.randn(1, 5, 8)
    v_t = torch.randn(1, 5, 8)

    policy_low = _make_policy(action_dim=8, num_steps=5)
    policy_low.noise_level = 0.05
    _, std_low = policy_low._compute_denoise_mean_std(x_t, v_t, timesteps, idx=2, mode="train")

    policy_high = _make_policy(action_dim=8, num_steps=5)
    policy_high.noise_level = 0.5
    _, std_high = policy_high._compute_denoise_mean_std(x_t, v_t, timesteps, idx=2, mode="train")

    assert std_high.abs().mean().item() > std_low.abs().mean().item(), \
        "Higher noise_level should produce larger std"
    print("✅ test_replay_chain_sigma_increases_at_higher_noise passed")


if __name__ == "__main__":
    test_denoise_mean_std_train_flow_sde()
    test_denoise_mean_std_eval()
    test_get_logprob_norm_nonzero()
    test_get_logprob_norm_zero_sigma()
    test_replay_chain_gradient_flow()
    test_replay_chain_sigma_increases_at_higher_noise()
    print("\n✅ All replay chain tests passed")
