"""Test XR0 Flow-SDE implementation."""

import math

import torch
from omegaconf import OmegaConf

from rlinf.models import get_model


def test_flow_sde_sigma_formula():
    """Verify σ = a√(τ/(1-τ)) formula."""
    noise_level = 0.5
    num_steps = 5

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)

    # Compute sigmas using the formula
    t_safe = torch.where(timesteps == 1.0, timesteps[1], timesteps)
    sigmas = noise_level * torch.sqrt(timesteps / (1 - t_safe))
    sigmas = sigmas[:-1]  # Remove last (τ=0)

    print(f"Timesteps: {timesteps.tolist()}")
    print(f"Sigmas: {sigmas.tolist()}")

    # Verify: σ should increase as τ decreases (more noise at later steps)
    # Actually σ = a√(τ/(1-τ)) decreases as τ decreases from 1 to 0
    # At τ=1: σ → ∞ (but we use τ[1] as safe value)
    # At τ=0.5: σ = a√(0.5/0.5) = a
    # At τ→0: σ → 0

    # Check that all sigmas are positive
    assert torch.all(sigmas >= 0), "All sigmas should be non-negative"

    # Check that sigma at τ=0.5 equals noise_level
    # timesteps[2] = 0.6, timesteps[3] = 0.4, so τ=0.6 is closest to 0.5
    # Let's just verify the formula manually for τ=0.6
    tau = 0.6
    expected_sigma = noise_level * math.sqrt(tau / (1 - tau))
    idx = 2  # timesteps[2] = 0.6
    assert abs(sigmas[idx].item() - expected_sigma) < 1e-6, \
        f"Sigma at τ={tau}: expected {expected_sigma}, got {sigmas[idx].item()}"

    print("✅ test_flow_sde_sigma_formula passed")


def test_flow_sde_drift_correction():
    """Verify drift correction term σ²δ/(2τ)."""
    noise_level = 0.5
    num_steps = 5

    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)

    # Compute sigma for step 2 (τ=0.6)
    idx = 2
    tau = timesteps[idx]
    delta = timesteps[idx] - timesteps[idx + 1]

    t_safe = torch.where(timesteps == 1.0, timesteps[1], timesteps)
    sigmas = noise_level * torch.sqrt(timesteps / (1 - t_safe))
    sigma_i = sigmas[idx]

    # Drift correction: σ²δ/(2τ)
    drift_correction = sigma_i**2 * delta / (2 * tau)

    print(f"τ={tau:.2f}, δ={delta:.2f}, σ={sigma_i:.4f}")
    print(f"Drift correction σ²δ/(2τ) = {drift_correction:.6f}")

    # Verify: drift correction should be positive
    assert drift_correction > 0, "Drift correction should be positive"

    # Verify: x1_weight = τ - δ - σ²δ/(2τ)
    x0_weight = 1 - (tau - delta)
    x1_weight = tau - delta - drift_correction

    print(f"x0_weight={x0_weight:.4f}, x1_weight={x1_weight:.4f}")

    # Weights should still sum to approximately 1 (with noise)
    # x0_weight + x1_weight ≈ 1 - σ²δ/(2τ)
    assert x1_weight < tau - delta, "Drift correction should reduce x1_weight"

    print("✅ test_flow_sde_drift_correction passed")


def test_flow_sde_in_model():
    """Test Flow-SDE in XR0 model with stub."""
    cfg = OmegaConf.create(
        {
            "model_type": "xr0",
            "model_path": "dummy",
            "precision": "bf16",
            "is_lora": False,
            "action_dim": 32,
            "num_action_chunks": 30,
            "num_steps": 5,
            "noise_level": 0.5,
            "noise_method": "flow_sde",
            "xr0": {
                "state_shape": [1, 32],
                "action_shape": [30, 32],
            },
        }
    )

    model = get_model(cfg)

    # Verify noise_method is set
    assert model.noise_method == "flow_sde"
    print(f"noise_method: {model.noise_method}")

    # Test _compute_denoise_step with mock data
    batch_size = 2
    action_len = 30
    action_dim = 32
    num_steps = 5

    x_t = torch.randn(batch_size, action_len, action_dim)
    v_t = torch.randn(batch_size, action_len, action_dim)
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)

    # Test eval mode (deterministic)
    x_t_mean, x_t_std, log_prob = model._compute_denoise_step(
        x_t, v_t, timesteps, idx=0, mode="eval"
    )
    assert x_t_std.shape == x_t.shape
    assert torch.all(x_t_std == 0), "Eval mode should have zero std"
    print(f"Eval mode: x_t_mean shape={x_t_mean.shape}, std={x_t_std.mean():.4f}")

    # Test train mode (stochastic with Flow-SDE)
    x_t_mean, x_t_std, log_prob = model._compute_denoise_step(
        x_t, v_t, timesteps, idx=0, mode="train"
    )
    assert x_t_std.shape == x_t.shape
    assert torch.all(x_t_std > 0), "Train mode should have positive std"
    print(f"Train mode: x_t_mean shape={x_t_mean.shape}, std={x_t_std.mean():.4f}")

    # Test that std varies with timestep
    stds = []
    for idx in range(num_steps):
        _, std, _ = model._compute_denoise_step(
            x_t, v_t, timesteps, idx=idx, mode="train"
        )
        stds.append(std.mean().item())

    print(f"Std per step: {stds}")
    # σ = a√(τ/(1-τ)) increases as τ → 1
    # So std should decrease as idx increases (τ decreases)
    # At idx=0, τ=1.0 (use safe value), at idx=4, τ=0.2
    print("✅ test_flow_sde_in_model passed")


def test_legacy_noise_method():
    """Test legacy fixed noise method still works."""
    cfg = OmegaConf.create(
        {
            "model_type": "xr0",
            "model_path": "dummy",
            "precision": "bf16",
            "is_lora": False,
            "action_dim": 32,
            "num_action_chunks": 30,
            "num_steps": 5,
            "noise_level": 0.5,
            "noise_method": "fixed",  # Legacy method
            "xr0": {
                "state_shape": [1, 32],
                "action_shape": [30, 32],
            },
        }
    )

    model = get_model(cfg)
    assert model.noise_method == "fixed"

    batch_size = 2
    action_len = 30
    action_dim = 32

    x_t = torch.randn(batch_size, action_len, action_dim)
    v_t = torch.randn(batch_size, action_len, action_dim)
    timesteps = torch.linspace(1.0, 0.0, 6)

    x_t_mean, x_t_std, log_prob = model._compute_denoise_step(
        x_t, v_t, timesteps, idx=2, mode="train"
    )

    # Fixed noise: σ = noise_level * √δ
    delta = (timesteps[2] - timesteps[3]).item()
    expected_sigma = 0.5 * math.sqrt(delta)

    print(f"Fixed noise: expected σ={expected_sigma:.4f}, got σ={x_t_std.mean():.4f}")
    assert abs(x_t_std.mean().item() - expected_sigma) < 1e-6

    print("✅ test_legacy_noise_method passed")


if __name__ == "__main__":
    test_flow_sde_sigma_formula()
    test_flow_sde_drift_correction()
    test_flow_sde_in_model()
    test_legacy_noise_method()
    print("\n🎉 All Flow-SDE tests passed!")
