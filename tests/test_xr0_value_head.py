"""Test XR0 Value Head implementation."""

import torch
from omegaconf import OmegaConf

from rlinf.models import get_model


def test_value_head_disabled():
    """Test that value_head is None when add_value_head=False."""
    cfg = OmegaConf.create(
        {
            "model_type": "xr0",
            "model_path": "dummy",
            "precision": "bf16",
            "is_lora": False,
            "action_dim": 32,
            "num_action_chunks": 30,
            "num_steps": 5,
            "add_value_head": False,
            "xr0": {
                "state_shape": [1, 32],
                "action_shape": [30, 32],
            },
        }
    )

    model = get_model(cfg)
    assert not hasattr(model, "value_head") or model.value_head is None
    print("✅ test_value_head_disabled passed")


def test_value_head_enabled():
    """Test that value_head is initialized when add_value_head=True."""
    cfg = OmegaConf.create(
        {
            "model_type": "xr0",
            "model_path": "dummy",
            "precision": "bf16",
            "is_lora": False,
            "action_dim": 32,
            "num_action_chunks": 30,
            "num_steps": 5,
            "add_value_head": True,
            "xr0": {
                "state_shape": [1, 32],
                "action_shape": [30, 32],
            },
        }
    )

    model = get_model(cfg)
    assert hasattr(model, "value_head")
    assert model.value_head is not None
    assert model.add_value_head is True
    print("✅ test_value_head_enabled passed")


def test_get_value_from_vlm():
    """Test get_value_from_vlm with mock data."""
    cfg = OmegaConf.create(
        {
            "model_type": "xr0",
            "model_path": "dummy",
            "precision": "bf16",
            "is_lora": False,
            "action_dim": 32,
            "num_action_chunks": 30,
            "num_steps": 5,
            "add_value_head": True,
            "xr0": {
                "state_shape": [1, 32],
                "action_shape": [30, 32],
            },
        }
    )

    model = get_model(cfg)

    # Mock VLM hidden states: (B=2, S=10, D=2560)
    hidden_states = torch.randn(2, 10, 2560)
    attention_mask = torch.ones(2, 10)
    # Mask out some tokens
    attention_mask[0, 5:] = 0

    values = model.get_value_from_vlm(hidden_states, attention_mask)

    assert values.shape == (2,), f"Expected shape (2,), got {values.shape}"
    assert values.dtype == torch.float32
    print(f"✅ test_get_value_from_vlm passed, values: {values.tolist()}")


def test_default_forward_with_value_head():
    """Test default_forward returns real values when value_head is enabled."""
    cfg = OmegaConf.create(
        {
            "model_type": "xr0",
            "model_path": "dummy",
            "precision": "bf16",
            "is_lora": False,
            "action_dim": 32,
            "num_action_chunks": 30,
            "num_steps": 5,
            "add_value_head": True,
            "xr0": {
                "state_shape": [1, 32],
                "action_shape": [30, 32],
            },
        }
    )

    model = get_model(cfg)

    # Build mock forward_inputs with VLM hidden states
    batch_size = 2
    forward_inputs = {
        "vlm_hidden_states": torch.randn(batch_size, 10, 2560),
        "vlm_attention_mask": torch.ones(batch_size, 10),
        # Stub model doesn't need real chains/denoise_inds for this test
    }

    result = model.default_forward(forward_inputs=forward_inputs)

    assert "values" in result
    assert result["values"].shape == (batch_size,), f"Expected shape ({batch_size},), got {result['values'].shape}"
    # Values should not be all zeros when value_head is enabled
    assert not torch.all(result["values"] == 0), "Values should not be all zeros"
    print(f"✅ test_default_forward_with_value_head passed, values: {result['values'].tolist()}")


if __name__ == "__main__":
    test_value_head_disabled()
    test_value_head_enabled()
    test_get_value_from_vlm()
    test_default_forward_with_value_head()
    print("\n🎉 All value head tests passed!")
