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

"""Pytest for XR0 model registration and skeleton interface."""

from omegaconf import OmegaConf

from rlinf.config import EMBODIED_MODEL, SupportedModel


def test_xr0_supported_model_registered():
    """Verify xr0 is registered in SupportedModel."""
    model = SupportedModel("xr0")
    assert model.value == "xr0"
    assert "xr0" in SupportedModel.models


def test_xr0_in_embodied_model_set():
    """Verify xr0 is in the EMBODIED_MODEL set."""
    xr0_model = SupportedModel("xr0")
    assert xr0_model in EMBODIED_MODEL


def test_xr0_model_registry():
    """Verify xr0 is registered in the model registry and get_model returns a policy."""
    from rlinf.models import _MODEL_REGISTRY, get_model

    assert "xr0" in _MODEL_REGISTRY

    cfg = OmegaConf.create(
        {
            "model_type": "xr0",
            "model_path": "dummy",
            "precision": "bf16",
            "is_lora": False,
            "action_dim": 32,
            "num_action_chunks": 30,
            "num_steps": 5,
            "xr0": {
                "state_shape": [1, 32],
                "action_shape": [30, 32],
                "dit_num_layers": 2,
                "dit_hidden_size": 64,
                "local_window": 4,
                "training_repeat": 1,
                "enable_freq": False,
                "flow_sampling": "uniform",
            },
        }
    )

    model = get_model(cfg)
    assert model is not None

    # Verify it implements the BasePolicy interface
    from rlinf.models.embodiment.base_policy import BasePolicy

    assert isinstance(model, BasePolicy)
    assert hasattr(model, "predict_action_batch")
    assert hasattr(model, "default_forward")
    assert hasattr(model, "_no_split_modules")
    assert hasattr(model, "_no_split_names")


def test_xr0_default_forward_returns_stub():
    """Verify default_forward returns the expected stub dict."""
    from rlinf.models import get_model

    cfg = OmegaConf.create(
        {
            "model_type": "xr0",
            "model_path": "dummy",
            "precision": "bf16",
            "is_lora": False,
            "action_dim": 32,
            "num_action_chunks": 30,
            "num_steps": 5,
            "xr0": {
                "state_shape": [1, 32],
                "action_shape": [30, 32],
                "dit_num_layers": 2,
                "dit_hidden_size": 64,
                "local_window": 4,
                "training_repeat": 1,
                "enable_freq": False,
                "flow_sampling": "uniform",
            },
        }
    )

    model = get_model(cfg)
    result = model.default_forward()

    assert "logprobs" in result
    assert "values" in result
    assert "entropy" in result
    # After commit e637c186, logprobs/entropy return full (B, C, D) shape
    # so loss functions can do their own aggregation.
    assert result["logprobs"].shape == (1, 30, 32)
    assert result["values"].shape == (1,)
    assert result["entropy"].shape == (1, 30, 32)


def test_xr0_real_model_with_generate_not_stub():
    """Verify a model that has generate() is NOT treated as stub.

    Real XR0 models (MiBoTForActionGeneration) inherit from
    PreTrainedModel which exposes generate().  The old hasattr check
    would incorrectly treat them as stubs, bypassing VLM/DiT inference.
    Only models with the explicit _rlinf_is_stub_xr0 marker should be stubs.
    """
    import torch.nn as nn

    from rlinf.models.embodiment.xr0 import _StubXR0
    from rlinf.models.embodiment.xr0.xr0_action_model import XR0ForRLActionPrediction

    # Model with generate() but no stub marker — should NOT be stub
    class FakeRealModel(nn.Module):
        def generate(self, **kwargs):
            raise NotImplementedError("This is a real model, not a stub")

    fake_model = FakeRealModel()
    policy = XR0ForRLActionPrediction(
        xr0_model=fake_model,
        action_dim=32,
        num_action_chunks=10,
        num_steps=5,
    )
    assert policy._is_stub is False, (
        "Real model with generate() should not be detected as stub"
    )

    # Explicit stub model — should be stub
    stub_model = _StubXR0()
    policy_stub = XR0ForRLActionPrediction(
        xr0_model=stub_model,
        action_dim=32,
        num_action_chunks=10,
        num_steps=5,
    )
    assert policy_stub._is_stub is True, (
        "_StubXR0 should be detected as stub"
    )
