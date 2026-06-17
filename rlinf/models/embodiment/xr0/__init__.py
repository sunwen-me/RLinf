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

"""XR0 VLA embodied policy wrapper for RLinf.

This module exposes ``get_model``, which instantiates the XR0 model and wraps
it into an ``XR0ForRLActionPrediction`` instance compatible with RLinf.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig

from rlinf.utils.logging import get_logger

from .action_mapping import ActionMapper, get_action_mapper
from .utils import ACTION_DIM
from .xr0_action_model import XR0ForRLActionPrediction


class _StubXR0(nn.Module):
    """Lightweight stub that mimics the XR0 interface without downloading weights.

    Used when ``model_path`` is ``"dummy"`` so tests and CI can run without
    network access or large model downloads.
    """

    def __init__(self, action_shape=(30, 32), num_steps=5):
        super().__init__()
        self.action_shape = action_shape
        self.num_steps = num_steps
        # Minimal parameter so the module is not empty
        self._dummy = nn.Linear(1, 1)

    @torch.no_grad()
    def generate(self, batch: dict) -> torch.Tensor:
        """Return random action predictions matching the expected shape."""
        # Infer batch size from any tensor in the batch
        batch_size = 1
        for v in batch.values():
            if isinstance(v, torch.Tensor) and v.ndim > 0:
                batch_size = v.shape[0]
                break
        device = self._dummy.weight.device
        return torch.randn(
            (batch_size, *self.action_shape),
            device=device,
            dtype=torch.bfloat16,
        )


def get_model(
    cfg: DictConfig,
    torch_dtype: torch.dtype | None = None,
) -> XR0ForRLActionPrediction:
    """Instantiate the XR0 model and wrap it for RLinf.

    When ``cfg.model_path`` is ``"dummy"``, a lightweight stub is used instead
    of the full XR0 model (no HuggingFace download required).

    Args:
        cfg: Model config.  Expected keys include ``action_dim``,
            ``num_action_chunks``, ``num_steps``, and XR0-specific params
            under ``cfg.xr0``.
        torch_dtype: Optional torch dtype for the model.

    Returns:
        An ``XR0ForRLActionPrediction`` instance.
    """
    logger = get_logger()

    action_dim = getattr(cfg, "action_dim", ACTION_DIM)
    num_action_chunks = getattr(cfg, "num_action_chunks", 30)
    num_steps = getattr(cfg, "num_steps", 5)
    noise_level = getattr(cfg, "noise_level", 0.5)
    noise_method = getattr(cfg, "noise_method", "flow_sde")
    action_env_dim = getattr(cfg, "action_env_dim", None)

    # robot_type: determines which action_mask and decode_action stats to use.
    # Must match a key in processor.get_action_mask() (e.g. "libero_all",
    # "so101_dual").  For LIBERO the default "libero_all" is correct.
    # For SO101, set robot_type: "so101_dual" in the model YAML.
    robot_type = getattr(cfg, "robot_type", "libero_all")

    # XR0-specific config (with defaults matching the original XR0 config)
    xr0_cfg = getattr(cfg, "xr0", cfg)
    local_window = getattr(xr0_cfg, "local_window", 4)
    async_train = getattr(xr0_cfg, "async_train", False)
    training_repeat = getattr(xr0_cfg, "training_repeat", 1)
    freq_coefficient = getattr(xr0_cfg, "freq_coefficient", 0.0)
    action_shape = tuple(
        getattr(xr0_cfg, "action_shape", [num_action_chunks, action_dim])
    )

    model_path = getattr(cfg, "model_path", None)

    if model_path == "dummy":
        logger.info("Using stub XR0 model (model_path=dummy)")
        xr0_model = _StubXR0(action_shape=action_shape, num_steps=num_steps)
    else:
        from .model.modeling_mibot import MiBoTForActionGeneration

        logger.info("Loading XR0 model from %s", model_path)
        _dtype = torch_dtype or torch.bfloat16
        xr0_model = MiBoTForActionGeneration.from_pretrained(
            model_path,
            torch_dtype=_dtype,
        )

    # Load action normalization stats (optional)
    action_mean, action_std = None, None
    stats_path = getattr(cfg, "stats_path", None) or getattr(
        xr0_cfg, "stats_path", None
    )
    if stats_path:
        import yaml

        logger.info("Loading action normalization stats from %s", stats_path)
        with open(stats_path) as f:
            stats = yaml.safe_load(f)
        action_mean = np.array(stats["mean"], dtype=np.float32)
        action_std = np.array(stats["std"], dtype=np.float32)

    # Value head for PPO critic
    add_value_head = getattr(cfg, "add_value_head", False) or getattr(
        xr0_cfg, "add_value_head", False
    )

    # Action mapper: maps between model's 32D space and env action space
    # with valid_action_mask for loss masking.
    action_mapper = None
    mapping_cfg = getattr(cfg, "action_mapping", None) or getattr(
        xr0_cfg, "action_mapping", None
    )
    if mapping_cfg is not None:
        mapping_cfg = dict(mapping_cfg)
        preset = mapping_cfg.pop("preset", None)
        indices = mapping_cfg.pop("env_action_indices", None)
        action_mapper = get_action_mapper(
            model_action_dim=action_dim,
            preset=preset,
            env_action_indices=indices,
        )
        logger.info(
            "ActionMapper: preset=%s, env_action_dim=%d, valid dims=%s",
            preset, action_mapper.env_action_dim, action_mapper.env_action_indices,
        )

    train_expert_only = getattr(cfg, "train_expert_only", False) or getattr(
        xr0_cfg, "train_expert_only", False
    )

    # Validate: current implementation uses cached VLM KV from rollout in
    # default_forward().  This means VLM gradients are NOT computed during
    # training, so train_expert_only must be True.  If VLM training is
    # needed, default_forward() must re-run VLM forward with gradients.
    if not train_expert_only and model_path != "dummy":
        logger.warning(
            "train_expert_only=False but XR0's default_forward uses cached "
            "VLM KV (no VLM gradients). Setting train_expert_only=True. "
            "To train VLM, modify default_forward to re-run VLM forward."
        )
        train_expert_only = True

    policy = XR0ForRLActionPrediction(
        xr0_model=xr0_model,
        action_dim=action_dim,
        num_action_chunks=num_action_chunks,
        num_steps=num_steps,
        action_mean=action_mean,
        action_std=action_std,
        noise_level=noise_level,
        model_path=model_path if model_path != "dummy" else None,
        add_value_head=add_value_head,
        noise_method=noise_method,
        action_env_dim=action_env_dim,
        action_mapper=action_mapper,
        train_expert_only=train_expert_only,
        local_window=local_window,
        async_train=async_train,
        training_repeat=training_repeat,
        freq_coefficient=freq_coefficient,
        robot_type=robot_type,
    )

    logger.info("XR0 robot_type=%s, action_env_dim=%s", robot_type, action_env_dim)

    return policy


__all__ = ["XR0ForRLActionPrediction", "get_model"]
