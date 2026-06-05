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

"""Action dimension mapping between XR0's 32D bimanual space and environment action spaces.

XR0 outputs 32D bimanual actions:
    [0:3]   left_ee_pos
    [3:6]   left_ee_axis_angle
    [6]     left_gripper
    [7:13]  left_joint (6 arm joints, NOT including gripper)
    [13]    reserved
    [14:17] right_ee_pos
    [17:20] right_ee_axis_angle
    [20]    right_gripper
    [21:27] right_joint (6 arm joints, NOT including gripper)
    [27:32] reserved

Different environments need different subsets of these 32 dimensions.
This module provides a config-driven mapper that:
1. Selects the relevant dimensions (32D -> env_dim)
2. Provides a valid_action_mask for loss masking (zeros out invalid dims)
3. Supports reverse mapping for chain replay (env_dim -> 32D)
"""

from __future__ import annotations

from typing import Optional

import torch


class ActionMapper:
    """Maps between XR0's 32D action space and an environment's action space.

    The mapper is defined by ``env_action_indices``: a list of 32D indices that
    the environment uses.  All other dimensions are masked as invalid.

    Example for SO101 dual-arm (12D = 5 arm joints + 1 gripper per arm)::

        env_action_indices = [
            7, 8, 9, 10, 11,    # left arm joints (XR0[7:12])
            6,                   # left gripper   (XR0[6])
            21, 22, 23, 24, 25,  # right arm joints (XR0[21:26])
            20,                  # right gripper  (XR0[20])
        ]

    Example for LIBERO single-arm (7D = ee_pos:3 + ee_aa:3 + gripper:1)::

        env_action_indices = [14, 15, 16, 17, 18, 19, 20]
        # right_ee_pos + right_ee_aa + right_gripper
    """

    def __init__(
        self,
        model_action_dim: int,
        env_action_indices: list[int],
    ):
        self.model_action_dim = model_action_dim
        self.env_action_indices = list(env_action_indices)
        self.env_action_dim = len(self.env_action_indices)

        # Validate indices
        for idx in self.env_action_indices:
            if not (0 <= idx < model_action_dim):
                raise ValueError(
                    f"Index {idx} out of range [0, {model_action_dim})"
                )

        # Build valid_action_mask: (model_action_dim,) with 1.0 for valid, 0.0 for invalid
        mask = torch.zeros(model_action_dim, dtype=torch.float32)
        mask[self.env_action_indices] = 1.0
        self._valid_action_mask = mask

        # Index tensor for gather (env_action_dim,)
        self._gather_indices = torch.tensor(
            self.env_action_indices, dtype=torch.long
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def valid_action_mask(self) -> torch.Tensor:
        """Float mask of shape ``(model_action_dim,)``.  1.0 = valid, 0.0 = invalid."""
        return self._valid_action_mask

    # ------------------------------------------------------------------
    # Forward: 32D -> env_dim
    # ------------------------------------------------------------------

    def map_to_env(self, actions: torch.Tensor) -> torch.Tensor:
        """Select valid dimensions from model actions.

        Args:
            actions: ``(..., model_action_dim)`` tensor.

        Returns:
            ``(..., env_action_dim)`` tensor with only valid dims.
        """
        indices = self._gather_indices.to(actions.device)
        return actions[..., indices]

    # ------------------------------------------------------------------
    # Reverse: env_dim -> 32D (scatter)
    # ------------------------------------------------------------------

    def map_to_model(self, actions_env: torch.Tensor) -> torch.Tensor:
        """Scatter env actions back into the full model action space.

        Invalid dimensions are filled with 0.0.

        Args:
            actions_env: ``(..., env_action_dim)`` tensor.

        Returns:
            ``(..., model_action_dim)`` tensor.
        """
        *leading_dims, env_dim = actions_env.shape
        device = actions_env.device
        indices = self._gather_indices.to(device)

        result = torch.zeros(
            *leading_dims, self.model_action_dim,
            device=device, dtype=actions_env.dtype,
        )
        # Expand indices for scatter
        idx_shape = [1] * len(leading_dims) + [env_dim]
        idx_expanded = indices.view(*idx_shape).expand_as(actions_env)
        result.scatter_(-1, idx_expanded, actions_env)
        return result

    # ------------------------------------------------------------------
    # Apply mask to logprobs / entropy
    # ------------------------------------------------------------------

    def apply_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Zero out invalid action dimensions.

        Args:
            tensor: ``(..., model_action_dim)`` logprobs or entropy.

        Returns:
            Same shape with invalid dims set to 0.0.
        """
        mask = self._valid_action_mask.to(
            device=tensor.device, dtype=tensor.dtype,
        )
        return tensor * mask

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, model_action_dim: int, config: dict) -> ActionMapper:
        """Create mapper from a config dict.

        The config must have ``env_action_indices`` (list of ints).

        Args:
            model_action_dim: Model's full action dimension (e.g. 32).
            config: Dict with ``env_action_indices``.

        Returns:
            An ``ActionMapper`` instance.
        """
        indices = config.get("env_action_indices")
        if indices is None:
            raise ValueError(
                "action_mapping config must have 'env_action_indices'"
            )
        return cls(model_action_dim=model_action_dim, env_action_indices=indices)


# ======================================================================
# Pre-defined mappings for common environments
# ======================================================================

ACTION_MAPPING_PRESETS = {
    # SO101 dual-arm: 5 arm joints + 1 gripper per arm = 12D
    "so101_dual": {
        "env_action_indices": [
            7, 8, 9, 10, 11,    # left arm joints  (XR0 left_joint[0:5])
            6,                   # left gripper     (XR0 left_gripper)
            21, 22, 23, 24, 25,  # right arm joints (XR0 right_joint[0:5])
            20,                  # right gripper    (XR0 right_gripper)
        ],
        "description": "SO101 dual-arm: 5 arm joints + 1 gripper per arm = 12D",
    },
    # LIBERO right-arm only: ee_pos:3 + ee_aa:3 + gripper:1 = 7D
    "libero_right_arm": {
        "env_action_indices": [14, 15, 16, 17, 18, 19, 20],
        "description": "LIBERO: right arm ee_pos + ee_aa + gripper = 7D",
    },
    # Full 32D (no masking, for debugging)
    "identity": {
        "env_action_indices": list(range(32)),
        "description": "Full 32D (no masking)",
    },
}


def get_action_mapper(
    model_action_dim: int = 32,
    preset: Optional[str] = None,
    env_action_indices: Optional[list[int]] = None,
) -> ActionMapper:
    """Get an ActionMapper by preset name or explicit indices.

    Args:
        model_action_dim: Model's full action dim (default 32).
        preset: Name of a preset in ``ACTION_MAPPING_PRESETS``.
        env_action_indices: Explicit list of 32D indices.  Takes precedence
            over *preset*.

    Returns:
        An ``ActionMapper`` instance.
    """
    if env_action_indices is not None:
        return ActionMapper(model_action_dim, env_action_indices)
    if preset is not None:
        if preset not in ACTION_MAPPING_PRESETS:
            raise ValueError(
                f"Unknown preset '{preset}'. "
                f"Available: {list(ACTION_MAPPING_PRESETS.keys())}"
            )
        return ActionMapper(
            model_action_dim,
            ACTION_MAPPING_PRESETS[preset]["env_action_indices"],
        )
    # Default: identity (no masking)
    return ActionMapper(model_action_dim, list(range(model_action_dim)))
