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

"""Xiaomi-Robotics-1 (XR-1) policy package."""

import os

import torch
from omegaconf import DictConfig


def get_model(cfg: DictConfig, torch_dtype=None):
    """Factory function to instantiate the XR-1 action model for RLinf."""

    from rlinf.models.embodiment.xr1.xr1_action_model import Xr1ActionModel

    if torch_dtype is None:
        torch_dtype = torch.bfloat16

    model = Xr1ActionModel(cfg, torch_dtype=torch_dtype)

    # Resuming from an RLinf FSDP checkpoint: the consolidated state dict lives
    # next to the HuggingFace export, so `model_path` can stay unchanged.
    checkpoint_dir = str(cfg.model_path)
    candidates = [
        os.path.join(checkpoint_dir, "model_state_dict", "full_weights.pt"),
        os.path.join(checkpoint_dir, "actor", "model_state_dict", "full_weights.pt"),
    ]
    for full_weights_path in candidates:
        if os.path.exists(full_weights_path):
            print(f"[XR-1] Loading RLinf FSDP weights from {full_weights_path}")
            state_dict = torch.load(full_weights_path, map_location="cpu")
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if unexpected:
                print(f"[XR-1] Unexpected keys in checkpoint: {len(unexpected)}")
            if missing:
                print(f"[XR-1] Keys kept from the base model: {len(missing)}")
            break

    return model


__all__ = ["get_model"]
