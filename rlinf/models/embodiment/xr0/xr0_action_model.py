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

"""RLinf policy wrapper for the XR0 VLA model.

XR0 uses Qwen3-VL as the vision-language backbone and a DiT with rectified
flow for continuous action prediction.  This module adapts the raw XR0 model
to RLinf's ``BasePolicy`` interface for rollout and training.
"""

from __future__ import annotations

import math
import os
import random
from contextlib import contextmanager
from typing import Any, Literal, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoProcessor

from rlinf.models.embodiment.base_policy import BasePolicy
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger

from .action_mapping import ActionMapper, get_action_mapper
from .utils import ACTION_DIM, denormalize_action, resize_image


@contextmanager
def _temporarily_eval(module: torch.nn.Module):
    """Temporarily set module to eval mode (disables dropout/BN randomness).

    Eval mode does NOT imply no_grad — gradients can still flow.
    This is important for PPO/GRPO where policy stochasticity should come
    from the Flow-SDE noise, not from dropout.
    """
    was_training = module.training
    module.eval()
    try:
        yield
    finally:
        module.train(was_training)


class XR0ForRLActionPrediction(nn.Module, BasePolicy):
    """RLinf policy wrapper for XR0 VLA checkpoints.

    Wraps the XR0 model (Qwen3-VL + DiT) and exposes the
    ``predict_action_batch`` / ``default_forward`` interface required by
    RLinf's rollout and actor workers.

    Args:
        xr0_model: The XR0 model instance (stub or real).
        action_dim: Action dimensionality (default 32).
        num_action_chunks: Number of action timesteps per chunk (default 30).
        num_steps: Number of rectified flow denoising steps.
        action_mean: Per-timestep action mean for denormalization.
        action_std: Per-timestep action std for denormalization.
        noise_level: Noise level for flow-SDE (default 0.5).

    TODO: Add π_RL-style learnable noise network (ExploreNoiseNet) for
    Flow-Noise method. Currently uses fixed noise_level (flow_sde).
    See lingbotvla for reference implementation with flow_noise.
    """

    def __init__(
        self,
        xr0_model: nn.Module,
        action_dim: int = ACTION_DIM,
        num_action_chunks: int = 30,
        num_steps: int = 5,
        action_mean: Optional[np.ndarray] = None,
        action_std: Optional[np.ndarray] = None,
        noise_level: float = 0.5,
        model_path: Optional[str] = None,
        add_value_head: bool = False,
        noise_method: str = "flow_sde",
        action_env_dim: Optional[int] = None,
        action_mapper: Optional[ActionMapper] = None,
        local_window: int = 4,
        async_train: bool = False,
        training_repeat: int = 1,
        freq_coefficient: float = 0.0,
        train_expert_only: bool = False,
        robot_type: str = "libero_all",
    ):
        super().__init__()
        self.logger = get_logger()

        self.xr0_model = xr0_model
        self.action_dim = int(action_dim)
        self.num_action_chunks = int(num_action_chunks)
        self.num_steps = int(num_steps)
        self.noise_level = float(noise_level)
        self.model_path = model_path
        self.add_value_head = add_value_head
        self.noise_method = noise_method
        # Must match Xiaomi server input_data["task_id"].
        # Used by processor.get_action_mask(...) and processor.decode_action(...).
        self.robot_type = robot_type
        self.local_window = local_window
        self.async_train = async_train
        self.training_repeat = training_repeat
        self.freq_coefficient = freq_coefficient
        self.prefix_mask_prob = 0.5

        # Action mapper: handles 32D <-> env_dim conversion with valid_action_mask.
        # Takes precedence over action_env_dim (simple slicing).
        if action_mapper is not None:
            self.action_mapper = action_mapper
            self.action_env_dim = action_mapper.env_action_dim
        else:
            self.action_mapper = None
            # Fallback: simple slicing to first N dims
            self.action_env_dim = int(action_env_dim) if action_env_dim else int(action_dim)

        # Freeze VLM backbone when train_expert_only (like pi0/pi0.5/lingbotvla).
        self.train_expert_only = train_expert_only
        if train_expert_only:
            self.freeze_vlm()

        # Action normalization stats
        self.register_buffer(
            "action_mean",
            torch.from_numpy(action_mean).float() if action_mean is not None else None,
        )
        self.register_buffer(
            "action_std",
            torch.from_numpy(action_std).float() if action_std is not None else None,
        )

        # Qwen3-VL processor (lazy-loaded on first use)
        self._processor = None

        # Action mask from processor: (1, num_action_chunks, action_dim)
        # Loaded lazily on first use via _get_action_mask().
        self._action_mask: torch.Tensor | None = None
        self._action_mask_robot_type: str | None = None

        # Cached VLM KV from the most recent sample_actions call.
        # Only used as fallback when forward_inputs doesn't have packed KV,
        # and only when batch sizes match exactly.
        # Value head for PPO critic (uses VLM hidden states)
        if add_value_head:
            # Get VLM hidden size from model config
            vlm_hidden_size = self._get_vlm_hidden_size()
            self.value_head = ValueHead(
                input_dim=vlm_hidden_size,
                hidden_sizes=(512, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            )
            # Match VLM dtype (typically bfloat16)
            vlm_dtype = self._get_vlm_dtype()
            self.value_head = self.value_head.to(dtype=vlm_dtype)
            self.logger.info(
                "ValueHead initialized with input_dim=%d, dtype=%s",
                vlm_hidden_size, vlm_dtype,
            )

        # FSDP wrap name for every submodule
        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

    # ------------------------------------------------------------------
    # FSDP hints
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # VLM KV cache packing helpers (for forward_inputs transport)
    # ------------------------------------------------------------------

    @staticmethod
    def _pack_vlm_kv(
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> dict[str, torch.Tensor]:
        """Flatten VLM KV cache into a dict of tensors for storage."""
        packed: dict[str, torch.Tensor] = {}
        for i, (k, v) in enumerate(past_key_values):
            packed[f"vlm_kv_{i}_k"] = k.detach().cpu()
            packed[f"vlm_kv_{i}_v"] = v.detach().cpu()
        return packed

    @staticmethod
    def _unpack_vlm_kv(
        forward_inputs: dict[str, torch.Tensor],
        device: torch.device,
    ) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
        """Reconstruct VLM KV cache from packed tensors.

        Infers the number of layers by counting ``vlm_kv_{i}_k`` keys.
        """
        past_key_values = []
        i = 0
        while True:
            k = forward_inputs.get(f"vlm_kv_{i}_k")
            v = forward_inputs.get(f"vlm_kv_{i}_v")
            if k is None or v is None:
                break
            past_key_values.append((k.to(device), v.to(device)))
            i += 1
        return past_key_values if past_key_values else None

    @staticmethod
    def _pack_vlm_pos_max(
        pos_max: torch.Tensor,
        batch_size: int,
    ) -> torch.Tensor:
        """Store vlm_pos_max as (B, 3) so RLinf micro-batch splitting slices
        the batch dimension correctly (not the M-RoPE dim)."""
        if pos_max.ndim == 2 and pos_max.shape[0] == 3 and pos_max.shape[1] == batch_size:
            return pos_max.transpose(0, 1).contiguous().detach().cpu()
        return pos_max.detach().cpu()

    @staticmethod
    def _unpack_vlm_pos_max(
        pos_max: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Restore vlm_pos_max from (B, 3) back to (3, B)."""
        pos_max = pos_max.to(device)
        if pos_max.ndim == 2 and pos_max.shape[0] == batch_size and pos_max.shape[1] == 3:
            return pos_max.transpose(0, 1).contiguous()
        return pos_max

    def freeze_vlm(self):
        """Freeze the VLM backbone (Qwen3-VL), only train DiT + projectors.

        Follows the pi0/pi0.5/lingbotvla pattern: train_expert_only freezes
        the vision-language model and only trains the action prediction head.
        """
        vlm = self.xr0_model.vlm
        vlm.eval()
        vlm_params = 0
        for param in vlm.parameters():
            param.requires_grad = False
            vlm_params += param.numel()
        self.logger.info(
            "[freeze_vlm] Frozen VLM (Qwen3-VL): %.1fM params", vlm_params / 1e6
        )

        # Log trainable params
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        self.logger.info(
            "[freeze_vlm] Trainable: %.1fM / %.1fB (%.1f%%)",
            trainable_params / 1e6,
            total_params / 1e9,
            trainable_params / total_params * 100,
        )

    @property
    def processor(self):
        """Lazy-load Qwen3-VL processor on first access.

        Loads from model_path if available (local weights), otherwise
        falls back to Qwen/Qwen3-VL-4B-Instruct from HuggingFace.
        """
        if self._processor is None:
            # Try loading from local model path first
            source = self.model_path if self.model_path else "Qwen/Qwen3-VL-4B-Instruct"
            self.logger.info("Loading processor from %s", source)
            self._processor = AutoProcessor.from_pretrained(
                source, trust_remote_code=True, use_fast=False
            )
            self._processor.tokenizer.padding_side = "right"
        return self._processor

    def _get_action_mask(
        self,
        device: torch.device,
        robot_type: Optional[str] = None,
    ) -> torch.Tensor:
        """Load action_mask from processor and cache it per robot_type.

        Xiaomi server uses ``input_data["task_id"]`` for both
        ``processor.get_action_mask(robot_type)`` and
        ``processor.decode_action(..., robot_type=robot_type)``.
        Using a hard-coded LIBERO mask for SO101/ManiSkill will decode
        actions with the wrong statistics and can produce huge actions.
        """
        requested_robot_type = robot_type or self.robot_type

        if (
            self._action_mask is None
            or self._action_mask_robot_type != requested_robot_type
        ):
            proc = self.processor
            if hasattr(proc, "get_action_mask"):
                available = list(proc.list_robot_types()) if hasattr(proc, "list_robot_types") else []
                if available and requested_robot_type not in available:
                    self.logger.warning(
                        "Requested robot_type=%s is not in processor.list_robot_types()=%s; "
                        "calling get_action_mask anyway.",
                        requested_robot_type, available,
                    )
                mask = proc.get_action_mask(requested_robot_type)
                self.logger.info(
                    "Loaded action_mask from processor: robot_type=%s, shape=%s",
                    requested_robot_type, tuple(mask.shape),
                )
            else:
                mask = torch.ones(
                    1, self.num_action_chunks, self.action_dim, dtype=torch.float32
                )
                self.logger.warning(
                    "Processor has no get_action_mask; using all-ones mask"
                )
            self._action_mask = mask
            self._action_mask_robot_type = requested_robot_type
        return self._action_mask.to(device=device, dtype=torch.bfloat16)

    @property
    def _no_split_modules(self) -> list[str]:
        return [
            "DecoderLayer",
            "Qwen3VLTextDecoderLayer",
            "Qwen3VLVisionBlock",
        ]

    @property
    def _no_split_names(self) -> list[str]:
        return [
            "vlm",
            "dit",
            "state_projector",
            "action_projector",
            "action_output_layer",
        ]

    # ------------------------------------------------------------------
    # Value head helpers (for RL critic)
    # ------------------------------------------------------------------

    def _get_vlm_hidden_size(self) -> int:
        """Get the hidden size of the VLM language model.

        Returns:
            Hidden dimension of the VLM's text backbone (e.g., 2560 for Qwen3-VL-4B).
        """
        # Try to get from VLM config
        if hasattr(self.xr0_model, "vlm"):
            vlm = self.xr0_model.vlm
            config = vlm.config
            # Qwen3VLConfig has text_config with hidden_size
            if hasattr(config, "text_config"):
                return config.text_config.hidden_size
            # Direct config
            if hasattr(config, "hidden_size"):
                return config.hidden_size
        # Fallback: try to infer from state_projector output
        if hasattr(self.xr0_model, "state_projector"):
            for module in self.xr0_model.state_projector.modules():
                if hasattr(module, "out_features"):
                    return module.out_features
        # Default for Qwen3-VL-4B
        return 2560

    def _get_vlm_dtype(self) -> torch.dtype:
        """Get the dtype of the VLM model.

        Returns:
            torch.dtype of the VLM parameters (e.g., torch.bfloat16).
        """
        if hasattr(self.xr0_model, "vlm"):
            try:
                return next(self.xr0_model.vlm.parameters()).dtype
            except StopIteration:
                pass
        # Default to bfloat16 for XR0
        return torch.bfloat16

    def get_value_from_vlm(
        self,
        vlm_hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Compute value estimate from VLM hidden states.

        Uses masked mean-pooling over the sequence dimension, then passes
        through the value head MLP to produce a scalar value per batch.

        Args:
            vlm_hidden_states: ``(B, S, D)`` VLM output hidden states.
            attention_mask: ``(B, S)`` attention mask (1 for valid, 0 for pad).

        Returns:
            ``(B,)`` value estimates.
        """
        # Masked mean-pooling: (B, S, D) -> (B, D)
        mask = attention_mask.unsqueeze(-1).to(vlm_hidden_states.dtype)
        sum_hidden = (vlm_hidden_states * mask).sum(dim=1)
        valid_token_count = mask.sum(dim=1).clamp(min=1e-6)
        pooled = sum_hidden / valid_token_count

        # Ensure value_head is on the same device and dtype as input
        device = pooled.device
        dtype = pooled.dtype
        if next(self.value_head.parameters()).device != device:
            self.value_head = self.value_head.to(device)
        if next(self.value_head.parameters()).dtype != dtype:
            self.value_head = self.value_head.to(dtype=dtype)

        # Value head: (B, D) -> (B, 1) -> (B,)
        values = self.value_head(pooled).squeeze(-1)
        return values

    # ------------------------------------------------------------------
    # State dimension handling
    # ------------------------------------------------------------------

    def _get_expected_state_dim(self) -> int:
        """Get the expected state dimension from the model's state_projector."""
        if hasattr(self.xr0_model, "state_projector"):
            # Get input dimension from the first linear layer
            for module in self.xr0_model.state_projector.modules():
                if isinstance(module, torch.nn.Linear):
                    return module.in_features
        # Default XR0 state dimension
        return 32

    def _pad_state(self, state: torch.Tensor) -> torch.Tensor:
        """Pad or truncate state to match model's expected dimension.

        This handles different environments producing different state dims:
        - LIBERO: 8D (eef_pos:3 + eef_quat:3 + gripper:2)
        - ManiSkill: 32D
        - etc.

        Args:
            state: (B, 1, D) state tensor from environment

        Returns:
            (B, 1, expected_D) state tensor padded/truncated to model's expected dim
        """
        expected_dim = self._get_expected_state_dim()
        current_dim = state.shape[-1]

        if current_dim == expected_dim:
            return state
        elif current_dim < expected_dim:
            # Pad with zeros
            padding = torch.zeros(
                *state.shape[:-1], expected_dim - current_dim,
                device=state.device, dtype=state.dtype
            )
            return torch.cat([state, padding], dim=-1)
        else:
            # Truncate
            return state[..., :expected_dim]

    def _random_mask_prefix(
        self,
        causal_mask: torch.Tensor,
        prefix_length: int,
        state_length: int,
        keep_last_k: int = 2,
    ) -> torch.Tensor:
        """Randomly mask prefix tokens in the causal mask for async training.

        Prevents the model from directly copying prefix values — it must
        understand the action sequence semantically.  Keeps the last
        ``keep_last_k`` prefix tokens always visible (most recent execution
        results are most important).

        Args:
            causal_mask: ``(B, q_len, q_len)`` attention mask.
            prefix_length: Number of prefix (already-executed) tokens.
            state_length: Number of state tokens (excluding sink).
            keep_last_k: Number of prefix tokens to always keep visible.

        Returns:
            Modified causal mask with some prefix tokens masked out.
        """
        if prefix_length <= keep_last_k:
            return causal_mask

        action_start = 1 + state_length  # +1 for sink token
        masked_prefix_end = action_start + prefix_length - keep_last_k
        suffix_start = action_start + prefix_length

        if suffix_start >= causal_mask.shape[-1]:
            return causal_mask

        causal_mask = causal_mask.clone()
        num_maskable = prefix_length - keep_last_k
        rand_mask = torch.rand(num_maskable, device=causal_mask.device) < self.prefix_mask_prob
        causal_mask[:, suffix_start:, action_start:masked_prefix_end] *= (~rand_mask).int()
        return causal_mask

    # ------------------------------------------------------------------
    # Log-probability helpers (for RL training)
    # ------------------------------------------------------------------

    @staticmethod
    def get_logprob_norm(
        sample: torch.Tensor,
        mu: torch.Tensor,
        sigma: torch.Tensor,
    ) -> torch.Tensor:
        """Log-probability of *sample* under N(mu, sigma).

        When sigma == 0 (deterministic step), returns 0 for that element.

        Returns:
            Tensor of same shape as *sample*.
        """
        sample = sample.float()
        mu = mu.float()
        sigma = sigma.float()

        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)

        # Full Gaussian log-density: -0.5*log(2π) - log(σ) - 0.5*((x-μ)/σ)²
        log_prob = (
            -0.5 * math.log(2 * math.pi)
            - torch.log(sigma_safe)
            - 0.5 * ((sample - mu) / sigma_safe) ** 2
        )
        log_prob = torch.where(mask, torch.zeros_like(log_prob), log_prob)
        return log_prob

    @staticmethod
    def gaussian_entropy(sigma: torch.Tensor) -> torch.Tensor:
        """Entropy of N(0, sigma).  Returns 0 where sigma == 0."""
        sigma = sigma.float()
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        entropy = 0.5 * torch.log(2 * math.pi * math.e * sigma_safe**2)
        entropy = torch.where(mask, torch.zeros_like(entropy), entropy)
        return entropy

    def compute_frequency_loss(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Frequency-domain loss: penalizes spectral differences between pred and target.

        Encourages smooth action sequences by penalizing high-frequency noise.

        Args:
            pred: Predicted action ``(B, L, D)``.
            target: Target action ``(B, L, D)``.
            weight: Optional per-element weight ``(B, L, D)``.

        Returns:
            Scalar frequency loss.
        """
        pred = pred.float()
        target = target.float()
        loss_freq = (torch.fft.rfft(pred, dim=1) - torch.fft.rfft(target, dim=1)).abs()
        if weight is not None:
            weight_dct = weight.float().mean(dim=[1, 2])
            loss_freq = (loss_freq * weight_dct.unsqueeze(1).unsqueeze(2))
        return loss_freq.mean()

    # ------------------------------------------------------------------
    # Flow-SDE denoising helpers
    # ------------------------------------------------------------------

    def _compute_denoise_mean_std(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        timesteps: torch.Tensor,
        idx: int,
        mode: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute the mean and std of the denoising distribution.

        Does NOT sample or compute log-prob — use this for replaying the
        mean in ``default_forward`` so that ``get_logprob_norm`` is evaluated
        at the correct mean, not at a fresh random sample.

        Returns:
            Tuple of ``(x_t_mean, x_t_std)``.
        """
        orig_dtype = x_t.dtype

        t_val = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        t_input = t_val.view(1, 1, 1).expand_as(x_t).to(dtype=orig_dtype)
        delta_input = delta.view(1, 1, 1).expand_as(x_t).to(dtype=orig_dtype)

        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            # Euler step: x_{t+1} = x_t + v * delta (matches checkpoint forward)
            x_t_mean = x_t + v_t * delta_input
            x_t_std = torch.zeros_like(x_t)
            return x_t_mean, x_t_std
        elif mode == "train":
            if self.noise_method == "flow_sde":
                t_safe = torch.where(
                    timesteps == 1.0, timesteps[1], timesteps
                )
                sigmas = self.noise_level * torch.sqrt(
                    timesteps / (1 - t_safe)
                )
                sigmas = sigmas[:-1]
                sigma_i = sigmas[idx].view(1, 1, 1).expand_as(x_t).to(dtype=orig_dtype)

                # Euler step + Flow-SDE drift correction
                x_t_mean = x_t + v_t * delta_input - (
                    sigma_i**2 * delta_input / (2 * t_input)
                ) * v_t
                x_t_std = torch.sqrt(delta_input) * sigma_i
            else:
                sigma = self.noise_level * math.sqrt(delta)
                x_t_mean = x_t + v_t * delta_input
                x_t_std = torch.full_like(x_t, sigma)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        return x_t_mean, x_t_std

    def _compute_denoise_step(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        timesteps: torch.Tensor,
        idx: int,
        mode: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one denoising step with Flow-SDE noise.

        Wraps ``_compute_denoise_mean_std`` and additionally samples
        ``x_t_next`` and computes its log-probability.

        Returns:
            Tuple of ``(x_t_next, x_t_std, log_prob)``.
        """
        x_t_mean, x_t_std = self._compute_denoise_mean_std(
            x_t, v_t, timesteps, idx, mode=mode
        )

        if mode == "train" and self.noise_method == "flow_sde":
            noise = torch.randn_like(x_t)
            x_t_next = x_t_mean + noise * x_t_std
            log_prob = self.get_logprob_norm(x_t_next, x_t_mean, x_t_std)
        else:
            x_t_next = x_t_mean
            log_prob = torch.zeros_like(x_t)

        return x_t_next, x_t_std, log_prob

    # ------------------------------------------------------------------
    # Step-by-step denoising (for RL chain recording)
    # ------------------------------------------------------------------

    def sample_actions(
        self,
        vlm_batch: dict[str, torch.Tensor],
        state_tensor: torch.Tensor,
        device: torch.device,
        mode: Literal["train", "eval"] = "train",
        robot_type: Optional[str] = None,
        seed: Optional[int] = None,
    ) -> dict[str, Any]:
        """Run rectified flow denoising step-by-step and record the chain.

        During ``mode="train"``, one random denoising step is made stochastic
        (nonzero std) so its log-prob can be used for the policy gradient.
        All other steps are deterministic (std=0, logprob=0).

        Args:
            vlm_batch: VLM processor output (input_ids, pixel_values, ...).
            state_tensor: ``(B, 1, STATE_DIM)`` proprioceptive state.
            device: Target device.
            mode: ``"train"`` or ``"eval"``.

        Returns:
            Dict with ``actions``, ``chains``, ``prev_logprobs``,
            ``denoise_inds``, ``prev_values``.
        """
        batch_size = state_tensor.shape[0]

        # Determine if stub or real model
        is_stub = hasattr(self.xr0_model, "generate")

        if is_stub:
            return self._sample_actions_stub(
                batch_size, device, mode
            )

        # --- Real model: step-by-step denoising ---
        # Use eval mode to disable dropout/BN randomness.  Policy stochasticity
        # comes from Flow-SDE noise, not from dropout.
        self.xr0_model.eval()

        # VLM forward to get KV-cache and hidden states.
        # Temporarily disable gradient checkpointing on ALL VLM sub-modules
        # so the @check_model_inputs decorator does not force use_cache=False.
        vlm_keys = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        vlm_inputs = {
            k: v.to(device)
            for k, v in vlm_batch.items()
            if k in vlm_keys and isinstance(v, torch.Tensor)
        }

        vlm_module = self.xr0_model.vlm
        gc_states: list[tuple[torch.nn.Module, bool]] = []
        for module in vlm_module.modules():
            if getattr(module, "gradient_checkpointing", False):
                gc_states.append((module, True))
                module.gradient_checkpointing = False
        try:
            vlm_outputs = vlm_module(
                **vlm_inputs, use_cache=True, output_hidden_states=True
            )
        finally:
            for module, state in gc_states:
                module.gradient_checkpointing = state
        past_key_values = list(vlm_outputs.past_key_values)

        # Pack KV cache into flat tensors for forward_inputs transport.
        packed_kv = self._pack_vlm_kv(past_key_values)
        vlm_pos_max = vlm_outputs.position_ids.max(dim=-1)[0]
        packed_pos_max = self._pack_vlm_pos_max(vlm_pos_max, batch_size)

        # Capture VLM hidden states for value computation.
        # Qwen3VLCausalLMOutputWithPast stores hidden_states as a tuple
        # (one per layer when output_hidden_states=True). The last element
        # is the final layer's hidden states with shape (B, seq_len, hidden_dim).
        if hasattr(vlm_outputs, "hidden_states") and vlm_outputs.hidden_states is not None:
            if isinstance(vlm_outputs.hidden_states, tuple):
                vlm_hidden_states = vlm_outputs.hidden_states[-1]
            else:
                vlm_hidden_states = vlm_outputs.hidden_states
        else:
            # Fallback: run a separate VLM forward with output_hidden_states=True
            vlm_outputs2 = self.xr0_model.vlm(**vlm_inputs, use_cache=False, output_hidden_states=True)
            if hasattr(vlm_outputs2, "hidden_states") and vlm_outputs2.hidden_states is not None:
                vlm_hidden_states = vlm_outputs2.hidden_states[-1] if isinstance(vlm_outputs2.hidden_states, tuple) else vlm_outputs2.hidden_states
            else:
                vlm_hidden_states = None

        if vlm_hidden_states is not None:
            vlm_attention_mask = vlm_inputs.get(
                "attention_mask",
                torch.ones(batch_size, vlm_hidden_states.shape[1], device=device, dtype=torch.long),
            )
        else:
            vlm_attention_mask = vlm_inputs.get(
                "attention_mask",
                torch.ones(batch_size, 1, device=device, dtype=torch.long),
            )

        # --- Match checkpoint forward: position_embeds + attn_mask ---
        # Action mask from processor (determines action_len).
        action_mask_base = self._get_action_mask(device, robot_type=robot_type)
        action_len = action_mask_base.shape[1]
        action_mask = action_mask_base.expand(batch_size, -1, -1)

        # Pad state to model's expected dim.
        state_tensor_padded = self._pad_state(state_tensor)
        _, state_len, _ = state_tensor_padded.shape
        dit_query_length = action_len + state_len + 1

        # Position ids (same as checkpoint forward line 1827-1831)
        position_ids = (
            torch.arange(0, dit_query_length, device=device).view(1, 1, -1).repeat(3, batch_size, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )
        # Position embeds (same as checkpoint forward line 1832)
        position_embeds = self.xr0_model.rotary_emb(action_mask, position_ids)

        # Attention mask with local causal window for action tokens
        state_len = state_tensor.shape[1] if state_tensor is not None else 1
        s_len = state_len + 1  # +1 for sink token
        a_len = action_len
        mask_ss = torch.tril(torch.ones(s_len, s_len, device=device))
        mask_sa = torch.zeros(s_len, a_len, device=device)
        mask_as = torch.ones(a_len, s_len, device=device)
        mask_aa = torch.tril(torch.ones(a_len, a_len, device=device))
        mask_aa = mask_aa * torch.triu(
            torch.ones(a_len, a_len, device=device), diagonal=-self.local_window
        )
        causal_mask = torch.cat(
            [torch.cat([mask_ss, mask_sa], dim=1),
             torch.cat([mask_as, mask_aa], dim=1)], dim=0,
        )
        cache_mask = vlm_outputs.attention_mask[:, None, :].expand(-1, dit_query_length, -1)
        attn_mask = torch.cat(
            [cache_mask, causal_mask[None].expand(batch_size, -1, -1)], dim=-1
        )[:, None].bool()

        # Async train: randomly set prefix_length for partial replay
        prefix_length = 0
        if mode == "train" and self.async_train and random.random() < 0.5:
            prefix_length = random.randint(1, min(6, action_len))

        # Apply random mask to prefix tokens if prefix_length > 2
        if mode == "train" and prefix_length > 2:
            causal_mask_for_prefix = self._random_mask_prefix(
                causal_mask[None].expand(batch_size, -1, -1),
                prefix_length, state_len,
            )
            attn_mask = torch.cat(
                [cache_mask, causal_mask_for_prefix], dim=-1
            )[:, None].bool()

        # Offset position IDs for non-prefix tokens (original XR0.py line 749-750)
        if prefix_length > 0 and action_len > prefix_length:
            position_ids[:, :, -(action_len - prefix_length):] += 10

        # State embedding
        state_embed = self.xr0_model.state_projector(
            state_tensor_padded.to(device=device, dtype=torch.bfloat16)
        )

        # Timestep schedule: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0] for 5 steps
        timesteps = torch.linspace(
            1.0, 0.0, self.num_steps + 1, device=device
        )

        # For Flow-SDE: pick one random step for policy gradient
        # (all steps add noise, but only one step's logprob is used)
        # Override with XR0_DENOISE_IND env var for debugging.
        # Default: skip low-sigma steps (high index) to avoid logprob
        # amplification of residual bf16 replay errors.
        _fixed_di = os.environ.get("XR0_DENOISE_IND")
        if mode == "train":
            if _fixed_di is not None:
                denoise_ind = int(_fixed_di)
            else:
                # Skip the last step (lowest sigma) to reduce instability.
                # With num_steps=5: choose from [0,1,2,3] instead of [0,1,2,3,4].
                denoise_ind = random.randint(0, max(self.num_steps - 2, 0))
        else:
            denoise_ind = -1  # all deterministic

        # Denoising loop — match Xiaomi server: seed controls initial noise.
        # server.py line 1857: torch.manual_seed(kwargs["seed"])
        if seed is not None:
            cpu_rng_state = torch.get_rng_state()
            gpu_rng_state = torch.cuda.get_rng_state(device) if device.type == "cuda" else None
            torch.manual_seed(seed)
        x_t = torch.randn(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )
        if seed is not None:
            torch.set_rng_state(cpu_rng_state)
            if gpu_rng_state is not None:
                torch.cuda.set_rng_state(gpu_rng_state, device)
        chains = [x_t.detach().clone()]
        log_probs = []
        # Debug: store v_t / mean / std at the chosen denoise step.
        debug_v_t = None
        debug_x_t_mean = None
        debug_x_t_std = None

        for idx in range(self.num_steps):
            t_val = timesteps[idx]
            if idx == 0:
                self.logger.info("[DENOISE_DBG] step0 x_t[0,0,:5]=%s attn_mask=%s pos_embeds=%s",
                    x_t[0,0,:5].float().tolist(),
                    tuple(attn_mask.shape),
                    tuple(position_embeds[0].shape))

            # DiT forward: predict velocity
            t_tensor = t_val.view(1, 1, 1).expand(
                batch_size, 1, 1
            ).to(dtype=torch.bfloat16)
            v_t = self.xr0_model.dit_forward(
                x_t, t_tensor, action_mask, state_embed,
                position_embeds, past_key_values, attn_mask,
            )
            if idx == 0:
                self.logger.info("[DENOISE_DBG] step0 v_t[0,0,:7]=%s max_abs=%.4f",
                    v_t[0,0,:7].float().tolist(), v_t.float().abs().max().item())

            # Compute denoising step with Flow-SDE
            step_mode = mode if idx == denoise_ind or mode == "eval" else "eval"
            x_t_next, x_t_std_step, log_prob = self._compute_denoise_step(
                x_t, v_t, timesteps, idx, mode=step_mode
            )

            # Save debug values at the chosen denoise step.
            if idx == denoise_ind and denoise_ind >= 0:
                debug_v_t = v_t.detach().float().cpu()
                mean_dbg, std_dbg = self._compute_denoise_mean_std(
                    chains[-1], v_t, timesteps, idx, mode="train"
                )
                debug_x_t_mean = mean_dbg.detach().float().cpu()
                debug_x_t_std = std_dbg.detach().float().cpu()

            x_t = x_t_next
            chains.append(x_t.detach().clone())
            log_probs.append(log_prob)

        # Aggregate: (B, num_steps+1, action_len, action_dim)
        chains_tensor = torch.stack(chains, dim=1)
        log_probs_tensor = torch.stack(log_probs, dim=1)

        # Pick logprob at the chosen denoise step.
        # Return shape (B, num_action_chunks, action_dim) so that
        # preprocess_loss_inputs can reshape/aggregate as needed.
        if denoise_ind >= 0:
            prev_logprobs = log_probs_tensor[:, denoise_ind]  # (B, C, D)
        else:
            prev_logprobs = torch.zeros(
                batch_size, action_len, self.action_dim, device=device
            )

        # Apply valid_action_mask to prev_logprobs so that stored old_logprobs
        # are consistent with the masked logprobs in default_forward.
        # This ensures the PPO ratio exp(logprobs - old_logprobs) only sees
        # valid action dimensions.
        if self.action_mapper is not None:
            prev_logprobs = self.action_mapper.apply_mask(prev_logprobs)

        # Compute value from VLM hidden states
        if self.add_value_head and vlm_hidden_states is not None:
            prev_values = self.get_value_from_vlm(
                vlm_hidden_states.detach(), vlm_attention_mask.detach()
            )
        else:
            prev_values = torch.zeros(batch_size, device=device)

        # Restore train mode (was set to eval above for dropout-free forward).
        self.xr0_model.train()

        result = {
            "actions": x_t[:, :action_len, : self.action_dim],
            "chains": chains_tensor,
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "denoise_inds": torch.full((batch_size,), denoise_ind, dtype=torch.long),
        }
        if vlm_hidden_states is not None:
            result["vlm_hidden_states"] = vlm_hidden_states.detach()
            result["vlm_attention_mask"] = vlm_attention_mask.detach()
        result["vlm_pos_max"] = packed_pos_max
        result.update(packed_kv)
        # Debug tensors for replay comparison.
        if debug_v_t is not None:
            result["debug_v_t"] = debug_v_t
            result["debug_x_t_mean"] = debug_x_t_mean
            result["debug_x_t_std"] = debug_x_t_std
        return result

    def _sample_actions_stub(
        self,
        batch_size: int,
        device: torch.device,
        mode: str,
    ) -> dict[str, Any]:
        """Stub version: generate random actions with dummy chain."""
        action_len = self.num_action_chunks

        # Random final action
        actions = torch.randn(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )

        # Build a dummy chain (num_steps+1 states)
        chains = []
        for _ in range(self.num_steps + 1):
            chains.append(
                torch.randn(
                    (batch_size, action_len, self.action_dim),
                    device=device, dtype=torch.bfloat16,
                )
            )
        chains_tensor = torch.stack(chains, dim=1)

        denoise_ind = random.randint(0, self.num_steps - 1) if mode == "train" else -1

        return {
            "actions": actions,
            "chains": chains_tensor,
            # Stub: return zeros with correct shapes matching the real model.
            # prev_logprobs: (B, C, D) so preprocess_loss_inputs can reshape.
            # prev_values: (B,) scalar per sample.
            "prev_logprobs": torch.zeros(
                batch_size, action_len, self.action_dim, device=device
            ),
            "prev_values": torch.zeros(batch_size, device=device),
            "denoise_inds": torch.full(
                (batch_size,), denoise_ind, dtype=torch.long
            ),
        }

    # ------------------------------------------------------------------
    # Rollout-time: predict_action_batch
    # ------------------------------------------------------------------

    def _checkpoint_forward_eval(self, state, action_mask, vlm_inputs, seed):
        """Replicate checkpoint forward for eval mode.

        Same logic as modeling_mibot.py:forward but without modifying the
        checkpoint file.  Handles seed internally so it's not forwarded
        to the VLM.
        """
        model = self.xr0_model

        # VLM forward
        vlm_outputs = model.vlm(**vlm_inputs, use_cache=True)
        past_key_values = list(vlm_outputs.past_key_values)

        action_bs, action_length, _ = action_mask.shape
        _, state_length, _ = state.shape
        dit_query_length = action_length + state_length + 1

        # Position embeds (matches checkpoint line 1827-1832)
        position_ids = (
            torch.arange(0, dit_query_length, device=action_mask.device)
            .view(1, 1, -1).repeat(3, action_bs, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )
        position_embeds = model.rotary_emb(action_mask, position_ids)

        # Attention mask with local causal window for action tokens
        s_len = state_length + 1  # +1 for sink token
        a_len = action_length
        mask_ss = torch.tril(torch.ones(s_len, s_len, device=action_mask.device))
        mask_sa = torch.zeros(s_len, a_len, device=action_mask.device)
        mask_as = torch.ones(a_len, s_len, device=action_mask.device)
        mask_aa = torch.tril(torch.ones(a_len, a_len, device=action_mask.device))
        mask_aa = mask_aa * torch.triu(
            torch.ones(a_len, a_len, device=action_mask.device), diagonal=-self.local_window
        )
        causal_mask = torch.cat(
            [torch.cat([mask_ss, mask_sa], dim=1),
             torch.cat([mask_as, mask_aa], dim=1)], dim=0,
        )
        cache_mask = vlm_outputs.attention_mask[:, None, :].expand(-1, dit_query_length, -1)
        attn_mask = torch.cat(
            [cache_mask, causal_mask[None].expand(action_bs, -1, -1)], dim=-1
        )[:, None].bool()

        # State embedding
        state_embed = model.state_projector(state)

        # Denoising loop with seed
        cpu_rng = torch.get_rng_state()
        gpu_rng = torch.cuda.get_rng_state(action_mask.device) if action_mask.is_cuda else None
        torch.manual_seed(seed)
        x = torch.randn_like(action_mask)
        torch.set_rng_state(cpu_rng)
        if gpu_rng is not None:
            torch.cuda.set_rng_state(gpu_rng, action_mask.device)

        dt = 1.0 / self.num_steps
        for step in range(self.num_steps):
            t = torch.ones((x.shape[0], 1, 1), device=x.device, dtype=x.dtype) * step / self.num_steps
            v = model.dit_forward(x, t, action_mask, state_embed, position_embeds, past_key_values, attn_mask)
            x = x + v * dt

        # Match ActionGenerationOutput interface
        class _Out:
            pass
        out = _Out()
        out.actions = x
        return out

    @torch.no_grad()
    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = True,
        **kwargs: Any,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Predict a batch of actions from environment observations.

        Returns:
            Tuple of ``(actions, result)`` where *actions* is ``[B, C, D]``
            and *result* has ``prev_logprobs``, ``prev_values``,
            ``forward_inputs``.
        """
        device = next(self.parameters()).device

        images = env_obs["main_images"]
        wrist_images = env_obs.get("wrist_images")  # optional
        states = env_obs["states"]
        task_descriptions = env_obs.get("task_descriptions") or [""] * len(images)
        batch_size = len(images)
        # Match Xiaomi server: robot_type comes from input_data["task_id"].
        robot_type = env_obs.get("task_id") or env_obs.get("robot_type") or self.robot_type

        # State tensor: (B, 1, STATE_DIM)
        state_tensor = torch.from_numpy(np.asarray(states, dtype=np.float32))
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(1)

        # Build VLM batch (with wrist images if available, matching Xiaomi server)
        vlm_batch = self._build_vlm_batch(
            images, task_descriptions, state_tensor, device,
            wrist_images=wrist_images,
        )

        # Denormalize actions using official processor.decode_action
        proc = self.processor
        action_mask = self._get_action_mask(device, robot_type=robot_type)

        if mode == "eval":
            # Eval: replicate checkpoint forward logic without modifying the
            # checkpoint file.  Calls VLM, builds position_embeds/attn_mask
            # exactly like modeling_mibot.py:forward, then runs the denoising
            # loop with Euler steps.
            st = self._pad_state(state_tensor).to(device=device, dtype=torch.bfloat16)
            rollout_seed = int(torch.randint(0, 2**31, (1,)).item())
            vlm_inputs = {k: v.to(device) for k, v in vlm_batch.items() if isinstance(v, torch.Tensor)}
            out = self._checkpoint_forward_eval(st, action_mask, vlm_inputs, rollout_seed)
            raw_actions_cpu = out.actions.float().cpu()
            actions_np = proc.decode_action(raw_actions_cpu, robot_type=robot_type).numpy()
            env_actions_np = actions_np[:, :self.num_action_chunks, : self.action_env_dim]
            result = {
                "prev_logprobs": torch.zeros(batch_size, self.num_action_chunks, self.action_dim),
                "prev_values": torch.zeros(batch_size),
                "forward_inputs": {},
            }
            return env_actions_np, result

        # Train: step-by-step denoising with chain recording.
        rollout_seed = int(torch.randint(0, 2**31, (1,)).item())
        outputs = self.sample_actions(
            vlm_batch, state_tensor, device, mode=mode,
            robot_type=robot_type, seed=rollout_seed,
        )

        raw_actions_cpu = outputs["actions"].float().cpu()
        decoded = proc.decode_action(raw_actions_cpu, robot_type=robot_type)
        actions_np = decoded.numpy()[:, :self.num_action_chunks, :]

        # ---- Step 2 diagnostic: print raw and decoded action chunk ----
        chunk_raw = raw_actions_cpu[0, :self.num_action_chunks, :7].numpy()
        chunk_7d = actions_np[0, :self.num_action_chunks, :7]
        self.logger.info(
            "[XR0_RAW] max_abs=%.4f first=%s",
            np.max(np.abs(chunk_raw)),
            np.array2string(chunk_raw[0], precision=4, suppress_small=True),
        )
        self.logger.info(
            "[XR0_DECODED] max_abs=%.4f first=%s",
            np.max(np.abs(chunk_7d)),
            np.array2string(chunk_7d[0], precision=4, suppress_small=True),
        )
        if chunk_7d.shape[0] > 1:
            step_diff = np.abs(chunk_7d[1:] - chunk_7d[:-1])
            self.logger.info(
                "[XR0_STEP_DIFF] max=%s",
                np.array2string(np.max(step_diff, axis=0), precision=4, suppress_small=True),
            )

        # Build forward_inputs for training replay
        # NOTE: robot_type is NOT stored here because forward_inputs must
        # only contain tensors (cat_list_of_dict_tensor will crash on strings).
        # robot_type is retrieved from env_obs or self.robot_type in default_forward.
        forward_inputs: dict[str, Any] = {
            "chains": outputs["chains"].cpu(),
            "denoise_inds": outputs["denoise_inds"].cpu(),
            "prefix_length": torch.tensor([prefix_length]),
        }
        for k, v in vlm_batch.items():
            if isinstance(v, torch.Tensor):
                forward_inputs[k] = v.detach().cpu()
        forward_inputs["state"] = state_tensor.detach().cpu()
        # Store full-dim action for training replay (chains are in full dim)
        forward_inputs["action"] = torch.from_numpy(
            actions_np.reshape(batch_size, -1).astype(np.float32)
        )
        # Store VLM hidden states for value computation in default_forward
        if "vlm_hidden_states" in outputs:
            forward_inputs["vlm_hidden_states"] = outputs["vlm_hidden_states"].cpu()
            forward_inputs["vlm_attention_mask"] = outputs["vlm_attention_mask"].cpu()
        # Store rollout-time VLM KV cache for exact training replay.
        for k, v in outputs.items():
            if k.startswith("vlm_kv_") or k == "vlm_pos_max":
                if isinstance(v, torch.Tensor):
                    forward_inputs[k] = v.cpu()
                else:
                    forward_inputs[k] = v
        # Store debug tensors for replay comparison.
        for k in ("debug_v_t", "debug_x_t_mean", "debug_x_t_std"):
            if k in outputs:
                forward_inputs[k] = outputs[k]

        # Slice/gather actions to environment's expected dimension.
        # XR0 outputs 32D (bimanual) but e.g. LIBERO expects 7D (right arm),
        # SO101 expects 12D (5 joints + gripper per arm).
        # The full 32D is preserved in forward_inputs for training replay.
        if self.action_mapper is not None:
            actions_tensor = torch.from_numpy(actions_np)
            env_actions_tensor = self.action_mapper.map_to_env(actions_tensor)
            env_actions_np = env_actions_tensor.numpy()
        else:
            env_actions_np = actions_np[:, :, : self.action_env_dim]

        # prev_values shape: (B, 1) — single value per chunk for PPO/GAE.
        # The value head produces (B,), expand to (B, 1).
        prev_values = outputs["prev_values"].cpu()
        if prev_values.ndim == 1:
            prev_values = prev_values.unsqueeze(1)

        result = {
            "prev_logprobs": outputs["prev_logprobs"].cpu(),
            "prev_values": prev_values,
            "forward_inputs": forward_inputs,
        }
        return env_actions_np, result

    # ------------------------------------------------------------------
    # Training-time: forward / default_forward
    # ------------------------------------------------------------------

    def forward(self, **kwargs: Any) -> dict[str, torch.Tensor]:
        """Entry point called by the FSDP actor worker.

        Delegates to ``default_forward`` so the actor can call
        ``self.model(forward_inputs=..., ...)`` directly.
        """
        return self.default_forward(**kwargs)

    def default_forward(
        self,
        forward_inputs: Optional[dict[str, torch.Tensor]] = None,
        compute_logprobs: bool = True,
        compute_entropy: bool = True,
        compute_values: bool = False,
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        """Replay the denoising chain and recompute logprobs/entropy.

        During RL training, the actor worker calls this with the stored
        ``forward_inputs`` from rollout.  It replays the single stochastic
        denoising step under the *current* policy weights to get the
        ``logprobs`` needed for the PPO/GRPO policy ratio.
        """
        if forward_inputs is None:
            forward_inputs = {}

        device = next(self.parameters()).device
        # robot_type is no longer stored in forward_inputs (it's a string,
        # not a tensor, and would crash cat_list_of_dict_tensor).
        robot_type = self.robot_type

        chains = forward_inputs.get("chains")  # (B, num_steps+1, C, D)
        denoise_inds = forward_inputs.get("denoise_inds")  # (B,)
        prefix_length = int(forward_inputs.get("prefix_length", torch.tensor([0])).item())

        def _compute_values(batch_size: int) -> torch.Tensor:
            """Compute values from VLM hidden states if available."""
            if self.add_value_head and "vlm_hidden_states" in forward_inputs:
                vlm_hidden = forward_inputs["vlm_hidden_states"].to(device)
                vlm_mask = forward_inputs["vlm_attention_mask"].to(device)
                return self.get_value_from_vlm(vlm_hidden, vlm_mask)
            return torch.zeros(batch_size, device=device)

        if chains is None or denoise_inds is None:
            # This fallback should not happen in normal usage.
            # Indicates forward_inputs was not properly built.
            batch_size = 1
            for v in forward_inputs.values():
                if isinstance(v, torch.Tensor) and v.ndim > 0:
                    batch_size = v.shape[0]
                    break
            return {
                "logprobs": torch.zeros(batch_size, self.num_action_chunks, self.action_dim, device=device),
                "values": _compute_values(batch_size).float(),
                "entropy": torch.zeros(batch_size, self.num_action_chunks, self.action_dim, device=device),
            }

        batch_size = chains.shape[0]
        denoise_ind = int(denoise_inds[0].item())

        if denoise_ind < 0:
            # Eval mode: no stochastic step, return zeros for logprobs/entropy
            return {
                "logprobs": torch.zeros(batch_size, self.num_action_chunks, self.action_dim, device=device),
                "values": _compute_values(batch_size).float(),
                "entropy": torch.zeros(batch_size, self.num_action_chunks, self.action_dim, device=device),
            }

        is_stub = hasattr(self.xr0_model, "generate")

        if is_stub:
            # For stub model, return dummy logprobs/entropy with correct shape.
            # Compute values from VLM hidden states if available.
            if self.add_value_head and "vlm_hidden_states" in forward_inputs:
                vlm_hidden = forward_inputs["vlm_hidden_states"].to(device)
                vlm_mask = forward_inputs["vlm_attention_mask"].to(device)
                values = self.get_value_from_vlm(vlm_hidden, vlm_mask)
            else:
                values = torch.zeros(batch_size, device=device)
            return {
                "logprobs": torch.zeros(batch_size, self.num_action_chunks, self.action_dim, device=device),
                "values": values.float(),
                "entropy": torch.zeros(batch_size, self.num_action_chunks, self.action_dim, device=device),
            }

        # --- Real model: replay the chain ---
        chains = chains.to(device)

        state_tensor = forward_inputs.get("state")
        if state_tensor is not None:
            state_tensor = state_tensor.to(device)

        # Unpack VLM KV cache from forward_inputs (stored by sample_actions).
        # This gives the exact same KV cache as during rollout, avoiding
        # batch-size-dependent numerical differences in the VLM attention.
        past_key_values = self._unpack_vlm_kv(forward_inputs, device)

        if past_key_values is not None and "vlm_pos_max" in forward_inputs:
            vlm_pos_max = self._unpack_vlm_pos_max(
                forward_inputs["vlm_pos_max"], batch_size, device,
            )
        else:
            # Fallback: re-run VLM forward (e.g. for eval).
            self.logger.warning(
                "[default_forward] No stored KV cache — falling back to VLM re-forward. "
                "This may produce first-step KL mismatch if Qwen3-VL is batch-size sensitive."
            )
            vlm_keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
            vlm_module = self.xr0_model.vlm
            gc_states: list[tuple[torch.nn.Module, bool]] = []
            for module in vlm_module.modules():
                if getattr(module, "gradient_checkpointing", False):
                    gc_states.append((module, True))
                    module.gradient_checkpointing = False
            try:
                vlm_batch = {}
                for k in vlm_keys:
                    if k in forward_inputs:
                        vlm_batch[k] = forward_inputs[k].to(device)
                with torch.no_grad():
                    vlm_outputs = vlm_module(**vlm_batch, use_cache=True)
                past_key_values = list(vlm_outputs.past_key_values)
                vlm_pos_max = vlm_outputs.position_ids.max(dim=-1)[0]
            finally:
                for module, state in gc_states:
                    module.gradient_checkpointing = state

        # Reconstruct attention mask from stored forward_inputs.
        fi_attn_mask = forward_inputs.get("attention_mask")
        if fi_attn_mask is not None:
            cache_attn_mask = fi_attn_mask.to(device)
        else:
            cache_attn_mask = torch.ones(batch_size, 1, device=device, dtype=torch.long)

        # Build position ids and attention mask (same as sample_actions)
        action_len = self.num_action_chunks
        state_len = state_tensor.shape[1] if state_tensor is not None else 1
        q_len = action_len + state_len + 1
        position_ids = (
            torch.arange(0, q_len, device=device)
            .view(1, 1, -1)
            .repeat(3, batch_size, 1)
            + vlm_pos_max[..., None]
            + 1
        )

        cache_mask = cache_attn_mask[:, None, :].expand(-1, q_len, -1)

        s_len = state_len + 1
        a_len = action_len
        mask_ss = torch.tril(torch.ones(s_len, s_len, device=device))
        mask_sa = torch.zeros(s_len, a_len, device=device)
        mask_as = torch.ones(a_len, s_len, device=device)
        # P2_Local-style local causal mask: each action token attends to
        # at most local_window previous tokens (matches original XR0.py).
        mask_aa = torch.tril(torch.ones(a_len, a_len, device=device))
        mask_aa = mask_aa * torch.triu(
            torch.ones(a_len, a_len, device=device), diagonal=-self.local_window
        )
        causal_mask = torch.cat(
            [torch.cat([mask_ss, mask_sa], dim=1),
             torch.cat([mask_as, mask_aa], dim=1)],
            dim=0,
        )
        attn_mask = torch.cat(
            [cache_mask, causal_mask[None].expand(batch_size, -1, -1)], dim=-1
        )[:, None].bool()

        # Apply prefix masking (must match sample_actions for consistency)
        if prefix_length > 2:
            causal_mask_prefixed = self._random_mask_prefix(
                causal_mask[None].expand(batch_size, -1, -1),
                prefix_length, state_len,
            )
            attn_mask = torch.cat(
                [cache_mask, causal_mask_prefixed], dim=-1
            )[:, None].bool()

        # Offset position IDs for non-prefix tokens (must match sample_actions)
        if prefix_length > 0 and action_len > prefix_length:
            position_ids[:, :, -(action_len - prefix_length):] += 10

        # State embedding (pad/truncate to match model's expected dim)
        state_tensor_padded = self._pad_state(state_tensor)
        state_embed = self.xr0_model.state_projector(
            state_tensor_padded.to(dtype=torch.bfloat16)
        )

        dummy_action = torch.zeros(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )
        position_embeds = self.xr0_model.rotary_emb(dummy_action, position_ids)

        # Action mask from processor (same as sample_actions)
        action_mask_base = self._get_action_mask(device, robot_type=robot_type)
        action_len = action_mask_base.shape[1]
        action_mask = action_mask_base.expand(batch_size, -1, -1)

        timesteps = torch.linspace(
            1.0, 0.0, self.num_steps + 1, device=device
        )

        # Replay: get x_t and x_{t+1} from the recorded chain
        x_t = chains[:, denoise_ind]  # (B, C, D)
        x_next = chains[:, denoise_ind + 1]  # recorded next state

        t_val = timesteps[denoise_ind]

        # Debug assertions: ensure batch dimensions are consistent.
        assert past_key_values[0][0].shape[0] == batch_size, (
            f"KV batch mismatch: kv B={past_key_values[0][0].shape[0]}, "
            f"current B={batch_size}"
        )
        assert cache_attn_mask.shape[0] == batch_size, (
            f"attention_mask batch mismatch: mask B={cache_attn_mask.shape[0]}, "
            f"current B={batch_size}"
        )
        assert chains.shape[0] == batch_size, (
            f"chains batch mismatch: chains B={chains.shape[0]}, current B={batch_size}"
        )

        # Current policy's velocity prediction (eval mode to disable dropout).
        t_tensor = t_val.view(1, 1, 1).expand(
            batch_size, 1, 1
        ).to(dtype=torch.bfloat16)
        with _temporarily_eval(self.xr0_model):
            v_t = self.xr0_model.dit_forward(
                x_t, t_tensor, action_mask, state_embed,
                position_embeds, past_key_values, attn_mask,
            )

        # Zero out prefix positions (original XR0.py line 618-619)
        if prefix_length > 0:
            v_t[:, :prefix_length] = 0.0

        # Compute mean and std using Flow-SDE (no sampling — we need the
        # exact mean to evaluate logprob of the recorded x_next).
        x_t_mean, x_t_std = self._compute_denoise_mean_std(
            x_t, v_t, timesteps, denoise_ind, mode="train"
        )

        # Debug: compare rollout vs training v_t / mean / std.
        _dbg_v = forward_inputs.get("debug_v_t")
        _dbg_mean = forward_inputs.get("debug_x_t_mean")
        _dbg_std = forward_inputs.get("debug_x_t_std")
        if _dbg_v is not None and _dbg_mean is not None and _dbg_std is not None:
            _dbg_v = _dbg_v.to(device=device, dtype=v_t.dtype)
            _dbg_mean = _dbg_mean.to(device=device, dtype=x_t_mean.dtype)
            _dbg_std = _dbg_std.to(device=device, dtype=x_t_std.dtype)
            v_diff = (v_t.float() - _dbg_v.float()).abs()
            mean_diff = (x_t_mean.float() - _dbg_mean.float()).abs()
            std_diff = (x_t_std.float() - _dbg_std.float()).abs()
            normed_mean = mean_diff / _dbg_std.float().clamp_min(1e-6)
            self.logger.info(
                "[replay debug] v_diff: max=%.6f | mean_diff: max=%.6f | "
                "std_diff: max=%.6f | mean_diff/sigma: mean=%.6f max=%.6f",
                v_diff.max().item(), mean_diff.max().item(),
                std_diff.max().item(), normed_mean.mean().item(), normed_mean.max().item(),
            )

        # Log-prob of recorded next state under current policy.
        # Return shape (B, num_action_chunks, action_dim) so that
        # preprocess_loss_inputs can reshape/aggregate as needed.
        logprobs = self.get_logprob_norm(x_next, x_t_mean, x_t_std)
        # logprobs: (B, C, D) — keep all dims for the loss function.

        # Entropy: same shape as logprobs.
        entropy = self.gaussian_entropy(x_t_std)
        # entropy: (B, C, D) — keep all dims for the loss function.

        # Apply valid_action_mask: zero out invalid action dimensions so that
        # logprob aggregation (sum over action_dim) and entropy computation
        # only use the dimensions that the environment actually consumes.
        # This prevents e.g. XR0's 20 unused dims from polluting the loss.
        if self.action_mapper is not None:
            logprobs = self.action_mapper.apply_mask(logprobs)
            entropy = self.action_mapper.apply_mask(entropy)

        # Debug: log logprobs statistics for diagnosing KL issues.
        _lp = logprobs.detach()
        _n_valid = int((_lp != 0).sum().item())
        self.logger.info(
            "[default_forward] logprobs: mean=%.4f, std=%.4f, min=%.4f, max=%.4f, "
            "valid_dims=%d, per_dim_abs_mean=%.6f",
            _lp.mean().item(), _lp.std().item(), _lp.min().item(), _lp.max().item(),
            _n_valid,
            _lp.abs().sum().item() / max(_n_valid, 1),
        )

        # Values from VLM hidden states (or stub zeros)
        if self.add_value_head and "vlm_hidden_states" in forward_inputs:
            vlm_hidden = forward_inputs["vlm_hidden_states"].to(device)
            vlm_mask = forward_inputs["vlm_attention_mask"].to(device)
            values = self.get_value_from_vlm(vlm_hidden, vlm_mask)
        else:
            values = torch.zeros(batch_size, device=device)

        return {
            "logprobs": logprobs.float(),
            "values": values.float(),
            "entropy": entropy.float(),
        }

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_vlm_batch(
        self,
        images: Union[np.ndarray, torch.Tensor],
        task_descriptions: list[str],
        state_tensor: torch.Tensor,
        device: torch.device,
        wrist_images: Union[np.ndarray, torch.Tensor, None] = None,
    ) -> dict[str, torch.Tensor]:
        """Convert env observations to Qwen3-VL processor format.

        This intentionally mirrors Xiaomi official ``server.py`` instead of
        using ``apply_chat_template``.  The assistant prefix is the empty COT
        block used by XR0:

            <|im_start|>assistant
            <|cot|><|/cot|><|im_end|>

        ``/no_cot`` is a user-side suffix, not text for the assistant to
        generate.
        """
        def _to_pil(img):
            if isinstance(img, torch.Tensor):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img
            return Image.fromarray(img_np.astype(np.uint8)).convert("RGB")

        if wrist_images is None or len(wrist_images) == 0:
            raise ValueError(
                "XR0 Xiaomi non-bridge/fractal prompt requires wrist_images "
                "because the official prompt contains both Base View and "
                "Left-Wrist View image pads."
            )

        texts: list[str] = []
        processor_images = []

        for i in range(len(images)):
            lang = task_descriptions[i] if i < len(task_descriptions) else ""
            instruction = (
                "<|im_start|>user\n"
                "The following observations are captured from multiple views.\n"
                "# Base View\n"
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                "# Left-Wrist View\n"
                "<|vision_start|><|image_pad|><|vision_end|>\n"
                "Generate robot actions for the task:\n"
                + lang.rstrip(".")
                + " /no_cot"
            )
            texts.append(instruction)

            # Flat image order must match the image_pad order in every text:
            # base_i first, then wrist_left_i.
            processor_images.append(_to_pil(images[i]))
            processor_images.append(_to_pil(wrist_images[i]))

        inputs = self.processor(
            text=texts,
            images=processor_images,
            videos=None,
            padding=True,
            return_tensors="pt",
        )

        batch = {
            k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        return batch

    # ------------------------------------------------------------------
    # Gradient checkpointing
    # ------------------------------------------------------------------

    def gradient_checkpointing_enable(self, **kwargs: Any) -> None:
        """Enable gradient checkpointing on supported submodules."""
        if hasattr(self.xr0_model, "vlm"):
            if hasattr(self.xr0_model.vlm, "gradient_checkpointing_enable"):
                self.xr0_model.vlm.gradient_checkpointing_enable(**kwargs)
            elif hasattr(self.xr0_model.vlm, "model") and hasattr(
                self.xr0_model.vlm.model, "visual"
            ):
                self.xr0_model.vlm.model.visual.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        """Disable gradient checkpointing on supported submodules."""
        if hasattr(self.xr0_model, "vlm"):
            if hasattr(self.xr0_model.vlm, "gradient_checkpointing_disable"):
                self.xr0_model.vlm.gradient_checkpointing_disable()
