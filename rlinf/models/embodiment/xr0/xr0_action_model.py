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
import random
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
    Flow-Noise method. Currently uses fixed noise_level. See lingbotvla
    for reference implementation with flow_noise / flow_sde / flow_cps.
    TODO: Add Flow-SDE method (ODE-to-SDE conversion) for better RL
    exploration. See lingbotvla.sample_mean_var_val for reference.
    TODO: Add value head (ValueHead MLP) for PPO critic. Currently
    values are stub zeros. Use rlinf.models.embodiment.modules.value_head.
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

        # Action mapper: handles 32D <-> env_dim conversion with valid_action_mask.
        # Takes precedence over action_env_dim (simple slicing).
        if action_mapper is not None:
            self.action_mapper = action_mapper
            self.action_env_dim = action_mapper.env_action_dim
        else:
            self.action_mapper = None
            # Fallback: simple slicing to first N dims
            self.action_env_dim = int(action_env_dim) if action_env_dim else int(action_dim)

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
                source, trust_remote_code=True
            )
            self._processor.tokenizer.padding_side = "right"
        return self._processor

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

        log_prob = -0.5 * ((sample - mu) / sigma_safe) ** 2
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

    # ------------------------------------------------------------------
    # Flow-SDE denoising helpers
    # ------------------------------------------------------------------

    def _compute_denoise_step(
        self,
        x_t: torch.Tensor,
        v_t: torch.Tensor,
        timesteps: torch.Tensor,
        idx: int,
        mode: Literal["train", "eval"] = "train",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute one denoising step with Flow-SDE noise.

        Implements the ODE-to-SDE conversion for rectified flow:
        - σ = a√(τ/(1-τ)) where a is noise_level, τ is current timestep
        - Drift correction: σ²δ/(2τ) subtracted from x1 weight
        - x_t_std = √δ · σ

        Args:
            x_t: Current noisy action ``(B, C, D)``.
            v_t: Predicted velocity ``(B, C, D)``.
            timesteps: Full timestep schedule ``(num_steps+1,)``.
            idx: Current step index.
            mode: ``"train"`` adds noise, ``"eval"`` is deterministic.

        Returns:
            Tuple of ``(x_t_mean, x_t_std, log_prob)``.
        """
        # Get the original dtype from x_t
        orig_dtype = x_t.dtype

        t_val = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        # Expand for broadcasting: (B, C, D) - use x_t's dtype
        t_input = t_val.view(1, 1, 1).expand_as(x_t).to(dtype=orig_dtype)
        delta_input = delta.view(1, 1, 1).expand_as(x_t).to(dtype=orig_dtype)

        # Predicted endpoints
        x0_pred = x_t - v_t * t_input  # x0 = x_t - t·v_t
        x1_pred = x_t + v_t * (1 - t_input)  # x1 = x_t + (1-t)·v_t

        if mode == "eval":
            # Deterministic: no noise
            x0_weight = 1 - (t_input - delta_input)
            x1_weight = t_input - delta_input
            x_t_std = torch.zeros_like(x_t)
        elif mode == "train":
            if self.noise_method == "flow_sde":
                # Flow-SDE: σ = a√(τ/(1-τ))
                # Handle τ=1 edge case: use τ[1] instead of 1-τ=0
                t_safe = torch.where(
                    timesteps == 1.0, timesteps[1], timesteps
                )
                sigmas = self.noise_level * torch.sqrt(
                    timesteps / (1 - t_safe)
                )
                # Remove the last element (τ=0)
                sigmas = sigmas[:-1]
                sigma_i = sigmas[idx].view(1, 1, 1).expand_as(x_t).to(dtype=orig_dtype)

                # Weights with drift correction
                x0_weight = 1 - (t_input - delta_input)
                x1_weight = t_input - delta_input - (
                    sigma_i**2 * delta_input / (2 * t_input)
                )
                x_t_std = torch.sqrt(delta_input) * sigma_i
            else:
                # Fallback: fixed noise (legacy behavior)
                sigma = self.noise_level * math.sqrt(delta)
                x0_weight = 1 - (t_input - delta_input)
                x1_weight = t_input - delta_input
                x_t_std = torch.full_like(x_t, sigma)

        # Mean prediction
        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight

        # Log probability of the next state under N(x_t_mean, x_t_std)
        if mode == "train" and self.noise_method == "flow_sde":
            # Sample from the distribution
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
        # VLM forward to get KV-cache and hidden states.
        # Temporarily disable gradient checkpointing on ALL VLM sub-modules
        # so the @check_model_inputs decorator does not force use_cache=False.
        vlm_inputs = {
            k: v.to(device) for k, v in vlm_batch.items() if isinstance(v, torch.Tensor)
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

        # Capture VLM hidden states for value computation
        # Qwen3VLCausalLMOutputWithPast has hidden_states (tuple) not last_hidden_state
        if hasattr(vlm_outputs, "hidden_states") and vlm_outputs.hidden_states is not None:
            vlm_hidden_states = vlm_outputs.hidden_states[-1]  # Last layer hidden states
        else:
            # Fallback: use logits as proxy (should not happen with output_hidden_states=True)
            vlm_hidden_states = vlm_outputs.logits
        vlm_attention_mask = vlm_inputs.get(
            "attention_mask",
            torch.ones(batch_size, vlm_hidden_states.shape[1], device=device, dtype=torch.long),
        )

        # Position ids for DiT (continue from VLM's last position)
        action_len = self.num_action_chunks
        state_len = state_tensor.shape[1]  # typically 1
        q_len = action_len + state_len + 1  # +1 for sink token
        position_ids = (
            torch.arange(0, q_len, device=device)
            .view(1, 1, -1)
            .repeat(3, batch_size, 1)
            + vlm_outputs.position_ids.max(dim=-1)[0][..., None]
            + 1
        )

        # Attention mask
        cache_mask = vlm_inputs.get(
            "attention_mask",
            torch.ones(batch_size, 1, device=device, dtype=torch.long),
        )
        cache_mask = cache_mask[:, None, :].expand(-1, q_len, -1)

        # Build causal mask for DiT tokens
        s_len = state_len + 1
        a_len = action_len
        mask_ss = torch.tril(torch.ones(s_len, s_len, device=device))
        mask_sa = torch.zeros(s_len, a_len, device=device)
        mask_as = torch.ones(a_len, s_len, device=device)
        mask_aa = torch.tril(torch.ones(a_len, a_len, device=device))
        local_window = getattr(self.xr0_model, "local_window", 4)
        mask_aa = mask_aa * torch.triu(
            torch.ones(a_len, a_len, device=device), diagonal=-local_window
        )
        causal_mask = torch.cat(
            [torch.cat([mask_ss, mask_sa], dim=1),
             torch.cat([mask_as, mask_aa], dim=1)],
            dim=0,
        )
        attn_mask = torch.cat(
            [cache_mask, causal_mask[None].expand(batch_size, -1, -1)], dim=-1
        )[:, None].bool()

        # State embedding (pad/truncate to match model's expected dim)
        state_tensor_padded = self._pad_state(state_tensor)
        state_embed = self.xr0_model.state_projector(
            state_tensor_padded.to(device=device, dtype=torch.bfloat16)
        )

        # Position embeddings (RoPE)
        dummy_action = torch.zeros(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )
        position_embeds = self.xr0_model.rotary_emb(dummy_action, position_ids)

        # Action mask (all ones)
        action_mask = torch.ones(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )

        # Timestep schedule: [1.0, 0.8, 0.6, 0.4, 0.2, 0.0] for 5 steps
        timesteps = torch.linspace(
            1.0, 0.0, self.num_steps + 1, device=device
        )

        # For Flow-SDE: pick one random step for policy gradient
        # (all steps add noise, but only one step's logprob is used)
        if mode == "train":
            denoise_ind = random.randint(0, self.num_steps - 1)
        else:
            denoise_ind = -1  # all deterministic

        # Denoising loop
        x_t = torch.randn(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )
        chains = [x_t.detach().clone()]
        log_probs = []

        for idx in range(self.num_steps):
            t_val = timesteps[idx]

            # DiT forward: predict velocity
            t_tensor = t_val.view(1, 1, 1).expand(
                batch_size, 1, 1
            ).to(dtype=torch.bfloat16)
            v_t = self.xr0_model.dit_forward(
                x_t, t_tensor, action_mask, state_embed,
                position_embeds, past_key_values, attn_mask,
            )

            # Compute denoising step with Flow-SDE
            step_mode = mode if idx == denoise_ind or mode == "eval" else "eval"
            x_t, x_t_std, log_prob = self._compute_denoise_step(
                x_t, v_t, timesteps, idx, mode=step_mode
            )

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
        if self.add_value_head:
            prev_values = self.get_value_from_vlm(
                vlm_hidden_states.detach(), vlm_attention_mask.detach()
            )
        else:
            prev_values = torch.zeros(batch_size, device=device)

        return {
            "actions": x_t[:, :action_len, : self.action_dim],
            "chains": chains_tensor,
            "prev_logprobs": prev_logprobs,
            "prev_values": prev_values,
            "denoise_inds": torch.full((batch_size,), denoise_ind, dtype=torch.long),
            "vlm_hidden_states": vlm_hidden_states.detach(),
            "vlm_attention_mask": vlm_attention_mask.detach(),
        }

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
        states = env_obs["states"]
        task_descriptions = env_obs.get("task_descriptions") or [""] * len(images)
        batch_size = len(images)

        # State tensor: (B, 1, STATE_DIM)
        state_tensor = torch.from_numpy(np.asarray(states, dtype=np.float32))
        if state_tensor.ndim == 1:
            state_tensor = state_tensor.unsqueeze(0)
        if state_tensor.ndim == 2:
            state_tensor = state_tensor.unsqueeze(1)

        # Build VLM batch
        vlm_batch = self._build_vlm_batch(
            images, task_descriptions, state_tensor, device
        )

        # Run step-by-step denoising with chain recording
        outputs = self.sample_actions(vlm_batch, state_tensor, device, mode=mode)

        # Denormalize actions
        actions_np = outputs["actions"].float().cpu().numpy()
        if self.action_mean is not None and self.action_std is not None:
            mean = self.action_mean.cpu().numpy()
            std = self.action_std.cpu().numpy()
            actions_np = denormalize_action(actions_np, mean, std)

        actions_np = actions_np[:, : self.num_action_chunks, :]

        # Build forward_inputs for training replay
        forward_inputs: dict[str, Any] = {
            "chains": outputs["chains"].cpu(),
            "denoise_inds": outputs["denoise_inds"].cpu(),
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

        result = {
            "prev_logprobs": outputs["prev_logprobs"].cpu(),
            "prev_values": outputs["prev_values"].cpu(),
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

        chains = forward_inputs.get("chains")  # (B, num_steps+1, C, D)
        denoise_inds = forward_inputs.get("denoise_inds")  # (B,)

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

        # Re-run VLM forward to get current KV-cache and position ids.
        # We rebuild the VLM batch from stored forward_inputs.
        vlm_keys = ["input_ids", "attention_mask", "pixel_values", "image_grid_thw"]
        vlm_batch = {}
        for k in vlm_keys:
            if k in forward_inputs:
                vlm_batch[k] = forward_inputs[k].to(device)

        # Run the VLM forward under torch.no_grad() and with gradient
        # checkpointing disabled.  We only need the KV cache for the DiT;
        # gradients flow only through the DiT, not the VLM.  Running with
        # no_grad prevents FSDP from resharding VLM parameters after the
        # forward (which would cause storage-of-size-0 errors during the
        # backward through the DiT's attention that references the cache).
        vlm_module = self.xr0_model.vlm
        gc_states: list[tuple[torch.nn.Module, bool]] = []
        for module in vlm_module.modules():
            if getattr(module, "gradient_checkpointing", False):
                gc_states.append((module, True))
                module.gradient_checkpointing = False
        try:
            with torch.no_grad():
                vlm_outputs = vlm_module(**vlm_batch, use_cache=True)
        finally:
            for module, state in gc_states:
                module.gradient_checkpointing = state

        if vlm_outputs.past_key_values is None:
            self.logger.error(
                "VLM past_key_values is None even after disabling gradient "
                "checkpointing. The KV cache will not be available for DiT."
            )
        past_key_values = list(vlm_outputs.past_key_values)
        vlm_pos_max = vlm_outputs.position_ids.max(dim=-1)[0]

        cache_attn_mask = vlm_batch.get(
            "attention_mask",
            torch.ones(batch_size, 1, device=device, dtype=torch.long),
        )

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
        mask_aa = torch.tril(torch.ones(a_len, a_len, device=device))
        local_window = getattr(self.xr0_model, "local_window", 4)
        mask_aa = mask_aa * torch.triu(
            torch.ones(a_len, a_len, device=device), diagonal=-local_window
        )
        causal_mask = torch.cat(
            [torch.cat([mask_ss, mask_sa], dim=1),
             torch.cat([mask_as, mask_aa], dim=1)],
            dim=0,
        )
        attn_mask = torch.cat(
            [cache_mask, causal_mask[None].expand(batch_size, -1, -1)], dim=-1
        )[:, None].bool()

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

        action_mask = torch.ones(
            (batch_size, action_len, self.action_dim),
            device=device, dtype=torch.bfloat16,
        )

        timesteps = torch.linspace(
            1.0, 0.0, self.num_steps + 1, device=device
        )

        # Replay: get x_t and x_{t+1} from the recorded chain
        x_t = chains[:, denoise_ind]  # (B, C, D)
        x_next = chains[:, denoise_ind + 1]  # recorded next state

        t_val = timesteps[denoise_ind]

        # Current policy's velocity prediction
        t_tensor = t_val.view(1, 1, 1).expand(
            batch_size, 1, 1
        ).to(dtype=torch.bfloat16)
        v_t = self.xr0_model.dit_forward(
            x_t, t_tensor, action_mask, state_embed,
            position_embeds, past_key_values, attn_mask,
        )

        # Compute mean and std using Flow-SDE
        x_t_mean, x_t_std, _ = self._compute_denoise_step(
            x_t, v_t, timesteps, denoise_ind, mode="train"
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
    ) -> dict[str, torch.Tensor]:
        """Convert env observations to Qwen3-VL processor format."""
        pil_images = []
        for img in images:
            # Handle both Tensor and numpy array inputs
            if isinstance(img, torch.Tensor):
                img_np = img.detach().cpu().numpy()
            else:
                img_np = img
            pil_img = Image.fromarray(img_np.astype(np.uint8))
            pil_img = resize_image(pil_img, factor=32, max_pixels=90000)
            pil_images.append(pil_img)

        messages = []
        for i, pil_img in enumerate(pil_images):
            instruction = task_descriptions[i] if i < len(task_descriptions) else ""
            messages.append(
                [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "\n# Ego View\n"},
                            {"type": "image", "image": pil_img},
                            {
                                "type": "text",
                                "text": (
                                    "\nGenerate robot actions"
                                    " for the task:\n"
                                    + instruction
                                ),
                            },
                        ],
                    },
                    {
                        "role": "assistant",
                        "content": [{"type": "text", "text": "<bot></bot>"}],
                    },
                ]
            )

        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            padding=True,
            images_kwargs={"do_resize": False},
        )

        batch = {
            k: v.to(device) for k, v in inputs.items() if isinstance(v, torch.Tensor)
        }
        batch["state"] = state_tensor.to(device=device, dtype=torch.bfloat16)
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
