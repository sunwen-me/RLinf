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

"""RLinf policy adapter for Xiaomi-Robotics-1 (XR-1).

XR-1 (HuggingFace ``model_type: mibot``) couples a Qwen3-VL VLM with a DiT
action expert that reuses the VLM KV cache (Mixture-of-Transformers).  The
action head is a rectified-flow model: the shipped sampler integrates
``x <- x + v * dt`` with ``t`` running from ``0`` (noise) to ``1`` (action).

RLinf's flow-matching RL recipes (see ``lingbotvla``/``openpi``) use the pi0
time convention ``tau = 1 - t``, in which the velocity is ``v_pi0 = -v_xr1`` and
the Euler grid is ``linspace(1, 1/N, N) + [0.0]``.  Substituting those two
identities into the pi0 sampler reproduces XR-1's update exactly, so all of the
SDE/logprob machinery (``flow_sde``/``flow_cps``/``flow_noise``) is reused
verbatim here and only the network call differs.
"""

import math
import random
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoConfig, AutoModel, AutoProcessor

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.modules.explore_noise_net import ExploreNoiseNet
from rlinf.models.embodiment.modules.value_head import ValueHead
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import copy_dict_tensor

PIL_BILINEAR = (
    Image.Resampling.BILINEAR if hasattr(Image, "Resampling") else Image.BILINEAR
)

DEFAULT_RL_TRAINABLE_SCOPE = "action_expert"

# The only robot type shipped with the RoboCasa release of XR-1.
DEFAULT_ROBOT_TYPE = "robocasa_mg"

# XR-1 consumes eight raw (unnormalized) proprioceptive values: the seven arm
# joint positions followed by the first gripper finger joint, zero-padded to
# ``config.state_dim``.  RLinf's RoboCasa state layout keeps the arm joints at
# ``[25:32]`` (requires ``env.*.include_joint_state: True`` and
# ``state_space: 32d``) and ``robot0_gripper_qpos`` at ``[7:9]``.
DEFAULT_STATE_INDICES = [25, 26, 27, 28, 29, 30, 31, 7]

# Prompt pieces of the official RoboCasa evaluation script.  Reproducing them
# by hand (instead of going through ``apply_chat_template``) keeps tokenization
# under our control, which is what lets us pad to a fixed width so rollout
# buffers can concatenate ``forward_inputs`` across environment steps.
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_VISION_START = "<|vision_start|>"
_VISION_END = "<|vision_end|>"
_IMAGE_PAD = "<|image_pad|>"
_MULTI_VIEW_HEADER = "The following observations are captured from multiple views.\n"
_BASE_VIEW_HEADER = "# Base View\n"
_WRIST_VIEW_HEADER = "\n# Left-Wrist View\n"
_TASK_HEADER = "\nGenerate robot actions for the task:\n"
_NO_COT_SUFFIX = " /no_cot"
_ASSISTANT_PREFILL = "<cot></cot>"

# Trainable keywords of ``rl_trainable_scope: action_expert``: everything that
# is not part of the frozen ``vlm`` subtree.
ACTION_EXPERT_KEYWORDS = (
    "dit.",
    "state_projector",
    "action_projector",
    "action_output_layer",
    "t_embedder",
    "t_projector",
    "sink",
)


@dataclass
class Observation:
    """Env-side observation consumed by the XR-1 adapter."""

    image: Any
    state: Any
    prompt: Optional[Any] = None
    wrist_images: Optional[Any] = None
    extra_view_images: Optional[Any] = None

    @classmethod
    def from_dict(cls, d: dict):
        return cls(
            image=d.get("image"),
            state=d.get("state"),
            prompt=d.get("prompt"),
            wrist_images=d.get("wrist_images"),
            extra_view_images=d.get("extra_view_images"),
        )


class Xr1ActionModel(nn.Module, BasePolicy):
    """Xiaomi-Robotics-1 (XR-1) wrapper for RLinf flow-matching RL (GRPO/PPO)."""

    @property
    def _no_split_modules(self) -> list[str]:
        # Class names are resolved by name against the loaded module tree, so the
        # vendored Qwen3-VL blocks inside ``modeling_mibot.py`` match as well.
        no_split_modules = [
            "Qwen3VLTextDecoderLayer",
            "Qwen3VLVisionBlock",
            "DecoderLayer",
        ]
        if self.noise_method == "flow_noise":
            no_split_modules.append("ExploreNoiseNet")
        return no_split_modules

    @property
    def _no_split_names(self) -> list[str]:
        # With ``rl_trainable_scope: action_expert`` the whole ``vlm`` subtree is
        # frozen, so every *trainable* leaf outside the DiT layers has to be
        # wrapped on its own -- otherwise FSDP1 would flatten it together with
        # the frozen VLM leftovers and reject the non-uniform ``requires_grad``.
        #
        # ``embed_tokens``/``lm_head`` are deliberately absent: XR-1's text tower
        # sets ``tie_word_embeddings: True`` and wrapping both halves of a shared
        # weight breaks the tie.  They stay in the root unit, which then holds
        # frozen parameters only.  Keep ``tie_word_embeddings: True`` in the
        # model YAML so the FSDP2 path skips its per-``nn.Embedding`` rule too.
        no_split_names = [
            "visual",
            "state_projector",
            "action_projector",
            "action_output_layer",
            "t_embedder",
            "t_projector",
            "sink",
            "value_head",
        ]
        if self.noise_method == "flow_noise":
            no_split_names.append("noise_head")
        return no_split_names

    def __init__(self, config, torch_dtype=torch.bfloat16):
        super().__init__()
        self.config = config
        self.torch_dtype = torch_dtype
        self.logger = get_logger()
        self.global_step = 0
        self._action_mask_cache = None

        model_path = getattr(config, "model_path", None)
        if not model_path:
            raise ValueError(
                "XR-1 requires actor.model.model_path pointing at a HuggingFace "
                "checkpoint exported with `model_type: mibot`."
            )

        hf_config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
        load_kwargs = {"trust_remote_code": True}
        attn_implementation = self._cfg_get("attn_implementation", None)
        if attn_implementation is not None:
            load_kwargs["attn_implementation"] = str(attn_implementation)

        if self._cfg_get("random_init", False):
            # Debug/CI path: build the architecture from config only, so the
            # denoising and log-probability plumbing can be exercised without
            # the ~5B checkpoint.
            self.logger.warning(
                "XR-1 random_init=True: building MiBoT from config without "
                "pretrained weights. Do not use this for training runs."
            )
            self.model = AutoModel.from_config(hf_config, **load_kwargs).to(
                self.torch_dtype
            )
        else:
            self.model = AutoModel.from_pretrained(
                model_path, config=hf_config, dtype=self.torch_dtype, **load_kwargs
            )

        # ``TimestepEmbedder`` stores its output dtype as a plain Python
        # attribute defaulting to bfloat16, so ``Module.to()`` cannot update it.
        # Re-align it or the timestep MLP sees a mismatched input dtype.
        t_embedder = getattr(self.model, "t_embedder", None)
        if t_embedder is not None and hasattr(t_embedder, "dtype"):
            t_embedder.dtype = self.torch_dtype

        self.model_action_dim = int(hf_config.action_dim)
        self.model_state_dim = int(hf_config.state_dim)
        self.state_length = int(getattr(hf_config, "state_length", 1))
        if self.state_length != 1:
            raise ValueError(
                "XR-1 adapter only supports checkpoints with state_length == 1, "
                f"got {self.state_length}."
            )
        self.dit_hidden_size = int(hf_config.dit_config.hidden_size)
        self.vlm_hidden_size = int(hf_config.vlm_config.text_config.hidden_size)

        processor_path = (
            self._cfg_get("processor_path", None)
            or getattr(config, "tokenizer_path", None)
            or model_path
        )
        self.processor = AutoProcessor.from_pretrained(
            processor_path,
            trust_remote_code=True,
            use_fast=bool(self._cfg_get("use_fast_processor", False)),
        )
        self.tokenizer = self.processor.tokenizer
        self.image_processor = self.processor.image_processor
        # Fixed-width left padding keeps the prompt at the right edge of the
        # prefix, which is what the DiT's position offset assumes.
        self.tokenizer.padding_side = "left"
        self.image_token_id = self.tokenizer.convert_tokens_to_ids(_IMAGE_PAD)

        self.robot_type = str(self._cfg_get("robot_type", DEFAULT_ROBOT_TYPE))
        available_robot_types = list(self.processor.action_config.keys())
        if self.robot_type not in available_robot_types:
            raise ValueError(
                f"XR-1 robot_type `{self.robot_type}` is not registered in the "
                f"processor. Available: {available_robot_types}."
            )
        action_stats_mean = self.processor.action_config[self.robot_type]["mean"]
        if action_stats_mean.shape[-1] != self.model_action_dim:
            raise ValueError(
                "XR-1 action statistics width does not match config.action_dim: "
                f"{action_stats_mean.shape[-1]} vs {self.model_action_dim}."
            )

        self.action_chunk = int(
            getattr(config, "num_action_chunks", action_stats_mean.shape[-2])
        )
        if self.action_chunk != action_stats_mean.shape[-2]:
            raise ValueError(
                "XR-1 predicts a fixed action horizon; set "
                f"actor.model.num_action_chunks to {action_stats_mean.shape[-2]} "
                f"(got {self.action_chunk})."
            )

        self.action_index_map = self._cfg_get("action_index_map", None)
        if self.action_index_map is not None:
            self.action_index_map = [int(i) for i in self.action_index_map]
        self.action_env_dim = int(
            self._cfg_get("action_env_dim", getattr(config, "action_dim", 7))
        )
        if not 0 < self.action_env_dim <= self.model_action_dim:
            raise ValueError(
                "XR-1 action_env_dim must be in the range "
                f"[1, {self.model_action_dim}], got {self.action_env_dim}."
            )
        active_action_dims = int(
            (self.processor.get_action_mask(self.robot_type)[0, 0] > 0).sum().item()
        )
        if self.action_env_dim > active_action_dims:
            self.logger.warning(
                "XR-1 action_env_dim=%d exceeds the %d dimensions that are "
                "active in the `%s` action statistics; the extra dimensions are "
                "untrained padding.",
                self.action_env_dim,
                active_action_dims,
                self.robot_type,
            )

        self.state_indices = [
            int(i) for i in self._cfg_get("state_indices", DEFAULT_STATE_INDICES)
        ]
        if len(self.state_indices) > self.model_state_dim:
            raise ValueError(
                f"XR-1 state_indices selects {len(self.state_indices)} values, "
                f"which exceeds config.state_dim={self.model_state_dim}."
            )

        self.num_steps = int(self._cfg_get("num_steps", 5))
        if self.num_steps < 2:
            raise ValueError(f"XR-1 num_steps must be >= 2, got {self.num_steps}.")
        self.noise_method = str(self._cfg_get("noise_method", "flow_sde"))
        self.image_size = int(self._cfg_get("image_size", 256))
        self.crop_ratio = float(self._cfg_get("crop_ratio", 0.95))
        self.max_prompt_length = int(self._cfg_get("max_prompt_length", 320))

        # --- RL knobs (names and semantics mirror the lingbotvla adapter) ---
        self.joint_logprob = bool(self._cfg_get("joint_logprob", False))
        self.ignore_last = bool(self._cfg_get("ignore_last", False))
        self.noise_level = float(self._cfg_get("noise_level", 0.5))
        self.noise_anneal = bool(self._cfg_get("noise_anneal", False))
        self.noise_params = [
            float(v) for v in self._cfg_get("noise_params", [0.7, 0.3, 400])
        ]
        self.chunk_critic_input = bool(self._cfg_get("chunk_critic_input", False))
        self.detach_critic_input = bool(self._cfg_get("detach_critic_input", False))
        self.add_value_head = bool(self._cfg_get("add_value_head", False))
        self.rl_trainable_scope = self._cfg_get(
            "rl_trainable_scope", DEFAULT_RL_TRAINABLE_SCOPE
        )
        trainable_keywords = self._cfg_get("trainable_keywords", None)
        self.trainable_keywords = (
            [str(k) for k in trainable_keywords]
            if trainable_keywords is not None
            else list(ACTION_EXPERT_KEYWORDS)
        )

        self.use_vlm_value = (
            bool(self._cfg_get("value_after_vlm", False)) and self.add_value_head
        )
        if self.add_value_head:
            self.value_head = ValueHead(
                input_dim=self.vlm_hidden_size
                if self.use_vlm_value
                else self.dit_hidden_size,
                hidden_sizes=(512, 256, 128),
                output_dim=1,
                activation="relu",
                bias_last=True,
            ).to(self.torch_dtype)

        if self.noise_method == "flow_noise":
            self.noise_head = ExploreNoiseNet(
                in_dim=self.dit_hidden_size,
                out_dim=self.model_action_dim,
                hidden_dims=[128, 64],
                activation_type="tanh",
                noise_logvar_range=[
                    float(v) for v in self._cfg_get("noise_logvar_range", [0.08, 0.16])
                ],
                noise_scheduler_type="learn",
            ).to(self.torch_dtype)

        for name, module in self.named_modules():
            path_parts = name.split(".")
            setattr(module, "_fsdp_wrap_name", path_parts[-1] if path_parts else name)

        self._apply_rl_trainable_scope()

    # ------------------------------------------------------------------
    # config / bookkeeping helpers
    # ------------------------------------------------------------------

    def _cfg_get(self, key: str, default=None):
        """Read ``key`` from the nested ``xr1:`` block, then from ``actor.model``.

        Looking at both levels keeps the YAML tolerant: knobs may be grouped
        under ``model.xr1`` for readability or set directly on ``actor.model``
        (which is where the shared embodied plumbing puts things like
        ``num_action_chunks``).  An explicitly configured ``null`` falls back to
        ``default``.
        """
        for source in (getattr(self.config, "xr1", None), self.config):
            if source is None:
                continue
            getter = getattr(source, "get", None)
            value = (
                getter(key, None) if callable(getter) else getattr(source, key, None)
            )
            if value is not None:
                return value
        return default

    def _apply_rl_trainable_scope(self):
        scope = self.rl_trainable_scope
        if scope is None or str(scope).lower() == "all":
            self._log_trainable_scope("all")
            return

        scope = str(scope)
        if scope != "action_expert":
            raise ValueError(
                "XR-1 rl_trainable_scope must be one of {None, 'all', "
                f"'action_expert'}}, got `{scope}`."
            )

        for param in self.model.parameters():
            param.requires_grad = False
        for name, param in self.model.named_parameters():
            if any(keyword in name for keyword in self.trainable_keywords):
                param.requires_grad = True
        # The RL-only heads never belong to the frozen backbone.
        for head_name in ("value_head", "noise_head"):
            head = getattr(self, head_name, None)
            if head is not None:
                for param in head.parameters():
                    param.requires_grad = True

        self._log_trainable_scope(scope)

    def _log_trainable_scope(self, scope: str):
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.parameters())
        self.logger.info(
            "XR-1 rl_trainable_scope=%s: trainable_params=%d / total_params=%d",
            scope,
            trainable_params,
            total_params,
        )

    def gradient_checkpointing_enable(self, **kwargs):
        # ``MiBoTForActionGeneration`` declares
        # ``supports_gradient_checkpointing = False``, but the Qwen3-VL tower it
        # wraps supports it; enabling it there is the only useful thing to do.
        vlm = getattr(self.model, "vlm", None)
        if vlm is not None and hasattr(vlm, "gradient_checkpointing_enable"):
            try:
                vlm.gradient_checkpointing_enable(**kwargs)
                return
            except ValueError as exc:  # pragma: no cover - depends on HF version
                self.logger.warning(
                    "XR-1: could not enable gradient checkpointing on the VLM: %s",
                    exc,
                )
                return
        self.logger.warning(
            "XR-1: gradient checkpointing is not supported by this checkpoint."
        )

    def set_global_step(self, global_step):
        self.global_step = global_step

    # ------------------------------------------------------------------
    # observation preprocessing
    # ------------------------------------------------------------------

    def obs_processor(self, env_obs):
        """Map RLinf env observations onto the XR-1 camera / state layout.

        XR-1 was trained on RoboCasa with the base view pair
        ``[robot0_agentview_left, robot0_agentview_right]`` followed by the
        ``robot0_eye_in_hand`` wrist view, which is exactly what
        ``RobocasaEnv`` emits as ``main_images`` / ``extra_view_images`` /
        ``wrist_images`` (already vertically flipped, so no extra flip here).
        """
        return {
            "image": env_obs.get("main_images", env_obs.get("prep_images")),
            "prompt": env_obs.get("task_descriptions", env_obs.get("prompt")),
            "state": env_obs.get("states", env_obs.get("prep_state")),
            "wrist_images": env_obs.get("wrist_images"),
            "extra_view_images": env_obs.get("extra_view_images"),
        }

    def _to_hwc_uint8(self, img) -> np.ndarray:
        if isinstance(img, Image.Image):
            return np.array(img.convert("RGB"))
        if isinstance(img, torch.Tensor):
            arr = img.detach().cpu().numpy()
        else:
            arr = np.asarray(img)
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]
        if arr.ndim == 2:
            arr = arr[..., None]
        if arr.ndim != 3:
            raise ValueError(f"XR-1 expects a single HWC/CHW image, got {arr.shape}.")
        # CHW -> HWC
        if arr.shape[0] in (1, 3, 4) and arr.shape[-1] not in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        if arr.dtype == np.uint8:
            return np.ascontiguousarray(arr)
        if arr.dtype == np.uint16:
            return np.ascontiguousarray((arr // 257).astype(np.uint8))
        arr = np.asarray(arr, dtype=np.float32)
        if arr.max() <= 1.0 + 1e-6:
            arr = arr * 255.0
        return np.ascontiguousarray(np.clip(arr, 0, 255).astype(np.uint8))

    def _to_pil(self, img) -> Image.Image:
        """Center-crop by ``crop_ratio`` then resize, as in the official eval."""
        arr = self._to_hwc_uint8(img)
        height, width = arr.shape[:2]
        crop_h = max(1, int(height * self.crop_ratio))
        crop_w = max(1, int(width * self.crop_ratio))
        top = (height - crop_h) // 2
        left = (width - crop_w) // 2
        arr = arr[top : top + crop_h, left : left + crop_w]
        pil = Image.fromarray(arr).convert("RGB")
        return pil.resize((self.image_size, self.image_size), PIL_BILINEAR)

    @staticmethod
    def _index_batch(images, i):
        if images is None:
            return None
        try:
            item = images[i]
        except (IndexError, KeyError, TypeError):
            return None
        return item

    def _collect_view_pils(self, observation: Observation, batch_size: int):
        """Return the flat, per-sample-ordered ``[base_l, base_r, wrist]`` list."""
        missing_extra_view = False
        missing_wrist = False
        flat_pils: list[Image.Image] = []
        for i in range(batch_size):
            base_left = self._index_batch(observation.image, i)
            if base_left is None:
                raise ValueError(
                    "XR-1 requires a base view; env observation is missing "
                    "`main_images`."
                )
            base_left_pil = self._to_pil(base_left)

            base_right = self._index_batch(observation.extra_view_images, i)
            if base_right is None:
                missing_extra_view = True
                base_right_pil = base_left_pil
            else:
                base_right_pil = self._to_pil(base_right)

            wrist = self._index_batch(observation.wrist_images, i)
            if wrist is None:
                missing_wrist = True
                wrist_pil = base_left_pil
            else:
                wrist_pil = self._to_pil(wrist)

            flat_pils.extend([base_left_pil, base_right_pil, wrist_pil])

        if missing_extra_view:
            self.logger.warning(
                "XR-1: `extra_view_images` missing; duplicating the base view. Set "
                "env.*.image_space: 3views for the layout the checkpoint expects."
            )
        if missing_wrist:
            self.logger.warning(
                "XR-1: `wrist_images` missing; duplicating the base view."
            )
        return flat_pils

    def _prepare_text(self, instructions: list[str], image_token_counts: list[int]):
        """Rebuild the eval-time chat prompt with pre-expanded image pads.

        The template is reproduced by hand instead of going through
        ``apply_chat_template`` so that tokenization stays under our control:
        RL rollout buffers concatenate ``forward_inputs`` across environment
        steps, which requires a fixed prompt width.
        """
        texts = []
        for i, instruction in enumerate(instructions):
            pads = [
                f"{_VISION_START}{_IMAGE_PAD * image_token_counts[3 * i + view]}{_VISION_END}"
                for view in range(3)
            ]
            texts.append(
                f"{_IM_START}user\n"
                f"{_MULTI_VIEW_HEADER}{_BASE_VIEW_HEADER}{pads[0]}{pads[1]}"
                f"{_WRIST_VIEW_HEADER}{pads[2]}"
                f"{_TASK_HEADER}{instruction}{_NO_COT_SUFFIX}{_IM_END}\n"
                f"{_IM_START}assistant\n{_ASSISTANT_PREFILL}{_IM_END}\n"
            )

        token_ids = self.tokenizer(texts, add_special_tokens=False)["input_ids"]
        longest = max(len(ids) for ids in token_ids)
        if longest > self.max_prompt_length:
            raise ValueError(
                f"XR-1 prompt needs {longest} tokens but max_prompt_length is "
                f"{self.max_prompt_length}. Raise model.xr1.max_prompt_length "
                "(or lower model.xr1.image_size)."
            )

        pad_id = self.tokenizer.pad_token_id
        if pad_id is None:
            pad_id = self.tokenizer.eos_token_id
        input_ids = torch.full(
            (len(token_ids), self.max_prompt_length), int(pad_id), dtype=torch.long
        )
        attention_mask = torch.zeros(
            (len(token_ids), self.max_prompt_length), dtype=torch.long
        )
        for i, ids in enumerate(token_ids):
            # Left padding keeps the prompt flush against the DiT suffix, which
            # is what the position-id offset below assumes.
            input_ids[i, self.max_prompt_length - len(ids) :] = torch.as_tensor(
                ids, dtype=torch.long
            )
            attention_mask[i, self.max_prompt_length - len(ids) :] = 1
        return input_ids, attention_mask

    def _prepare_state(self, states_raw, batch_size: int) -> torch.Tensor:
        """Select the XR-1 proprio dims and zero-pad to ``config.state_dim``."""
        if isinstance(states_raw, torch.Tensor):
            states = states_raw.detach().to(dtype=torch.float32, device="cpu")
        else:
            states = torch.as_tensor(np.asarray(states_raw), dtype=torch.float32)
        if states.ndim == 1:
            states = states[None]
        if states.ndim == 3 and states.shape[1] == 1:
            states = states[:, 0]
        if states.ndim != 2:
            raise ValueError(f"XR-1 expects a (B, D) state, got {tuple(states.shape)}.")

        max_index = max(self.state_indices)
        if states.shape[-1] <= max_index:
            raise ValueError(
                f"XR-1 state_indices needs at least {max_index + 1} state dims but "
                f"the env provides {states.shape[-1]}. For RoboCasa set "
                "`include_joint_state: True` and `state_space: 32d`."
            )
        indices = torch.as_tensor(self.state_indices, dtype=torch.long)
        selected = states.index_select(-1, indices)
        padded = torch.zeros(
            (batch_size, self.state_length, self.model_state_dim), dtype=torch.float32
        )
        padded[:, 0, : selected.shape[-1]] = selected
        return padded

    def _preprocess_vision_state(self, observation: Observation) -> dict[str, Any]:
        """Everything except the text prompt (recomputed on every RL forward)."""
        device = next(self.parameters()).device
        states_raw = observation.state
        if states_raw is None:
            raise ValueError("XR-1 requires `states` in the env observation.")
        batch_size = (
            states_raw.shape[0] if hasattr(states_raw, "shape") else len(states_raw)
        )

        flat_pils = self._collect_view_pils(observation, batch_size)
        vision_inputs = self.image_processor(images=flat_pils, return_tensors="pt")
        image_grid_thw = vision_inputs["image_grid_thw"]
        merge_length = int(self.image_processor.merge_size) ** 2
        image_token_counts = [
            int(grid.prod().item()) // merge_length for grid in image_grid_thw
        ]

        return {
            "batch_size": batch_size,
            "image_token_counts": image_token_counts,
            "pixel_values": vision_inputs["pixel_values"].to(
                device, dtype=self.torch_dtype
            ),
            "image_grid_thw": image_grid_thw.to(device),
            "state": self._prepare_state(states_raw, batch_size).to(
                device, dtype=self.torch_dtype
            ),
        }

    def _preprocess_observation(self, observation: Observation) -> dict[str, Any]:
        device = next(self.parameters()).device
        prepared = self._preprocess_vision_state(observation)
        batch_size = prepared["batch_size"]

        prompt = observation.prompt
        if prompt is None:
            instructions = [""] * batch_size
        elif isinstance(prompt, str):
            instructions = [prompt] * batch_size
        else:
            instructions = [str(p) for p in prompt]
        if len(instructions) != batch_size:
            raise ValueError(
                f"XR-1 got {len(instructions)} prompts for {batch_size} envs."
            )

        input_ids, attention_mask = self._prepare_text(
            instructions, prepared["image_token_counts"]
        )

        return {
            "input_ids": input_ids.to(device),
            "attention_mask": attention_mask.to(device),
            "pixel_values": prepared["pixel_values"],
            "image_grid_thw": prepared["image_grid_thw"],
            "state": prepared["state"],
        }

    # ------------------------------------------------------------------
    # action post-processing
    # ------------------------------------------------------------------

    def _select_env_action_dims(self, action_tensor: torch.Tensor) -> torch.Tensor:
        if self.action_index_map is None:
            return action_tensor[..., : self.action_env_dim]
        indices = torch.as_tensor(
            self.action_index_map, device=action_tensor.device, dtype=torch.long
        )
        return action_tensor.index_select(-1, indices)[..., : self.action_env_dim]

    def output_transform(self, outputs: dict) -> dict:
        """Unnormalize the flow output and cut it down to the env action dims."""
        actions = outputs["actions"][:, : self.action_chunk, :]
        actions = actions.to(torch.float32).cpu()
        if actions.shape[-1] == self.model_action_dim:
            # ``decode_action`` broadcasts the (1, horizon, action_dim) stats.
            actions = self.processor.decode_action(actions, self.robot_type)
        if actions.shape[-1] != self.action_env_dim:
            actions = self._select_env_action_dims(actions)
        outputs["actions"] = actions.to(torch.float32)
        return outputs

    # ------------------------------------------------------------------
    # flow-matching core
    # ------------------------------------------------------------------

    def _get_action_mask(self, batch_size: int, device) -> torch.Tensor:
        """Cached ``(B, horizon, action_dim)`` mask of the trained action dims."""
        cached = self._action_mask_cache
        if (
            cached is None
            or cached.shape[0] != batch_size
            or cached.device != device
            or cached.dtype != self.torch_dtype
        ):
            mask = self.processor.get_action_mask(self.robot_type, batch_size)
            cached = mask.to(device=device, dtype=self.torch_dtype)
            self._action_mask_cache = cached
        return cached

    @property
    def _vlm_is_frozen(self) -> bool:
        return not any(p.requires_grad for p in self.model.vlm.parameters())

    def _run_vlm(self, model_inputs: dict[str, Any]):
        """Single prefix pass that fills the KV cache the DiT attends to.

        ``self.model.vlm.model`` (rather than ``self.model.vlm``) is called on
        purpose: the extra ``lm_head`` projection over the 150k-token vocabulary
        is pure waste here, and the only two extra fields the CausalLM wrapper
        adds are ``position_ids`` (also on the base output) and an echo of the
        ``attention_mask`` we passed in.
        """
        return self.model.vlm.model(
            input_ids=model_inputs["input_ids"],
            attention_mask=model_inputs["attention_mask"],
            pixel_values=model_inputs["pixel_values"],
            image_grid_thw=model_inputs["image_grid_thw"],
            use_cache=True,
        )

    def _build_dit_context(
        self, model_inputs: dict[str, Any], vlm_outputs
    ) -> dict[str, Any]:
        """Precompute everything the DiT reuses across denoising steps."""
        attention_mask = model_inputs["attention_mask"]
        state = model_inputs["state"]
        batch_size = state.shape[0]
        device = state.device

        action_mask = self._get_action_mask(batch_size, device)
        dit_query_length = self.action_chunk + self.state_length + 1

        position_ids = getattr(vlm_outputs, "position_ids", None)
        if position_ids is None:
            # Defensive fallback for transformers versions whose Qwen3-VL output
            # does not carry mrope position ids: with left padding the last real
            # token sits at index ``sum(mask) - 1``.
            offset = (attention_mask.sum(dim=-1) - 1)[None, :, None].expand(
                3, batch_size, 1
            )
        else:
            offset = position_ids.max(dim=-1)[0][..., None]
        dit_position_ids = (
            torch.arange(0, dit_query_length, device=device)
            .view(1, 1, -1)
            .repeat(3, batch_size, 1)
            + offset
            + 1
        )
        position_embeds = self.model.rotary_emb(action_mask, dit_position_ids)

        causal = torch.ones(
            (dit_query_length, dit_query_length), dtype=torch.bool, device=device
        ).tril(diagonal=0)
        cache_mask = attention_mask.bool()[:, None, :].expand(-1, dit_query_length, -1)
        attn_mask = torch.cat(
            [cache_mask, causal[None].expand(batch_size, -1, -1)], dim=-1
        )[:, None]

        return {
            "action_mask": action_mask,
            "position_embeds": position_embeds,
            "attn_mask": attn_mask,
            "state_embed": self.model.state_projector(state),
            "past_key_values": vlm_outputs.past_key_values,
        }

    def _dit_forward(self, x_t: torch.Tensor, t_xr1: torch.Tensor, ctx: dict):
        """XR-1's ``dit_forward``, split so the value/noise heads see the hidden."""
        model = self.model
        t_embeds = model.t_embedder(t_xr1.to(self.torch_dtype) * 1000)
        t_embeds = model.t_projector(t_embeds).view(t_embeds.shape[0], 6, -1)

        noisy_action = x_t.to(self.torch_dtype) * ctx["action_mask"]
        noisy_action = model.action_projector(noisy_action)

        # ``sink`` is an ``nn.Embedding(1, dit_hidden)`` that FSDP wraps as its
        # own unit, so ``sink.weight`` is a flat-parameter view that only the
        # unit's own pre-forward hook re-creates.  Reading the attribute (as
        # upstream does) therefore keeps a view whose base the optimizer has
        # since mutated in place, and autograd rejects it from the second
        # update onwards ("Output 0 of ViewBackward0 is a view and its base ...
        # has been modified inplace").  Looking the single row up through the
        # module runs the hook and yields the same ``(B, 1, dit_hidden)``
        # tensor with the same gradient.
        sink = model.sink(
            torch.zeros(
                (noisy_action.shape[0], 1),
                dtype=torch.long,
                device=noisy_action.device,
            )
        )
        hidden_states = torch.cat(
            [sink, ctx["state_embed"], noisy_action], dim=1
        ).contiguous()
        hidden_states = model.dit(
            hidden_states,
            ctx["past_key_values"],
            ctx["attn_mask"],
            ctx["position_embeds"],
            t_embeds,
        )
        suffix_out = hidden_states[:, -noisy_action.shape[1] :, :]
        return suffix_out, model.action_output_layer(suffix_out)

    def sample_noise(self, shape, device) -> torch.Tensor:
        return torch.normal(
            mean=0.0, std=1.0, size=shape, dtype=torch.float32, device=device
        )

    def sample_mean_var_val(
        self,
        x_t,
        idx,
        ctx,
        mode,
        denoise_steps,
        compute_values=True,
    ):
        """One Euler step expressed as a Gaussian transition ``N(mean, std)``.

        Written in the pi0 time convention ``tau = 1 - t_xr1``, where the XR-1
        velocity flips sign (``v_pi0 = -v_xr1``).  With ``mode="eval"`` and
        ``std = 0`` this reduces to XR-1's deterministic ``x <- x + v_xr1 / N``.
        """
        bsize = x_t.shape[0]
        device = x_t.device
        if isinstance(idx, int):
            idx = torch.tensor(idx, device=device).expand(bsize)

        if self.noise_anneal:
            noise_start, noise_end, anneal_steps = self.noise_params
            noise_level = noise_start + (noise_end - noise_start) * min(
                self.global_step, anneal_steps
            ) / max(anneal_steps, 1)
            noise_level = torch.tensor(noise_level, device=device)
        else:
            noise_level = torch.tensor(self.noise_level, device=device)

        timesteps = torch.linspace(
            1, 1 / denoise_steps, denoise_steps, device=device, dtype=torch.float32
        )
        timesteps = torch.cat(
            [timesteps, torch.zeros(1, device=device, dtype=torch.float32)]
        )

        t_input = timesteps[idx]
        delta = timesteps[idx] - timesteps[idx + 1]

        suffix_out, dit_out = self._dit_forward(x_t, 1.0 - t_input, ctx)
        # ``dit_out`` points from noise to data; the pi0 parameterization below
        # integrates the opposite direction.
        v_t = (-dit_out).to(torch.float32)

        if self.add_value_head and compute_values and not self.use_vlm_value:
            if self.chunk_critic_input:
                suffix_out_value = torch.mean(
                    suffix_out[:, : self.action_chunk], dim=1, keepdim=False
                )
            else:
                suffix_out_value = torch.mean(suffix_out, dim=1, keepdim=False)
            if self.detach_critic_input:
                suffix_out_value = suffix_out_value.detach()
            value_t = self.value_head(suffix_out_value)[:, 0]
        else:
            value_t = torch.zeros((bsize), device=device, dtype=self.torch_dtype)

        delta = delta[:, None, None].expand_as(x_t)
        t_input = t_input[:, None, None].expand_as(x_t)
        x0_pred = x_t - v_t * t_input
        x1_pred = x_t + v_t * (1 - t_input)

        if mode == "eval":
            x0_weight = 1 - (t_input - delta)
            x1_weight = t_input - delta
            x_t_std = torch.zeros_like(t_input)
        elif mode == "train":
            if self.noise_method == "flow_sde":
                sigmas = (
                    noise_level
                    * torch.sqrt(
                        timesteps
                        / (1 - torch.where(timesteps == 1, timesteps[1], timesteps))
                    )[:-1]
                )
                sigma_i = sigmas[idx][:, None, None].expand_as(x_t)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = t_input - delta - sigma_i**2 * delta / (2 * t_input)
                x_t_std = torch.sqrt(delta) * sigma_i
            elif self.noise_method == "flow_cps":
                cos_term = torch.cos(math.pi * noise_level / 2).to(device)
                sin_term = torch.sin(math.pi * noise_level / 2).to(device)
                x0_weight = torch.ones_like(t_input) - (t_input - delta)
                x1_weight = (t_input - delta) * cos_term
                x_t_std = (t_input - delta) * sin_term
            elif self.noise_method == "flow_noise":
                x0_weight = 1 - (t_input - delta)
                x1_weight = t_input - delta
                x_t_std = self.noise_head(suffix_out).to(torch.float32)
            else:
                raise ValueError(f"Invalid noise method: {self.noise_method}")
        else:
            raise ValueError(f"Invalid sampling mode: {mode}")

        x_t_mean = x0_pred * x0_weight + x1_pred * x1_weight
        return x_t_mean, x_t_std, value_t

    @staticmethod
    def get_logprob_norm(sample, mu, sigma):
        """Per-element Gaussian log-density; deterministic steps contribute 0."""
        sample = sample.to(torch.float32)
        mu = mu.to(torch.float32)
        sigma = sigma.to(torch.float32)
        mask = sigma == 0
        sigma_safe = torch.where(mask, torch.ones_like(sigma), sigma)
        log_prob = (
            -((sample - mu) ** 2) / (2 * sigma_safe**2)
            - torch.log(sigma_safe)
            - 0.5 * math.log(2 * math.pi)
        )
        return torch.where(mask, torch.zeros_like(log_prob), log_prob)

    @staticmethod
    def gaussian_entropy(sigma):
        sigma = sigma.to(torch.float32)
        sigma_safe = torch.where(sigma == 0, torch.ones_like(sigma), sigma)
        return 0.5 * torch.log(2 * math.pi * math.e * sigma_safe**2)

    def get_value_from_vlm(self, last_hidden_state, attention_mask):
        """Masked mean-pool of the VLM prefix, used when ``value_after_vlm``."""
        mask = attention_mask[..., None].to(last_hidden_state.dtype)
        pooled = (last_hidden_state * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        if self.detach_critic_input:
            pooled = pooled.detach()
        return self.value_head(pooled).squeeze(-1)

    # ------------------------------------------------------------------
    # rollout
    # ------------------------------------------------------------------

    @torch.no_grad()
    def sample_actions(
        self,
        observation: Observation,
        noise=None,
        mode: str = "train",
        compute_values: bool = True,
    ) -> dict[str, Any]:
        """Run the full XR-1 denoising rollout and record the sampling chain."""
        device = next(self.parameters()).device
        num_steps = self.num_steps

        model_inputs = self._preprocess_observation(observation)
        bsize = model_inputs["state"].shape[0]

        if noise is None:
            noise = self.sample_noise(
                (bsize, self.action_chunk, self.model_action_dim), device
            )
        else:
            noise = noise.to(device=device, dtype=torch.float32)
            if noise.shape[-1] < self.model_action_dim:
                pad_noise = self.sample_noise(
                    (*noise.shape[:-1], self.model_action_dim - noise.shape[-1]),
                    device,
                )
                noise = torch.cat([noise, pad_noise], dim=-1)

        vlm_outputs = self._run_vlm(model_inputs)
        ctx = self._build_dit_context(model_inputs, vlm_outputs)

        x_t = noise
        chains = [x_t]
        log_probs = []
        values = []

        if self.use_vlm_value:
            values_vlm = self.get_value_from_vlm(
                vlm_outputs.last_hidden_state, model_inputs["attention_mask"]
            )

        if self.joint_logprob:
            log_probs.append(
                self.get_logprob_norm(x_t, torch.zeros_like(x_t), torch.ones_like(x_t))
            )

        # A single denoising index is sampled per rollout step and repeated so
        # that the actor can rebuild exactly one stochastic transition later.
        if mode == "train":
            if self.joint_logprob:
                denoise_inds = torch.arange(num_steps, device=device)
            else:
                last_index = num_steps - 2 if self.ignore_last else num_steps - 1
                denoise_inds = torch.tensor(
                    [random.randint(0, last_index)] * num_steps, device=device
                )
        else:
            denoise_inds = torch.tensor([-1] * num_steps, device=device)
        denoise_inds = denoise_inds[None].repeat(bsize, 1)

        for idx in range(num_steps):
            sample_mode = "train" if idx == denoise_inds[0][idx] else "eval"
            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                x_t,
                idx,
                ctx,
                sample_mode,
                num_steps,
                compute_values,
            )
            x_t = x_t_mean + self.sample_noise(x_t.shape, device) * x_t_std
            log_prob = self.get_logprob_norm(x_t, x_t_mean, x_t_std)

            values.append(value_t)
            chains.append(x_t)
            log_probs.append(log_prob)

        # Keep all ``model_action_dim`` dims so ``decode_action`` can broadcast
        # its (1, horizon, action_dim) statistics; ``output_transform`` slices
        # down to the env action dims afterwards.
        model_actions = x_t[:, : self.action_chunk, :]
        chains = torch.stack(chains, dim=1)

        log_probs = self._select_env_action_dims(
            torch.stack(log_probs, dim=1)[:, :, : self.action_chunk, :]
        )
        if self.joint_logprob:
            log_probs = log_probs.mean(dim=1)
        else:
            log_probs = log_probs[torch.arange(log_probs.shape[0]), denoise_inds[:, 0]]

        if self.use_vlm_value:
            values = values_vlm[:, None]
        else:
            values = torch.stack(values, dim=1).mean(dim=-1, keepdim=True)

        return {
            "actions": model_actions,
            "chains": chains,
            "prev_logprobs": log_probs,
            "prev_values": values,
            "denoise_inds": denoise_inds,
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
        }

    def predict_action_batch(
        self,
        env_obs,
        mode: Literal["train", "eval"] = "train",
        compute_values: bool = True,
        **kwargs,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        processed_obs = self.obs_processor(env_obs)
        observation = Observation.from_dict(processed_obs)

        outputs = self.sample_actions(
            observation, mode=mode, compute_values=compute_values
        )
        actions = self.output_transform({"actions": outputs["actions"]})[
            "actions"
        ].numpy()

        forward_inputs = {
            "chains": outputs["chains"].cpu(),
            "denoise_inds": outputs["denoise_inds"].cpu(),
            "input_ids": outputs["input_ids"].cpu(),
            "attention_mask": outputs["attention_mask"].cpu(),
        }
        # The raw env views/states are ~4x cheaper to keep around than the
        # patchified ``pixel_values``, so the actor recomputes those instead.
        forward_inputs.update(
            copy_dict_tensor(
                {
                    k: v
                    for k, v in env_obs.items()
                    if k not in ["task_descriptions", "prompt"] and v is not None
                }
            )
        )

        return actions, {
            "prev_logprobs": outputs["prev_logprobs"].to(torch.float32),
            "prev_values": outputs["prev_values"].to(torch.float32),
            "forward_inputs": forward_inputs,
        }

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def get_log_prob_value(
        self,
        model_inputs: dict[str, Any],
        chains: torch.Tensor,
        denoise_inds: torch.Tensor,
        compute_values: bool = False,
    ):
        bsize = model_inputs["state"].shape[0]
        device = model_inputs["state"].device

        # The prefix only needs gradients when the VLM itself is being trained;
        # under ``rl_trainable_scope: action_expert`` it is frozen, and skipping
        # its activations is what makes the ~5B backbone fit alongside the DiT.
        with torch.set_grad_enabled(not self._vlm_is_frozen):
            vlm_outputs = self._run_vlm(model_inputs)
        ctx = self._build_dit_context(model_inputs, vlm_outputs)

        chains_log_probs = []
        chains_values = []
        chains_entropy = []

        if self.joint_logprob:
            num_steps = self.num_steps
            chains_log_probs.append(
                self.get_logprob_norm(
                    chains[:, 0],
                    torch.zeros_like(chains[:, 0]),
                    torch.ones_like(chains[:, 0]),
                )
            )
            chains_entropy.append(self.gaussian_entropy(torch.ones_like(chains[:, 0])))
        else:
            num_steps = 1

        batch_index = torch.arange(bsize, device=chains.device)
        for idx in range(num_steps):
            denoise_ind = denoise_inds[:, idx]
            chains_pre = chains[batch_index, denoise_ind]
            chains_next = chains[batch_index, denoise_ind + 1]

            x_t_mean, x_t_std, value_t = self.sample_mean_var_val(
                chains_pre,
                denoise_ind,
                ctx,
                "train",
                self.num_steps,
                compute_values,
            )

            chains_log_probs.append(
                self.get_logprob_norm(chains_next, x_t_mean, x_t_std)
            )
            chains_entropy.append(self.gaussian_entropy(x_t_std))

            if not self.use_vlm_value:
                chains_values.append(value_t)

        if self.use_vlm_value:
            chains_values.append(
                self.get_value_from_vlm(
                    vlm_outputs.last_hidden_state, model_inputs["attention_mask"]
                )
            )
        elif not chains_values:
            chains_values.append(
                torch.zeros(bsize, device=device, dtype=self.torch_dtype)
            )

        chains_log_probs = torch.stack(chains_log_probs, dim=1)
        chains_values = torch.stack(chains_values, dim=1)

        if self.noise_method == "flow_noise":
            chains_entropy = torch.stack(chains_entropy, dim=1)
        else:
            chains_entropy = torch.zeros_like(chains_log_probs)

        return chains_log_probs, chains_values, chains_entropy

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        raise NotImplementedError(
            f"XR-1 does not implement forward_type={forward_type}."
        )

    def default_forward(
        self,
        forward_inputs: dict[str, torch.Tensor],
        **kwargs,
    ) -> dict[str, Any]:
        compute_values = kwargs.get("compute_values", False)
        device = next(self.parameters()).device
        chains = forward_inputs["chains"].to(device, dtype=torch.float32)
        denoise_inds = forward_inputs["denoise_inds"].to(device)

        observation = Observation.from_dict(
            {
                "image": forward_inputs.get(
                    "main_images",
                    forward_inputs.get("prep_images", forward_inputs.get("images")),
                ),
                "state": forward_inputs.get("states", forward_inputs.get("prep_state")),
                "wrist_images": forward_inputs.get("wrist_images"),
                "extra_view_images": forward_inputs.get("extra_view_images"),
            }
        )
        prepared = self._preprocess_vision_state(observation)

        # The prompt is reused verbatim from the rollout so the KV cache the DiT
        # attends to is bit-identical; only the vision patches are recomputed.
        input_ids = forward_inputs["input_ids"].to(device)
        attention_mask = forward_inputs["attention_mask"].to(device)
        expected_image_tokens = (
            sum(prepared["image_token_counts"]) // prepared["batch_size"]
        )
        actual_image_tokens = int((input_ids[0] == self.image_token_id).sum().item())
        if expected_image_tokens != actual_image_tokens:
            raise ValueError(
                "XR-1 image token count changed between rollout and training "
                f"({actual_image_tokens} stored vs {expected_image_tokens} "
                "recomputed); keep model.xr1.image_size fixed across workers."
            )

        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "pixel_values": prepared["pixel_values"],
            "image_grid_thw": prepared["image_grid_thw"],
            "state": prepared["state"],
        }

        log_probs, value_t, entropy = self.get_log_prob_value(
            model_inputs, chains, denoise_inds, compute_values
        )

        log_probs = self._select_env_action_dims(
            log_probs[:, :, : self.action_chunk, :]
        )
        entropy = self._select_env_action_dims(entropy[:, :, : self.action_chunk, :])

        log_probs = log_probs.mean(dim=1)
        entropy = entropy.mean(dim=[1, 2, 3], keepdim=False)[:, None]
        value_t = value_t.mean(dim=-1, keepdim=False)

        return {
            "logprobs": log_probs.to(torch.float32),
            "values": value_t.to(torch.float32),
            "entropy": entropy.to(torch.float32),
        }
