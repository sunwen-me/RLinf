# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Pin how ``model.lora_target_modules`` scopes LoRA adapters.

XR-1 freezes its Qwen3-VL backbone (``rl_trainable_scope: action_expert``) and
only trains the diffusion action expert.  The built-in target list in
:func:`rlinf.models.get_model` matches by module-name suffix, which would attach
adapters to the frozen ``vlm`` subtree as well and would miss the action
expert's fused ``qkv_proj``.  These tests pin the regex escape hatch that lets a
model keep its adapters inside one subtree, and the census that makes the
resulting scope visible in a training log.
"""

import pytest
import torch
import torch.nn as nn

peft = pytest.importorskip("peft")

# The regex used by the XR-1 configs. ``peft`` matches a string
# ``target_modules`` against every module path with ``re.fullmatch``.
XR1_LORA_REGEX = r".*dit\..*(qkv_proj|o_proj|gate_proj|up_proj|down_proj)"

# The historical default: matched by name suffix, not by subtree.
DEFAULT_TARGETS = [
    "proj",
    "qkv",
    "fc1",
    "fc2",
    "q",
    "kv",
    "fc3",
    "out_proj",
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
    "lm_head",
]

D = 8


class _Attn(nn.Module):
    """Separate q/k/v, the way the Qwen3-VL text tower names them."""

    def __init__(self):
        super().__init__()
        self.q_proj = nn.Linear(D, D)
        self.k_proj = nn.Linear(D, D)
        self.v_proj = nn.Linear(D, D)
        self.o_proj = nn.Linear(D, D)


class _FusedAttn(nn.Module):
    """Fused qkv, the way the XR-1 action expert names it."""

    def __init__(self):
        super().__init__()
        self.qkv_proj = nn.Linear(D, 3 * D)
        self.o_proj = nn.Linear(D, D)


class _Mlp(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(D, D)
        self.up_proj = nn.Linear(D, D)
        self.down_proj = nn.Linear(D, D)


class _Block(nn.Module):
    def __init__(self, fused: bool):
        super().__init__()
        self.attn = _FusedAttn() if fused else _Attn()
        self.mlp = _Mlp()
        self.norm = nn.LayerNorm(D)


class _Inner(nn.Module):
    """Mimics the ``vlm`` / ``dit`` sibling layout of the XR-1 checkpoint."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.vlm = nn.ModuleDict(
            {"layers": nn.ModuleList(_Block(fused=False) for _ in range(n_layers))}
        )
        self.dit = nn.ModuleDict(
            {"layers": nn.ModuleList(_Block(fused=True) for _ in range(n_layers))}
        )
        self.action_projector = nn.Sequential(
            nn.Linear(D, D), nn.GELU(), nn.Linear(D, D)
        )


class _Policy(nn.Module):
    """The RL wrapper: holds the HF model under ``self.model``."""

    def __init__(self, n_layers: int = 2):
        super().__init__()
        self.model = _Inner(n_layers)

    def freeze_backbone(self):
        """What ``_apply_rl_trainable_scope('action_expert')`` does."""
        for name, param in self.model.named_parameters():
            param.requires_grad = not name.startswith("vlm.")


def _adapted_modules(model) -> set:
    """Paths of every module that received a LoRA adapter."""
    return {
        name
        for name, module in model.named_modules()
        if hasattr(module, "lora_A") and len(module.lora_A) > 0
    }


def _wrap(policy, target_modules, rank: int = 4, autocast: bool = False):
    cfg = peft.LoraConfig(
        r=rank,
        lora_alpha=rank,
        lora_dropout=0.0,
        target_modules=target_modules,
        init_lora_weights="gaussian",
    )
    return peft.get_peft_model(policy, cfg, autocast_adapter_dtype=autocast)


def test_regex_targets_only_the_action_expert():
    policy = _Policy(n_layers=2)
    policy.freeze_backbone()
    wrapped = _wrap(policy, XR1_LORA_REGEX)

    adapted = _adapted_modules(wrapped)
    assert adapted, "the regex matched nothing"
    assert not [n for n in adapted if ".vlm." in n], (
        f"LoRA leaked into the frozen backbone: {sorted(n for n in adapted if '.vlm.' in n)}"
    )
    # Five linear families per DiT block: qkv_proj, o_proj, gate/up/down_proj.
    assert len(adapted) == 2 * 5
    leaves = sorted({n.rsplit(".", 1)[-1] for n in adapted})
    assert leaves == [
        "down_proj",
        "gate_proj",
        "o_proj",
        "qkv_proj",
        "up_proj",
    ]


def test_default_target_list_would_leak_into_the_frozen_backbone():
    """Why XR-1 needs the regex: the suffix list ignores subtrees."""
    policy = _Policy(n_layers=2)
    policy.freeze_backbone()
    wrapped = _wrap(policy, DEFAULT_TARGETS)

    adapted = _adapted_modules(wrapped)
    assert [n for n in adapted if ".vlm." in n], (
        "expected the default list to reach the frozen VLM"
    )
    # ``qkv_proj`` matches neither "qkv" nor "q_proj" under peft's suffix rule,
    # so the action expert's attention input projection is left untouched.
    assert not [n for n in adapted if n.endswith("qkv_proj")]


def test_only_adapters_are_trainable():
    policy = _Policy(n_layers=2)
    policy.freeze_backbone()
    wrapped = _wrap(policy, XR1_LORA_REGEX)

    trainable = [n for n, p in wrapped.named_parameters() if p.requires_grad]
    assert trainable, "nothing is trainable"
    assert all("lora_" in n for n in trainable), (
        f"non-adapter parameters are trainable: {[n for n in trainable if 'lora_' not in n]}"
    )
    # Every adapted module contributes one A and one B matrix.
    assert len(trainable) == 2 * len(_adapted_modules(wrapped))


def test_adapter_starts_as_the_identity():
    """``lora_B`` is zero-initialised, so run6 starts from the SFT policy."""
    torch.manual_seed(0)
    policy = _Policy(n_layers=1)
    policy.freeze_backbone()
    x = torch.randn(2, D)
    before = policy.model.dit["layers"][0].mlp.up_proj(x).clone()

    wrapped = _wrap(policy, XR1_LORA_REGEX)
    after = wrapped.base_model.model.model.dit["layers"][0].mlp.up_proj(x)

    torch.testing.assert_close(before, after)


def test_rank_sets_the_adapter_size():
    policy = _Policy(n_layers=1)
    policy.freeze_backbone()
    for rank in (2, 8):
        wrapped = _wrap(_Policy(n_layers=1), XR1_LORA_REGEX, rank=rank)
        n_lora = sum(p.numel() for n, p in wrapped.named_parameters() if "lora_" in n)
        # qkv_proj is D->3D, the other four are D->D.
        expected = rank * (D + 3 * D) + 4 * (rank * D + D * rank)
        assert n_lora == expected, f"rank={rank}: {n_lora} != {expected}"


def _census_lines(monkeypatch):
    """Capture what :func:`rlinf.models.log_lora_census` would log."""
    logging_mod = pytest.importorskip("rlinf.utils.logging")
    records = []

    class _Logger:
        def info(self, msg, *args):
            records.append(msg % args)

    monkeypatch.setattr(logging_mod, "get_logger", lambda: _Logger())
    return records


def test_census_reports_the_adapted_module_count(monkeypatch):
    """The census is the only in-log record of where the adapters landed."""
    models = pytest.importorskip("rlinf.models")
    records = _census_lines(monkeypatch)

    policy = _Policy(n_layers=3)
    policy.freeze_backbone()
    wrapped = _wrap(policy, XR1_LORA_REGEX)
    models.log_lora_census(wrapped)

    assert len(records) == 1
    line = records[0]
    assert f"adapted_modules={3 * 5}" in line, line
    n_lora = sum(p.numel() for n, p in wrapped.named_parameters() if "lora_" in n)
    assert f"trainable_params={n_lora}" in line, line


def test_census_is_silent_outside_a_worker(monkeypatch):
    """``get_logger`` returns ``None`` off-worker; the census must not raise."""
    models = pytest.importorskip("rlinf.models")
    logging_mod = pytest.importorskip("rlinf.utils.logging")
    monkeypatch.setattr(logging_mod, "get_logger", lambda: None)

    models.log_lora_census(_wrap(_Policy(n_layers=1), XR1_LORA_REGEX))


def test_adapters_are_built_in_the_base_dtype():
    """Why ``get_model`` passes ``autocast_adapter_dtype=False``.

    FSDP wraps every LoRA leaf and casts its compute parameter to
    ``mixed_precision.param_dtype``, but peft reads the adapter dtype off the
    wrapper and casts the layer input to match. fp32 adapters therefore feed an
    fp32 activation into a bf16 matmul and the forward dies with "expected mat1
    and mat2 to have the same dtype". Equal dtypes on both sides avoid it.
    """
    policy = _Policy(n_layers=1).to(torch.bfloat16)
    policy.freeze_backbone()
    wrapped = _wrap(policy, XR1_LORA_REGEX)

    dtypes = {p.dtype for n, p in wrapped.named_parameters() if "lora_" in n}
    assert dtypes == {torch.bfloat16}, dtypes


def test_peft_default_would_upcast_the_adapters():
    """Pin the trap itself, so a peft upgrade that changes it is visible."""
    policy = _Policy(n_layers=1).to(torch.bfloat16)
    policy.freeze_backbone()
    wrapped = _wrap(policy, XR1_LORA_REGEX, autocast=True)

    dtypes = {p.dtype for n, p in wrapped.named_parameters() if "lora_" in n}
    assert dtypes == {torch.float32}, dtypes
