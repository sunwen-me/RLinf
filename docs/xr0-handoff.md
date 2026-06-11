# XR0 VLA 接入 RLinf 交接文档

## 项目概述

将小米 XR0 VLA 模型接入 RLinf 框架，支持推理和 RL 训练。

XR0 架构：Qwen3-VL（4B） + DiT（16层） + Rectified Flow（5步去噪）
动作空间：30步 × 32维（双臂：位置、旋转、夹爪、关节角）

## 已完成的工作

### PR 1：模型骨架
- 注册 `model_type: "xr0"` 到 RLinf
- 创建 `rlinf/models/embodiment/xr0/` 包
- stub 模型、config YAML、单元测试

### PR 2：真实推理
- Qwen3-VL processor 接入（图片→PIL→resize→tokenize）
- `predict_action_batch` 完整实现
- 加载真实权重（`AutoModel.from_pretrained`）
- 已验证：LIBERO checkpoint 推理输出 (1, 30, 32)

### PR 3：RL训练支持
- `sample_actions`：逐步去噪，记录chain
- `get_logprob_norm`：计算高斯log概率
- `default_forward`：重放chain，算logprobs/entropy
- 简化版Flow-SDE（固定噪声，单步随机）

### PR 4：Value Head + Flow-SDE + State Padding + FSDP修复
- **Value Head**：用VLM hidden states实现PPO critic（`ValueHead` MLP，masked mean-pooling）
- **Flow-SDE完整版**：`σ=a√(τ/(1-τ))` + 漂移修正 `σ²δ/(2τ)`
- **State Padding**：处理LIBERO 8D state → XR0 32D（`_pad_state`方法）
- **FSDP修复**：修正 `_no_split_modules` 类名（`Qwen3VLTextDecoderLayer`等）
- **Processor加载**：从本地路径加载 + `trust_remote_code`

### PR 5：Pipeline修复（split_with_sizes + action维度）
- **split_with_sizes修复**：Qwen3-VL的 `pixel_values` 形状是 `(total_patches, patch_dim)`，不是 `(B, ...)`。在 `_split_rollout_result` 中新增专用分支，根据 `image_grid_thw` 切分 `pixel_values`。
- **action_env_dim**：配置中已有 `action_env_dim: 7`，模型代码已实现切片 `actions_np[:, :, :self.action_env_dim]`。经验证模型正确返回 `(B, 30, 7)`。

## 关键文件

```
rlinf/models/embodiment/xr0/
├── __init__.py              # get_model()工厂，_StubXR0
├── xr0_action_model.py      # XR0ForRLActionPrediction（核心）
├── utils.py                 # ACTION_DIM, normalize/denormalize, resize_image
└── model/                   # 从xr0_src搬运的模型代码（暂时不用，用AutoModel加载）
    ├── xr0_model.py
    ├── qwen3vl.py
    └── __init__.py

examples/embodiment/config/model/
└── xr0.yaml                 # 模型配置

tests/unit_tests/
└── test_xr0_model_registration.py  # 6个测试
```

## 当前能力

| 功能 | 状态 | 备注 |
|------|------|------|
| 模型注册 | ✅ | `model_type: "xr0"` |
| stub测试 | ✅ | `model_path: "dummy"` |
| 真实推理 | ✅ | CPU可用，GPU需≥24GB |
| predict_action_batch | ✅ | 返回 (B, 30, 7) 动作 + chain |
| action_env_dim切片 | ✅ | 32D → 7D，配置驱动 |
| split_with_sizes修复 | ✅ | pixel_values按patch切分 |
| State Padding | ✅ | LIBERO 8D → XR0 32D |
| Value Head | ✅ | VLM hidden states → PPO critic |
| Flow-SDE完整版 | ✅ | σ=a√(τ/(1-τ)) + 漂移修正 |
| FSDP修复 | ✅ | _no_split_modules类名修正 |
| default_forward | ✅ | 已修复：gradient_checkpointing + torch.no_grad()，KV cache已序列化 |
| 可学习噪声 | ❌ | TODO |
| 文档/CI | ❌ | TODO |

## ~~当前阻塞问题：default_forward past_key_values None~~ ✅ 已修复

### 原问题

训练阶段 actor worker 调用 `default_forward` 重放去噪链时崩溃，`past_key_values` 含 `None` 条目。

### 根因

transformers 4.57 的 `@check_model_inputs` 装饰器，在 `gradient_checkpointing=True` 且 `self.training=True` 时，**静默**将 `use_cache` 强制设为 `False`，导致 VLM forward 返回空 KV cache。

### 修复（commit e637c186）

1. VLM forward 前临时关闭所有子模块的 `gradient_checkpointing`
2. 用 `torch.no_grad()` 跑 VLM forward（梯度只走 DiT）
3. `try/finally` 恢复 `gradient_checkpointing` 状态

### 后续优化

KV cache 已序列化到 `forward_inputs`（`_pack_vlm_kv` / `_unpack_vlm_kv`），训练时 `default_forward` 主路径直接解包复用，不再重跑 VLM，彻底绕过此问题。PPO 训练已验证通过（2026-06-10）。

## 去噪公式说明

`_compute_denoise_mean_std` 中 eval 模式使用 Euler-step 公式 `x_t + v_t * δ`，与 `_fixed.py` / `_patched` 中的端点插值公式 `x0_pred * w0 + x1_pred * w1` 数学上等价（在真实 velocity field 下均为 `x_t - v_true * δ`）。

当前 Euler-step 公式与 checkpoint 的推理循环（`_flow_generate` / `_checkpoint_forward_eval`：`x = x + v * dt`）保持一致，是正确实现。训练时走 `mode="train"` 分支（flow_sde），两边公式完全相同，不存在 train-eval 不一致。

**开发中间文件**（`xr0_action_model.py.bak`、`xr0_action_model_fixed.py`、`xr0_action_model_patched (1).py`）为迭代调试产物，不再维护，可安全删除。

## 参考实现

最重要的参考是 **lingbotvla**：
```
rlinf/models/embodiment/lingbotvla/lingbotvla_action_model.py
```

它实现了完整的flow-matching RL训练：
- `sample_actions`：逐步去噪+chain录制
- `sample_mean_var_val`：三种噪声方法
- `get_logprob_norm`：log概率计算
- `get_value_from_vlm`：VLM做critic
- `default_forward`：chain重放

## 权重下载

```bash
# 用hf-mirror下载LIBERO checkpoint
HF_ENDPOINT=https://hf-mirror.com python download_xr0.py

# 权重位置
/home/sw/models/Xiaomi-Robotics-0-LIBERO/
```

## 环境配置

```bash

pip install -e .
pip install transformers==4.57.1
pip install torchvision
```

注意：transformers版本必须是4.57.1，更高版本不兼容。

## 测试

```bash
conda run -n rlinf python -m pytest tests/unit_tests/test_xr0_model_registration.py -v
```

## 已知问题


1. stub模型的logprobs/entropy是零（正常，stub没有真实去噪）
2. transformers版本锁定4.57.1（XR0自定义代码依赖）

## Git分支

```
feat/xr0-vla  →  https://github.com/sunwen-me/RLinf/tree/feat/xr0-vla
```

18个commit，全部已push。
