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
| default_forward | ❌ | **past_key_values有None条目** |
| 可学习噪声 | ❌ | TODO |
| 文档/CI | ❌ | TODO |

## 当前阻塞问题：default_forward past_key_values None

### 症状

训练阶段 actor worker 调用 `default_forward` 重放去噪链时崩溃：

```
File "modeling_mibot.py", line 1644, in forward
    k_cache = repeat_kv(k_cache, self.num_key_value_groups)
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
AttributeError: 'NoneType' object has no attribute 'shape'
```

### 根因分析

`default_forward` 重放流程：
1. 从 `forward_inputs` 恢复 VLM batch（input_ids, pixel_values, image_grid_thw）
2. 调用 `self.xr0_model.vlm(**vlm_batch, use_cache=True)` 获取 KV cache
3. 将 `past_key_values` 传给 `dit_forward` 做去噪

问题出在步骤 2-3：VLM 的 `past_key_values` 中某些层的 key/value 是 `None`，传给 DiT 的 attention 层后 `repeat_kv` 崩溃。

### 可能原因

1. **VLM KV cache 格式不兼容**：Qwen3-VL 的 `past_key_values` 可能使用了 `HybridCache`（transformers 4.57+），其中某些层（如 sliding window 层）的 cache 为 `None`。
2. **DiT 不应使用 VLM 的 KV cache**：DiT 有自己的 attention 层，应该独立构建 KV cache，而不是复用 VLM 的。`sample_actions` 中的 `dit_forward` 传入的 `past_key_values` 来自 VLM，但 DiT 的 attention 层可能期望不同的格式。
3. **pixel_values 缺失或格式错误**：如果 `forward_inputs` 中的 `pixel_values` 经过 split 后格式不对，VLM forward 可能产生不完整的 cache。

### 调试建议

1. 在 `default_forward` 的 `vlm_outputs = self.xr0_model.vlm(**vlm_batch, use_cache=True)` 之后，检查 `past_key_values` 中是否有 `None` 条目：
   ```python
   for i, (k, v) in enumerate(past_key_values):
       if k is None or v is None:
           print(f"Layer {i}: k={k}, v={v}")
   ```

2. 对比 `sample_actions`（rollout时正常工作）和 `default_forward`（训练时崩溃）的 VLM batch 内容，看是否有差异。

3. 检查 `modeling_mibot.py` 中 DiT 的 attention 层如何使用 `past_key_values`，是否需要做格式转换。

4. 如果 VLM KV cache 不能直接传给 DiT，可能需要在 `default_forward` 中重新实现 DiT 的 KV cache 管理，参考 `sample_actions` 中的实现。

### 参考文件

- `modeling_mibot.py:1644` — `repeat_kv` 崩溃点
- `modeling_mibot.py:1697` — DiT attention 调用
- `xr0_action_model.py:820-835` — `default_forward` VLM forward
- `xr0_action_model.py:543-563` — `sample_actions` DiT forward（正常工作）

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

7个commit，全部已push。
