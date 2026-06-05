# SO101 双臂 ManiSkill 环境集成指南

> 本文档记录了将自定义双臂 SO100/SO101 ManiSkill 环境集成到 RLinf 框架的完整过程，
> 供后续开发和 AI Agent 参考。

## 目录

- [1. 概述](#1-概述)
- [2. 文件结构](#2-文件结构)
- [3. 架构设计](#3-架构设计)
- [4. 各文件详解](#4-各文件详解)
- [5. 修改指南](#5-修改指南)
- [6. 测试验证](#6-测试验证)
- [7. 常见问题](#7-常见问题)

---

## 1. 概述

### 场景描述

`DualSO100LemonCupScene-v1`：两个 SO100/SO101 机器人并排摆放（间距 24cm），
桌面上有一个杯子和一个柠檬片，任务是将柠檬片的槽口对准杯沿放置。

### 机器人规格

| 属性 | 值 |
|------|-----|
| 机器人型号 | SO100 / SO101（可混搭） |
| 每臂关节数 | 5（shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll） |
| 每臂夹爪 | 1（gripper） |
| 每臂动作维度 | 6（5 关节 + 1 夹爪） |
| 双臂总动作维度 | 12 |
| URDF 关节命名 | `"1"` - `"6"`（数字命名） |

### 相机配置

| 相机 | 位置 | 分辨率 |
|------|------|--------|
| `ego` | 固定外部视角，覆盖整个工作空间 | 128x128 |
| `wrist_left` | 左臂 wrist link 上 | 128x128 |
| `wrist_right` | 右臂 wrist link 上 | 128x128 |
| `render_camera` | 渲染用（human render） | 512x512 |

---

## 2. 文件结构

```
RLinf/
├── rlinf/envs/maniskill/
│   ├── tasks/
│   │   ├── so101_agent.py                    # SO101 机器人 Agent 定义
│   │   └── dual_so100_lemon_cup_scene.py     # 双臂场景 Task 定义
│   └── assets/robots/so100/
│       ├── so101_fixed.urdf                  # SO101 URDF（已去除 transmission）
│       ├── so100.urdf                        # SO100 URDF
│       ├── so101.urdf                        # SO101 URDF（原始版）
│       └── so101_assets/                     # STL 模型文件（14 个）
│           ├── base_motor_holder_so101_v1.stl
│           ├── base_so101_v2.stl
│           ├── ... (共 14 个 STL 文件)
│
├── rlinf/envs/action_utils.py                # [已修改] 添加 so101/so100 动作分支
├── rlinf/config.py                           # [已修改] 添加 so101/so100 控制模式映射
│
├── examples/embodiment/config/env/
│   └── maniskill_dual_so101.yaml             # 环境配置 YAML
│
└── maniskill_so101_env/                      # [源文件] 原始开发目录（不参与运行时加载）
    ├── examples/custom_tasks/
    │   ├── so101_agent.py                    # 原始 agent
    │   └── dual_so100_lemon_cup_scene.py     # 原始 task
    └── mani_skill/assets/robots/so100/       # 原始 URDF 资源
```

**关键路径关系**：
- `so101_agent.py` 中 URDF 路径使用相对路径：`../assets/robots/so100/so101_fixed.urdf`
- 即 `rlinf/envs/maniskill/tasks/../assets/robots/so100/so101_fixed.urdf`
- 实际解析为：`rlinf/envs/maniskill/assets/robots/so100/so101_fixed.urdf`

---

## 3. 架构设计

### ManiSkill 双臂实现模式

ManiSkill 使用 **MultiAgent** 模式实现双臂，而非单一的"双臂 Agent"类：

```
┌─────────────────────────────────────────────────┐
│  DualSO100LemonCupScene (BaseEnv)               │
│                                                 │
│  SUPPORTED_ROBOTS = [                           │
│    ("so100_wristcam", "so100_wristcam"),  # 左,右│
│    ("so101_wristcam", "so101_wristcam"),        │
│    ("so100_wristcam", "so101_wristcam"),  # 混搭 │
│  ]                                              │
│                                                 │
│  agent: MultiAgent                              │
│    ├── agents[0] → left_agent  (SO100/SO101)    │
│    └── agents[1] → right_agent (SO100/SO101)    │
└─────────────────────────────────────────────────┘
```

- **每个臂是独立的 `BaseAgent` 子类**（`SO101` / `SO101WristCam` / `SO100WristCam`）
- **Task 通过 `MultiAgent` 组合两个单臂 agent**
- **运行时通过 `self.agent.agents[0]` / `self.agent.agents[1]` 访问左右臂**

### RLinf 集成链路

```
用户配置 YAML (env_type: maniskill)
    ↓
rlinf/config.py → get_robot_control_mode("so101_wristcam") → "pd_joint_delta_pos"
    ↓
rlinf/envs/__init__.py → get_env_cls() → ManiskillEnv
    ↓
rlinf/envs/maniskill/__init__.py → import_all_tasks() → 自动导入 tasks/ 下所有模块
    ↓
mani_skill.utils.registration → @register_env("DualSO100LemonCupScene-v1")
    ↓
gym.make("DualSO100LemonCupScene-v1", control_mode="pd_joint_delta_pos", ...)
    ↓
rlinf/envs/action_utils.py → prepare_actions_for_maniskill() → so101 分支 → pass through
```

---

## 4. 各文件详解

### 4.1 `so101_agent.py` — 机器人定义

**两个 Agent 类**：

| 类名 | uid | 说明 |
|------|-----|------|
| `SO101` | `"so101"` | SO101 基础 agent，无相机 |
| `SO101WristCam` | `"so101_wristcam"` | 带腕部相机的 SO101 |

**URDF 关节映射**（重要！URDF 用数字命名）：

| URDF joint name | 语义 | Agent 中的变量 |
|-----------------|------|---------------|
| `"1"` | shoulder_pan | `arm_joint_names[0]` |
| `"2"` | shoulder_lift | `arm_joint_names[1]` |
| `"3"` | elbow_flex | `arm_joint_names[2]` |
| `"4"` | wrist_flex | `arm_joint_names[3]` |
| `"5"` | wrist_roll | `arm_joint_names[4]` |
| `"6"` | gripper (jaw) | `gripper_joint_names[0]` |

**URDF Link 映射**：

| Link name | 用途 |
|-----------|------|
| `gripper` | 夹爪固定指（等同 SO100 的 Fixed_Jaw） |
| `jaw` | 夹爪活动指（等同 SO100 的 Moving_Jaw） |
| `Fixed_Jaw_tip` | 固定指尖（手动添加到 URDF） |
| `Moving_Jaw_tip` | 活动指尖（手动添加到 URDF） |
| `handeye_cam` | 腕部相机 link |
| `gripper_camera_mount` | 相机安装座 |

**控制器配置**：

| 控制模式 | 说明 | 用途 |
|---------|------|------|
| `pd_joint_pos` | 绝对关节位置 | 直接控制 |
| `pd_joint_delta_pos` | 增量关节位置 | **RL 训练默认** |
| `pd_joint_target_delta_pos` | 带 target 的增量关节位置 | 有目标的增量控制 |
| `pd_ee_delta_pose` | 末端增量位姿 | VLA 模型 |
| `pd_ee_target_delta_pose` | 带 target 的末端增量位姿 | XR0 等模型 |

### 4.2 `dual_so100_lemon_cup_scene.py` — 场景定义

**环境 ID**: `"DualSO100LemonCupScene-v1"`

**场景物体**：

| 物体 | 类型 | 说明 |
|------|------|------|
| `table` | TableSceneBuilder | 标准桌面 |
| `cup` | 动态 actor（mesh） | 杯子，trimesh 生成的空心圆柱 |
| `lemon_slice` | 动态 actor（mesh） | 柠檬片，带槽口 |
| `target_marker` | kinematic | 绿色球体，标记杯沿目标位置 |
| `ego_camera_marker` | kinematic | 绿色方块，标记 ego 相机位置 |
| `cam_marker_left/right` | kinematic | 青色方块，跟随腕部相机 |

**评估指标 (`evaluate()`)**：

| 指标 | 条件 | 说明 |
|------|------|------|
| `slot_near_rim` | 距离 < 1.5cm | 槽口中心接近杯沿 |
| `height_ok` | 高差 < 8mm | 槽口高度匹配 |
| `orientation_ok` | 角度 < 20° | 槽口朝向正确 |
| `lemon_stable` | 速度 < 0.1 | 柠檬片稳定 |
| `cup_stable` | 位移 < 5cm | 杯子没被碰倒 |
| `success` | 以上全部满足 | 任务成功 |

**奖励函数 (`compute_dense_reward()`)**：

```
reward = 1.0 * reaching_reward        # 右臂 TCP 接近柠檬
       + 2.0 * positioning_reward      # 槽口接近杯沿
       + 1.0 * alignment_reward        # 槽口朝向对齐
       + 1.0 * height_reward           # 高度匹配
       - 0.5 * cup_knock_penalty       # 杯子被碰倒的惩罚
       + 10.0 * success_bonus          # 成功奖励
```

最大奖励 = 15.0（归一化后 = 1.0）

**观测 (`_get_obs_extra()`)**：

state 模式下额外返回：
- `left_tcp`, `right_tcp`：左右臂末端位姿 (7D each)
- `cup_pose`, `lemon_pose`, `target_pose`：物体位姿
- `left_tcp_to_lemon`, `right_tcp_to_lemon`：TCP 到柠檬的向量
- `lemon_to_target`：柠檬到目标的向量
- `slot_center`, `slot_forward`：槽口帧
- `rim_target_pos`, `rim_normal`：杯沿帧

### 4.3 `action_utils.py` — 动作处理

在 `prepare_actions_for_maniskill()` 中添加的分支：

```python
if "so101" in policy or "so100" in policy:
    return raw_chunk_actions  # 直接透传，由 ManiSkill 控制器处理
```

### 4.4 `config.py` — 控制模式映射

在 `get_robot_control_mode()` 中添加的分支：

```python
elif "so101" in robot or "so100" in robot:
    return "pd_joint_delta_pos"
```

### 4.5 `maniskill_dual_so101.yaml` — 环境配置

关键配置项：

| 配置项 | 值 | 说明 |
|--------|-----|------|
| `env_type` | `maniskill` | 使用 ManiSkill 环境类型 |
| `wrap_obs_mode` | `raw` | 使用 task 自定义的观测 |
| `reward_mode` | `raw` | 使用环境自带的奖励 |
| `auto_reset` | `True` | episode 结束自动重置 |
| `init_params.id` | `DualSO100LemonCupScene-v1` | 环境注册 ID |
| `init_params.obs_mode` | `state` | 观测模式（可选 `rgb`） |
| `init_params.control_mode` | `null` | 由 config.py 自动填入 |
| `init_params.sim_backend` | `physx_cuda` | GPU 仿真（调试可用 `physx_cpu`） |
| `sim_config.sim_freq` | `500` | 仿真频率 |
| `sim_config.control_freq` | `5` | 控制频率 |

---

## 5. 修改指南

### 5.1 修改场景逻辑（物体、奖励、评估）

**直接编辑**：`rlinf/envs/maniskill/tasks/dual_so100_lemon_cup_scene.py`

常见修改点：
- `_load_scene()` — 添加/修改场景物体
- `_initialize_episode()` — 修改初始状态随机化
- `evaluate()` — 修改成功条件
- `compute_dense_reward()` — 修改奖励函数
- `_get_obs_extra()` — 修改额外观测

### 5.2 修改机器人定义（关节、控制器）

**直接编辑**：`rlinf/envs/maniskill/tasks/so101_agent.py`

常见修改点：
- `arm_joint_names` — 修改关节名（需与 URDF 一致）
- `_controller_configs` — 添加/修改控制器
- `_sensor_configs` — 修改相机配置
- `tcp_pos` / `tcp_pose` — 修改末端执行器定义

### 5.3 修改 URDF / 模型

1. 修改 `maniskill_so101_env/mani_skill/assets/robots/so100/` 下的文件
2. 同步到 RLinf：
   ```bash
   cp -r /root/RLinf/maniskill_so101_env/mani_skill/assets/robots/so100/* \
         /root/RLinf/rlinf/envs/maniskill/assets/robots/so100/
   ```

### 5.4 添加新的控制模式

1. 在 `so101_agent.py` 的 `_controller_configs` 中添加新配置
2. 在 `config.py` 的 `get_robot_control_mode()` 中添加映射：
   ```python
   elif "so101" in robot or "so100" in robot:
       return "你的新模式名"
   ```

### 5.5 添加新的 policy_setup

在 `action_utils.py` 的 `prepare_actions_for_maniskill()` 中添加分支：

```python
if "你的policy名" in policy:
    return raw_chunk_actions  # 或自定义处理逻辑
```

### 5.6 创建新场景（复用同一机器人）

1. 在 `rlinf/envs/maniskill/tasks/` 下新建文件
2. 从 `dual_so100_lemon_cup_scene.py` 复制结构
3. 修改 `@register_env("YourNewScene-v1", ...)` 注册名
4. 修改 `_load_scene()`, `_initialize_episode()`, `evaluate()` 等
5. 创建对应的 YAML 配置文件

---

## 6. 测试验证

### 快速测试（代码层面，无需 GPU）

```bash
cd /root/RLinf
PYTHONPATH=/root/RLinf/maniskill_so101_env:/root/RLinf \
/opt/venv/openpi/bin/python3 -c "
from rlinf.envs.maniskill.tasks.so101_agent import SO101, SO101WristCam
from rlinf.envs.maniskill.tasks.dual_so100_lemon_cup_scene import DualSO100LemonCupScene
print('✅ Import 成功')
"
```

### 完整测试（需要 GPU + ManiSkill）

```bash
cd /root/RLinf
PYTHONPATH=/root/RLinf/maniskill_so101_env:/root/RLinf \
CUDA_VISIBLE_DEVICES=0 \
/opt/venv/openpi/bin/python3 -c "
import gymnasium as gym
from rlinf.envs.maniskill.tasks import dual_so100_lemon_cup_scene

env = gym.make(
    'DualSO100LemonCupScene-v1',
    obs_mode='state',
    control_mode='pd_joint_delta_pos',
    sim_backend='physx_cpu',  # 调试用 CPU；训练用 physx_cuda
    render_mode=None,
    num_envs=1,
)
obs, info = env.reset(seed=42)
print('obs shape:', obs.shape)

for i in range(10):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

info_eval = env.unwrapped.evaluate()
print('evaluate keys:', list(info_eval.keys()))
env.close()
print('✅ 全部通过')
"
```

### RLinf 框架内测试

```bash
cd /root/RLinf
# 使用训练脚本（需要完整 RLinf 环境）
python examples/embodiment/train_embodied_agent.py \
    --config-name maniskill_dual_so101
```

---

## 7. 常见问题

### Q: 为什么是单臂 Agent 而不是双臂 Agent？

ManiSkill 的 MultiAgent 架构设计：每个臂是独立的 `BaseAgent` 子类，Task 通过 `SUPPORTED_ROBOTS` 元组指定左右臂的 agent 类型，运行时由 `MultiAgent` 自动组合。这样可以灵活混搭不同型号的左右臂。

### Q: URDF 关节名是数字 `"1"` - `"6"` 而不是语义名？

是的，原始 SO101 URDF 使用数字命名关节。Agent 文件中的 `arm_joint_names` 必须与 URDF 一致，所以也是 `["1", "2", "3", "4", "5"]`。如果修改了 URDF 的关节名，必须同步修改 Agent。

### Q: `maniskill_so101_env/` 目录和 `rlinf/envs/maniskill/tasks/` 的关系？

- `maniskill_so101_env/` 是开发/源文件目录，**不参与运行时加载**
- `rlinf/envs/maniskill/tasks/` 是 RLinf 实际加载的文件
- 修改代码请改 `tasks/` 下的文件
- 修改 URDF 请改 `maniskill_so101_env/` 然后同步到 `assets/`

### Q: 如何切换 SO100 和 SO101？

在 YAML 配置或 `gym.make()` 中通过 `robot_uids` 参数指定：

```python
# 双 SO100
env = gym.make("DualSO100LemonCupScene-v1", robot_uids=("so100_wristcam", "so100_wristcam"))

# 双 SO101
env = gym.make("DualSO100LemonCupScene-v1", robot_uids=("so101_wristcam", "so101_wristcam"))

# 混搭：左 SO100 + 右 SO101
env = gym.make("DualSO100LemonCupScene-v1", robot_uids=("so100_wristcam", "so101_wristcam"))
```

### Q: 依赖问题（manifold3d, coacd）？

cup 和 lemon 的 mesh 生成需要：
- `manifold3d` — trimesh boolean 运算
- `coacd` — 凸分解（碰撞体生成）

安装：`pip install manifold3d coacd`

### Q: 如何从 state 模式切换到 rgb 模式？

修改 YAML 中 `init_params.obs_mode: "rgb"`，并确保 `_sensor_configs` 中定义了相机。

---

## 附录：XR0 VLA 模型集成

### 动作维度映射

XR0 输出 32D 双臂动作，SO101 需要 12D（每臂 5 关节 + 1 夹爪）。

**XR0 32D 动作空间**：
```
[0:3]   left_ee_pos
[3:6]   left_ee_axis_angle
[6]     left_gripper
[7:13]  left_joint (6 arm joints)
[13]    reserved
[14:17] right_ee_pos
[17:20] right_ee_axis_angle
[20]    right_gripper
[21:27] right_joint (6 arm joints)
[27:32] reserved
```

**SO101 12D 映射**：
```
SO101 left 6D  = [XR0[7], XR0[8], XR0[9], XR0[10], XR0[11], XR0[6]]
                  ───────────── 5 arm joints ─────────────    gripper

SO101 right 6D = [XR0[21], XR0[22], XR0[23], XR0[24], XR0[25], XR0[20]]
                  ───────────── 5 arm joints ─────────────    gripper

丢弃: XR0[12], XR0[13], XR0[26], XR0[27:32]
```

**关键**：XR0 的 `joint[7:13]` 是 6 个臂关节（不含 gripper），gripper 是单独的 `action[6]`。
SO101 的 `pd_joint_delta_pos` 有 6 个 active joints = 5 臂 + 1 夹爪（joint "6"）。

### ActionMapper 架构

为了确保训练信号只流过有效维度，使用 `ActionMapper` 类：

```
rlinf/models/embodiment/xr0/action_mapping.py
```

核心功能：
1. `map_to_env(actions_32d)` → 12D：gather 有效维度
2. `map_to_model(actions_12d)` → 32D：scatter 回 32D（用于 chain replay）
3. `apply_mask(logprobs)` → 零化无效维度的 logprob/entropy

**训练时的 mask 流程**：
```
Rollout:
  model 输出 32D logprobs → apply_mask → 只保留 12D 有效值
  model 输出 32D actions → map_to_env → 12D 给环境

Training (default_forward):
  replay chain → 32D logprobs → apply_mask → 只保留 12D 有效值
  Loss: logprobs.sum(dim=action_dim) 只包含有效维度
```

### 训练配置

模型配置：`examples/embodiment/config/model/xr0_so101.yaml`
训练配置：`examples/embodiment/config/dual_so101_grpo_xr0.yaml`

启动训练：
```bash
python examples/embodiment/train_embodied_agent.py --config-name dual_so101_grpo_xr0
```

### 相关文件

| 文件 | 说明 |
|------|------|
| `rlinf/models/embodiment/xr0/action_mapping.py` | ActionMapper 定义 + 预设 |
| `rlinf/models/embodiment/xr0/xr0_action_model.py` | 模型 wrapper（已集成 mapper） |
| `rlinf/models/embodiment/xr0/__init__.py` | 工厂函数（从 config 加载 mapper） |
| `examples/embodiment/config/model/xr0_so101.yaml` | SO101 模型配置 |
| `examples/embodiment/config/dual_so101_grpo_xr0.yaml` | 完整训练配置 |
