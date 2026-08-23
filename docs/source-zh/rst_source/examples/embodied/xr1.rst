基于 Xiaomi-Robotics-1（XR-1）的强化学习训练
================================================

.. TODO: 待 RLinf/misc 提供 pic/xr1.png 架构图后替换。

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robocasa.jpeg
   :align: center
   :width: 90%

   XR-1 在 RoboCasa 厨房基准上训练（图片来源：`RoboCasa <https://robocasa.ai/>`__）。

`Xiaomi-Robotics-1 <https://github.com/XiaomiRobotics/Xiaomi-Robotics-1>`__（XR-1）是一个
机器人基础模型，它以 Mixture-of-Transformers 的方式将 **Qwen3-VL** 骨干网络与
**Diffusion Transformer** 动作专家耦合：DiT 与 VLM 层数一致、复用其 KV 缓存，并通过
rectified-flow 头解码动作块。RLinf **原生**\ 接入该模型——通过 ``trust_remote_code``
将 HuggingFace 检查点直接加载到 RLinf 自身的内存空间——并在 RoboCasa 的原子厨房任务上
使用 GRPO 进行微调。

概览
----------------------------------------

在 RoboCasa 的多个移动操作厨房任务上，用 GRPO 微调 XR-1 的动作专家。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 环境
      :text-align: center

      RoboCasa

   .. grid-item-card:: 算法
      :text-align: center

      GRPO · PPO

   .. grid-item-card:: 任务
      :text-align: center

      9 个原子任务

   .. grid-item-card:: 硬件
      :text-align: center

      1 节点 · 8 GPUs

| **你将完成：** 安装 → 下载厨房资产 + XR-1 检查点 → 启动 ``run_embodiment.sh`` → 观察 ``env/success_once``。
| **前置条件：** :doc:`安装 </rst_source/start/installation>` · RoboCasa 厨房资产 · ``Xiaomi-Robotics-1-RoboCasa`` 检查点（步骤见下）。

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

每个任务都以三件套的形式提供：与模型无关的环境配置
（``examples/embodiment/config/env/robocasa_<task>.yaml``）、GRPO 训练配置，以及独立评测配置
（``evaluations/robocasa/robocasa_<task>_xr1_eval.yaml``）。``max_episode_steps`` 按任务单独
选取：要短到让每个 rollout group 中仍有一部分 episode 未完成——GRPO 的基线是组均值，
全成功或全失败的组优势恒为零。

.. list-table::
   :header-rows: 1
   :widths: 22 32 12 34

   * - 任务
     - 配置
     - ``max_episode_steps``
     - 说明
   * - ``CloseDrawer``
     - ``robocasa_closedrawer_grpo_xr1``
     - 200
     - 使用 PandaOmron 移动机械臂关闭厨房抽屉。
   * - ``OpenDrawer``
     - ``robocasa_opendrawer_grpo_xr1``
     - 500
     - 使用 PandaOmron 移动机械臂打开厨房抽屉。
   * - ``CloseDoubleDoor``
     - ``robocasa_closedoubledoor_grpo_xr1``
     - 500
     - 关闭双开柜门。
   * - ``TurnOnStove``
     - ``robocasa_turnonstove_grpo_xr1``
     - 500
     - 打开指定的炉灶旋钮。
   * - ``TurnOffSinkFaucet``
     - ``robocasa_turnoffsinkfaucet_grpo_xr1``
     - 500
     - 关闭水槽水龙头。
   * - ``TurnSinkSpout``
     - ``robocasa_turnsinkspout_grpo_xr1``
     - 500
     - 将水槽出水口转到指定一侧。
   * - ``CoffeeSetupMug``
     - ``robocasa_coffeesetupmug_grpo_xr1``
     - 500
     - 把杯子放到咖啡机出水口下方。
   * - ``PnPCabToCounter``
     - ``robocasa_pnpcabtocounter_grpo_xr1``
     - 500
     - 从柜中拿起物体放到台面上。
   * - ``PnPCounterToSink``
     - ``robocasa_pnpcountertosink_grpo_xr1``
     - 500
     - 从台面拿起物体放入水槽。
   * - 任务套件
     - ``robocasa_atomic_suite_grpo_xr1``
     - 500
     - 在全部 8 个任务上做多任务 GRPO：每个 rollout group 只跑一个任务。

.. note::

   当 ``task_names`` 含多个任务时，任务索引每 ``algorithm.group_size`` 个环境前进一次，
   而不是每个环境前进一次，因此组相对基线不会把两个不同任务平均在一起。为了让各任务出现
   次数均衡，请将 ``env.train.total_num_envs`` 设为
   ``algorithm.group_size × len(task_names)`` 的整数倍——否则 RLinf 会打印警告。RoboCasa
   无法在 reset 时更换场景，因此同一组内的 episode 是同一任务的不同厨房布局。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

XR-1 需要与其官方 RoboCasa 评测一致的接口，因此该配方覆盖了 RoboCasa 的默认设置：
三路 256×256 相机视角、7 维末端执行器动作空间，以及末尾包含原始机械臂关节角的状态向量。

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - 字段
     - 规格
   * - 观测
     - 三路 256×256 RGB 视角（``robot0_agentview_left``、``robot0_agentview_right``、
       ``robot0_eye_in_hand``），按该顺序送入 VLM；以及由 7 个机械臂关节角和第一个夹爪
       关节位置构成的 8 维状态。
   * - 动作
     - 7 维末端执行器控制（3 维位置增量、3 维旋转增量、夹爪）。RLinf 使用
       ``ROBOCASA_DEFAULT_ACTION`` 将其补齐为 RoboCasa 的 12 维动作，使移动底座保持固定。
   * - 奖励
     - 稀疏任务完成奖励。
   * - 提示词
     - RoboCasa 任务生成的自然语言指令。

接口约定
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``rlinf/models/embodiment/xr1/`` 中的适配器读取 batch-first 的 ``env_obs`` 字典
（第 0 维为批大小 ``B``）：

* ``main_images`` → ``robot0_agentview_left``，``torch.uint8``，形状 ``[B, H, W, 3]``。
* ``extra_view_images`` → ``robot0_agentview_right``，dtype 与形状同上。
* ``wrist_images`` → ``robot0_eye_in_hand``，dtype 与形状同上。
* ``states``：``torch.float32``，形状 ``[B, D_state]``；``actor.model.xr1.state_indices``
  从中选出 XR-1 使用的 8 项，模型 60 维状态槽的其余部分按上游评测客户端的做法补零。
* ``task_descriptions``：长度为 ``B`` 的 ``list[str]``。

每次前向都用 ``num_steps`` 个 Euler 步去噪出 ``num_action_chunks`` 个动作。
在 RL 中，确定性的 flow 采样器被视作 SDE，使每个去噪步都能给出高斯对数概率；
``noise_method`` 选择具体变体（``flow_sde``、``flow_cps``、``flow_noise``），
``noise_level`` 控制其尺度。

安装
----------------------------------------

.. include:: _setup_common.rst

**方式一：Docker 镜像** —— 镜像标签 ``agentic-rlinf0.4-robocasa``：

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-robocasa
      # 中国大陆镜像：docker.1ms.run/rlinf/rlinf:agentic-rlinf0.4-robocasa

   # 在容器内切换到 XR-1 虚拟环境：
   source switch_env xr1

**方式二：自定义环境** —— 安装组合 ``--env robocasa``：

.. code:: bash

   # 中国大陆用户可添加 --use-mirror 加速下载。
   bash requirements/install.sh embodied --model xr1 --env robocasa
   source .venv/bin/activate

安装 RoboCasa 后下载厨房资产：

.. code:: bash

   python -m robocasa.scripts.download_kitchen_assets

.. warning::

   RoboCasa 厨房资产约 5 GB。请在启动训练前下载一次。

下载模型
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

XR-1 的建模代码随检查点一起发布，无需额外克隆仓库。请使用 RoboCasa 的 SFT 权重
``Xiaomi-Robotics-1-RoboCasa``\ ：强化学习需要从监督微调后的策略开始，而且只有该仓库
是 HuggingFace 格式（``config.json`` + 分片 ``safetensors`` + ``modeling_mibot.py``）。
``Xiaomi-Robotics-1-5B`` 仓库只包含预训练的原始 ``model_states.pt`` 状态字典，
无法用 ``from_pretrained`` 加载：

.. code:: bash

   # 方法 1：git clone
   git lfs install
   git clone https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa

   # 方法 2：huggingface-hub（中国大陆可设置 HF_ENDPOINT=https://hf-mirror.com）
   uv pip install huggingface-hub
   hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa --local-dir ./Xiaomi-Robotics-1-RoboCasa

下载完成后，将 ``actor.model.model_path`` 指向该检查点。配方中
``rollout.model.model_path`` 由它插值得到，因此只需设置一处：

.. code:: yaml

   actor:
     model:
       model_path: /path/to/Xiaomi-Robotics-1-RoboCasa

运行
----------------------------------------

**1. 配置**

上表中的每个配方都是
``examples/embodiment/config/robocasa_<task>_grpo_xr1.yaml``，其任务列表与时域来自对应的
``env/robocasa_<task>.yaml``；模型默认值位于
``examples/embodiment/config/model/xr1.yaml``。将路径指向你的下载目录，并保持环境空间
与检查点一致：

.. code:: yaml

   env:
     train:
       action_space: 7d          # 3 维位置增量 + 3 维旋转增量 + 夹爪
       state_space: 32d          # 25 维末端执行器块 + 7 个原始机械臂关节角
       image_space: 3views
       include_joint_state: True # state_space: 32d 必需
       init_params:
         camera_heights: 256
         camera_widths: 256

   actor:
     model:
       model_path: "/path/to/Xiaomi-Robotics-1-RoboCasa"
       num_action_chunks: 10     # 由检查点的动作时域固定
       num_steps: 5              # 每个动作块的 Euler 步数
       action_dim: 7
       add_value_head: False     # GRPO 不需要 critic
       rl_trainable_scope: "action_expert"   # 设为 "all" 可同时微调 VLM
       policy_setup: ${env.train.action_space}
       xr1:
         robot_type: "robocasa_mg"
         image_size: 256
         state_indices: [25, 26, 27, 28, 29, 30, 31, 7]
         noise_method: "flow_sde"
         noise_level: 0.5

.. note::

   ``max_steps_per_rollout_epoch`` 必须是 ``num_action_chunks``\ （10）的整数倍，
   以保证每个 rollout epoch 都在动作块边界结束。若要用 PPO 替代 GRPO，请设置
   ``algorithm.adv_type: gae``、``algorithm.loss_type: actor_critic`` 和
   ``actor.model.add_value_head: True``。

**2. 启动**

.. code:: bash

   export MUJOCO_GL=egl

   # 单任务
   bash examples/embodiment/run_embodiment.sh robocasa_closedrawer_grpo_xr1

   # 在一次训练中覆盖 8 个 500 步任务，每个 rollout group 只跑一个任务
   bash examples/embodiment/run_embodiment.sh robocasa_atomic_suite_grpo_xr1

评测
----------------------------------------

每个训练配方都有对应的
``evaluations/robocasa/robocasa_<task>_xr1_eval.yaml``，用于在不训练的情况下评测某个
checkpoint。它沿用训练配方的观测与动作空间、任务列表与时域，但固定了初始状态，因此不同次
运行、不同 checkpoint 都在同一批 episode 上打分：

.. code:: bash

   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   bash evaluations/run_eval.sh robocasa_closedrawer_xr1_eval \
      rollout.model.model_path=/path/to/Xiaomi-Robotics-1-RoboCasa

纯评测模式下，env 与 rollout worker 读取的是 ``rollout.model``，因此模型配置写在这里而不是
``actor`` 下。要评测 RL 训练得到的策略，请保持 ``rollout.model.model_path`` 指向 SFT 权重，
再加载训练器保存的 state dict：

.. code:: bash

   bash evaluations/run_eval.sh robocasa_closedrawer_xr1_eval \
      rollout.model.model_path=/path/to/Xiaomi-Robotics-1-RoboCasa \
      runner.ckpt_path=/path/to/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt

成功率记录在 ``eval/success_once`` 与 ``eval/success_at_end``，视频写入
``<runner.logger.log_path>/video/eval``。

.. note::

   ``run_eval.sh`` 会把 ``MUJOCO_GL`` **和** ``PYOPENGL_PLATFORM`` 都默认设为 ``osmesa``，
   而 robosuite 在两者不一致时会拒绝启动 —— 如上同时导出这两个变量，才能用 GPU 渲染。

可视化与结果
----------------------------------------

在 RLinf 仓库根目录启动 TensorBoard：

.. code:: bash

   tensorboard --logdir ../results --port 6006

关注 **``env/success_once``** 以查看任务成功率。全部日志指标见
:doc:`训练指标 <../../reference/metrics>`。

视频通过环境的 video 配置保存：

.. code:: yaml

   video_cfg:
     save_video: True
     video_base_dir: ${runner.logger.log_path}/video/eval
