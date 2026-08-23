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

在 RoboCasa 的多个移动操作厨房任务上，用 GRPO 或 PPO 微调 XR-1 的动作专家。

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
（``evaluations/robocasa/robocasa_<task>_xr1_eval.yaml``）。``CloseDrawer`` 另外提供一份
PPO 配方 ``robocasa_closedrawer_ppo_xr1.yaml``。``max_episode_steps`` 取
RoboCasa 为该任务本身设定的时域，配方不做改动；只有 ``CloseDrawer`` 从 300 步缩短为 200 步，
因为已发布的权重在 300 步的 episode 中每次都能关上抽屉。

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

时域能否给 GRPO 留出空间取决于权重本身。训练时 episode 一旦成功就会终止，因此一条轨迹的
得分实际上等价于“该 episode 是否在 ``max_episode_steps`` 内成功”——即 ``success_once``，
而不是 ``success_at_end``\ （评测配置使用 ``ignore_terminations: True``，成功后仍会继续运行，
状态可能被破坏）。GRPO 随后会减去组均值：组内全部成功——或全部失败——的优势恒为零，也就
不产生梯度。下表是用上面的评测配置对已发布 SFT 权重做的两次独立测量，每次都是同样 8 个固定
种子 episode（动作专家本身随机采样，8 个 episode 在两次之间最多相差 0.25——请把两列当作区间
而不是精确值）：

.. list-table::
   :header-rows: 1
   :widths: 28 18 20 34

   * - 任务
     - ``max_episode_steps``
     - 两次 ``success_once``
     - 更短时域的探测结果
   * - ``CloseDrawer``
     - 200
     - 1.00 / 0.625
     - 150 → 0.00、100 → 0.00
   * - ``OpenDrawer``
     - 500
     - 1.00 / 0.75
     - 350 → 0.875、200 → 0.375
   * - ``CloseDoubleDoor``
     - 500
     - 1.00 / 0.75
     - 350 → 0.00、200 → 0.00
   * - ``TurnOnStove``
     - 500
     - 0.50 / 0.50
     - 200 → 0.625
   * - ``TurnOffSinkFaucet``
     - 500
     - 0.75 / 0.875
     - 200 → 0.875
   * - ``TurnSinkSpout``
     - 500
     - 0.875 / 0.625
     - 200 → 0.625
   * - ``CoffeeSetupMug``
     - 500
     - 0.75 / 0.50
     - 200 → 0.00
   * - ``PnPCabToCounter``
     - 500
     - 0.625 / 0.875
     - 200 → 0.00
   * - ``PnPCounterToSink``
     - 500
     - 0.25 / 0.25
     - 200 → 0.00

``CloseDrawer`` 是唯一在 RoboCasa 原时域下每个 episode 都成功的任务——300 步与 500 步
均为 1.00，因此配方把它缩短到 200 步，两次测量在该时域下分别为 1.00 与 0.625。``OpenDrawer`` 与 ``CloseDoubleDoor`` 处于区间上端，一组 4 个
episode 全部成功的情况很常见：那次恰好抽到这两个任务的单步 GRPO 运行中，两组都是 4/4，
``advantages_max``、``advantages_mean``、``advantages_min`` 与 ``actor/grad_norm`` 全为 0；
而同样一步在 ``TurnOnStove`` 上得到 ``advantages_max`` 0.87、``actor/grad_norm`` 133.5。
首先可以缩短 ``max_episode_steps``——350 步下 ``OpenDrawer`` 为 0.875；但成功率不是平滑
下降而是崖式跳变：``CloseDoubleDoor`` 从 500 步的 1.00 直接降到 350 步的 0.00，
``CloseDrawer`` 从 200 步的 1.00 降到 150 步的 0.00，而 200 步下咖啡和抓放类任务成功率为 0。
改动任何时域后都要重新测量；若该数值重要，请用多于 8 个 episode。

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
   以保证每个 rollout epoch 都在动作块边界结束。PPO 的配置见
   ``robocasa_closedrawer_ppo_xr1.yaml``，详见 `用 PPO 替代 GRPO`_。

**2. 启动**

.. code:: bash

   export MUJOCO_GL=egl

   # 单任务
   bash examples/embodiment/run_embodiment.sh robocasa_closedrawer_grpo_xr1

   # 在一次训练中覆盖 8 个 500 步任务，每个 rollout group 只跑一个任务
   bash examples/embodiment/run_embodiment.sh robocasa_atomic_suite_grpo_xr1

   # 与 GRPO 配方使用同一任务、同一时域的 PPO
   bash examples/embodiment/run_embodiment.sh robocasa_closedrawer_ppo_xr1

.. note::

   每个 RoboCasa 环境都在独立子进程中加载完整的 MuJoCo 场景，单个进程约占 6--8 GB 主机内存，
   因此 ``total_num_envs`` 的上限通常由主机内存而非显存决定：115 GB 内存的机器在 16 个环境
   之前就会耗尽内存。配置中的默认值沿用了既有 ``robocasa_closedrawer_ppo_openpi`` 配方的规模，
   请按机器实际内存下调；多任务配方在下调后仍应保持为
   ``algorithm.group_size x len(task_names)`` 的整数倍。

.. note::

   所有 XR-1 配方都设置了 ``runner.save_interval: -1``，即默认不写权重：一份合并后的 XR-1
   state dict 约 23 GB，按默认间隔保存几百步就能把磁盘写满。需要保留中间权重时，把它改成
   正整数步数即可。环境的视频配置同样会往 ``<runner.logger.log_path>/video`` 写文件，磁盘
   紧张时请关闭 ``save_video``。

用 PPO 替代 GRPO
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``robocasa_closedrawer_ppo_xr1.yaml`` 与 GRPO 配方使用同一任务、同一时域、同一套观测与
动作空间，区别只在优势的估计方式：

.. code:: yaml

   algorithm:
     group_size: 1            # PPO 用价值头而不是同组轨迹作为基线
     adv_type: gae
     loss_type: actor_critic
     gamma: 0.99              # 稀疏的终止奖励，折扣方式沿用 OpenPI 配方
     gae_lambda: 0.95

   actor:
     model:
       add_value_head: True   # loss_type: actor_critic 的必要条件
       xr1:
         value_after_vlm: False    # critic 读取 DiT 后缀的均值池化结果
         detach_critic_input: True # 价值损失不会回传到动作专家
     optim:
       value_lr: 1.0e-4
       critic_warmup_steps: 0      # 调大可先单独训练新初始化的 critic，再让策略跟随其优势

价值头是挂在 DiT 后缀上的 MLP，并不包含在 SFT 权重里，因此它从随机初始化开始，最初几步的
预测没有意义。这正是它与 GRPO 的取舍：在权重已经能稳定完成的任务上（同组每条轨迹得分相同、
GRPO 优势恒为 0），PPO 仍然有可用的梯度；代价是必须先从零学出一个 critic，早期 ``value_lr``、
``critic_warmup_steps`` 与 ``value_clip`` 都会明显影响结果。

在单卡 A800 上以探测规模实测（``env.train.total_num_envs=8``\ 、
``env.train.rollout_epoch=1``\ 、``actor.global_batch_size=16``\ 、
``algorithm.update_epoch=1``\ ，即一个训练步恰好是 10 个优化器步），起点为 SFT 权重。
在 ``critic_warmup_steps: 0`` 下，第一步就是一次真实的 PPO 更新：``advantages_max`` 2.561、
``advantages_min`` -2.621（均值为 0），``actor/grad_norm`` 109.0、``actor/approx_kl`` 0.286、
``actor/clip_fraction`` 0.190、``critic/value_loss`` 0.069，该批 ``env/success_once`` 为 0.875。
把 ``critic_warmup_steps`` 改成 10（即整整一个训练步只训 critic）后，行为与设计一致：
第 1 步的 ``actor/lr`` 与 ``actor/policy_loss`` 均为 0，``critic/value_loss`` 降到 0.226；
第 2 步才是第一次真正的策略更新，且更温和：``actor/grad_norm`` 61.2、``actor/approx_kl`` 0.224、
``critic/value_loss`` 0.063。两次运行采到的 episode 并不相同，8 个 episode 也谈不上受控对比 ——
请把它当作方向而不是测量值。配方仍保持 ``critic_warmup_steps: 0``\ ，与仓库中其他 PPO 配方一致。

.. note::

   ``critic_warmup_steps`` 计的是\ **优化器**\ 步，而不是训练步::

      每训练步的样本数 = total_num_envs x rollout_epoch
                        x max_steps_per_rollout_epoch / num_action_chunks
      优化器步数       = 样本数 / global_batch_size x update_epoch

   按配方给出的默认值，每个训练步是 40 个优化器步（上面的探测规模是 10 个），
   因此小于一个训练步的 warmup 会在训练步中途结束。

.. note::

   当一批数据接近饱和时，``critic/explained_variance`` 不可用：8 个 episode 全部成功时
   returns 只分布在 0.920--1.079 之间，该指标读出 -88.9，而 ``critic/value_loss`` 其实
   仍在下降。这种情况下请看 ``critic/value_loss``\ ，并把 ``critic/value_clip_ratio``
   理解为 critic 相对 rollout 时刻移动了多少。

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
