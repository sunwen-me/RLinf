RoboCasa 评测
=============

RoboCasa 是基于 robosuite（MuJoCo）的厨房级移动操作基准，原子任务包括开关抽屉、旋转旋钮与水龙头、抓取放置物体等。RLinf 在并行子进程中评测 VLA 策略，并汇报 ``eval/success_once`` 与 ``eval/success_at_end``\ 。

相关训练文档：:doc:`../../examples/embodied/xr1`

环境准备
--------

**安装依赖**

.. code-block:: bash

   bash requirements/install.sh embodied --model xr1 --env robocasa
   source .venv/bin/activate

``robocasa`` 组合同样支持 ``--model openpi``\ ，按需替换 ``--model``\ 。

**厨房资产**

RoboCasa 的厨房场景来自单独的资产包（约 5 GB）。安装完成后下载一次即可：

.. code-block:: bash

   python -m robocasa.scripts.download_kitchen_assets

**Docker（可选）**

镜像 ``rlinf/rlinf:agentic-rlinf0.4-robocasa`` 已包含 RoboCasa 依赖。在容器内按模型选择虚拟环境：

- XR-1：\ ``source switch_env xr1``
- OpenPI（π\ :sub:`0`\ / π\ :sub:`0.5`\ ）：\ ``source switch_env openpi``

**渲染**

``run_eval.sh`` 会把 ``MUJOCO_GL`` 与 ``PYOPENGL_PLATFORM`` 都默认设为 ``osmesa``\ （软件渲染）。要使用 GPU 渲染，必须\ **同时**\ 导出这两个变量——robosuite 在两者不一致时会拒绝启动：

.. code-block:: bash

   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

示例配置
--------

位于 ``evaluations/robocasa/``\ ，与训练配方一一对应：

.. list-table::
   :header-rows: 1
   :widths: 44 24 14 18

   * - 配置文件
     - 任务
     - ``max_episode_steps``
     - 模型
   * - ``robocasa_closedrawer_xr1_eval.yaml``
     - ``CloseDrawer``
     - 200
     - XR-1
   * - ``robocasa_opendrawer_xr1_eval.yaml``
     - ``OpenDrawer``
     - 500
     - XR-1
   * - ``robocasa_closedoubledoor_xr1_eval.yaml``
     - ``CloseDoubleDoor``
     - 500
     - XR-1
   * - ``robocasa_turnonstove_xr1_eval.yaml``
     - ``TurnOnStove``
     - 500
     - XR-1
   * - ``robocasa_turnoffsinkfaucet_xr1_eval.yaml``
     - ``TurnOffSinkFaucet``
     - 500
     - XR-1
   * - ``robocasa_turnsinkspout_xr1_eval.yaml``
     - ``TurnSinkSpout``
     - 500
     - XR-1
   * - ``robocasa_coffeesetupmug_xr1_eval.yaml``
     - ``CoffeeSetupMug``
     - 500
     - XR-1
   * - ``robocasa_pnpcabtocounter_xr1_eval.yaml``
     - ``PnPCabToCounter``
     - 500
     - XR-1
   * - ``robocasa_pnpcountertosink_xr1_eval.yaml``
     - ``PnPCounterToSink``
     - 500
     - XR-1
   * - ``robocasa_atomic_suite_xr1_eval.yaml``
     - 全部 8 个 500 步任务
     - 500
     - XR-1

每个时域都取 RoboCasa 为该任务本身设定的值，只有 ``CloseDrawer`` 被 XR-1 配方缩短为 200 步。评测时域与训练时域保持一致，\ ``eval/success_once`` 才能与训练中记录的 ``env/success_once`` 相比较。

OpenPI 在 RoboCasa 上暂无 ``evaluations/robocasa/`` 配置；\ ``run_eval.sh`` 会回退到 ``examples/embodiment/config/`` 下的同名配置，因此可以直接复用训练配方（如 ``robocasa_closedrawer_ppo_openpi``\ ），加上 ``runner.only_eval=True runner.task_type=embodied_eval`` 即可。

完整流程
--------

**第 1 步：激活环境**

.. code-block:: bash

   source .venv/bin/activate
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

**第 2 步：准备模型**

XR-1 的建模代码随检查点一起发布。请使用 RoboCasa 的 SFT 权重
`XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa <https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa>`_\ ——只有该仓库是 HuggingFace 格式：

.. code-block:: bash

   hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa --local-dir ./Xiaomi-Robotics-1-RoboCasa

**第 3 步：修改配置**

复制或编辑目标 YAML，至少设置 ``rollout.model.model_path``\ 。纯评测模式下 env 与 rollout worker 读取的是 ``rollout.model``\ ，因此整个模型配置写在这里而不是 ``actor`` 下。通用 ``env.eval`` 字段见 :doc:`../reference/configuration`\ （:ref:`env-eval-fields`），RoboCasa 特有的协议见下文 :ref:`robocasa-eval-config`。

**第 4 步：启动评测**

.. code-block:: bash

   bash evaluations/run_eval.sh robocasa robocasa_closedrawer_xr1_eval \
     rollout.model.model_path=/path/to/Xiaomi-Robotics-1-RoboCasa

benchmark 参数可以省略——\ ``run_eval.sh`` 会从 ``robocasa_`` 前缀推断出 ``robocasa``\ 。

**第 5 步：查看结果**

终端会打印 ``eval/success_once`` 与 ``eval/success_at_end``\ ；视频写入 ``<runner.logger.log_path>/video/eval``\ 。详见 :doc:`../reference/results`。

.. _robocasa-eval-config:

评测配置
--------

评测协议
~~~~~~~~

RoboCasa 无法通过 reset 选项更换场景，因此 ``RobocasaEnv`` 在构造时就固定了 episode：第 *i* 个并行环境以种子 ``env.eval.seed + i`` 创建，该种子决定它的厨房布局、物体摆放与语言指令。由此有两点：

- **可复现性来自种子，而不是 reset-state 开关。** 对 RoboCasa 来说 ``use_fixed_reset_state_ids`` 与 ``use_ordered_reset_state_ids`` 并未生效，示例配置写上它们只是表明意图。只要 ``total_num_envs`` 与 ``seed`` 相同，重复运行评测的就是同一批 episode，唯一的变量是策略自身的采样。
- **覆盖范围随 ``total_num_envs`` 增长。** 每个并行环境在每个 ``max_episode_steps`` 内贡献一条 episode。要评测更多布局，可以增大 ``total_num_envs`` 或修改 ``env.eval.seed``\ ；当 ``auto_reset: True`` 且 ``max_steps_per_rollout_epoch`` 是 ``max_episode_steps`` 的整数倍时，每个环境会连续跑多条 episode，其布局同样由构造种子确定性地推出。

多任务配置（\ ``robocasa_atomic_suite_xr1_eval``\ ）为每个 rollout group 分配一个任务，且组号跨 env rank 全局编号，因此 ``total_num_envs`` 应保持为 ``group_size × len(task_names)`` 的整数倍；不满足时 RLinf 会打印警告。

成功率指标
~~~~~~~~~~

示例配置使用 ``ignore_terminations: True``\ ，即成功之后 episode 仍会继续运行：

- ``eval/success_once`` —— 至少成功过一次的 episode 比例。这是与训练可比的指标，因为训练时 episode 一旦成功就会终止。
- ``eval/success_at_end`` —— 最后一步是否成功。由于策略在成功后仍在动作，它可能破坏自己的成果（把抽屉重新拉开、把杯子推倒），因此这个数通常更低，不能当作前者的替代。

并行度与主机内存
~~~~~~~~~~~~~~~~

每个 RoboCasa 环境都在独立子进程中加载完整的 MuJoCo 场景，单个进程约占 6--8 GB 主机内存，因此 ``total_num_envs`` 的上限通常由主机内存而非显存决定：115 GB 内存的机器在 16 个环境之前就会耗尽内存。\ ``total_num_envs`` 还必须能被 ``cluster.component_placement`` 中 env rank 的数量整除。

.. code-block:: yaml

   env:
     eval:
       rollout_epoch: 1
       total_num_envs: 16        # 16 个不同的厨房布局，每个跑一条 episode
       max_episode_steps: 500
       max_steps_per_rollout_epoch: 500   # 必须是 num_action_chunks 的整数倍
       auto_reset: True
       ignore_terminations: True
       is_eval: True

观测与动作空间
~~~~~~~~~~~~~~

RoboCasa 自身的默认值是两路 224×224 视角、12 维动作与 25 维状态。模型相关的设置必须与训练配方一致，否则策略会在从未训练过的输入上被打分。XR-1 的设置为：

.. code-block:: yaml

   env:
     eval:
       action_space: 7d          # 3 维位置增量 + 3 维旋转增量 + 夹爪
       state_space: 32d          # 25 维末端执行器块 + 7 个原始机械臂关节角
       image_space: 3views
       include_joint_state: True # state_space: 32d 必需
       init_params:
         camera_heights: 256
         camera_widths: 256

评测 RL 训练得到的检查点
~~~~~~~~~~~~~~~~~~~~~~~~

``rollout.model.model_path`` 仍指向 SFT 权重——它提供建模代码与配置——训练得到的权重再通过 ``runner.ckpt_path`` 叠加：

.. code-block:: bash

   bash evaluations/run_eval.sh robocasa robocasa_closedrawer_xr1_eval \
     rollout.model.model_path=/path/to/Xiaomi-Robotics-1-RoboCasa \
     runner.ckpt_path=/path/to/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt

进阶用法
--------

**调整并行度**

.. code-block:: bash

   bash evaluations/run_eval.sh robocasa robocasa_turnonstove_xr1_eval \
     env.eval.total_num_envs=8 \
     rollout.model.model_path=/path/to/model

**探测更短的时域**

RoboCasa 的成功率随时域缩短不是平滑下降，而是崖式跳变（测量数据见 :doc:`../../examples/embodied/xr1`），因此任何改动后都要重新测量：

.. code-block:: bash

   bash evaluations/run_eval.sh robocasa robocasa_opendrawer_xr1_eval \
     env.eval.max_episode_steps=350 \
     env.eval.max_steps_per_rollout_epoch=350 \
     rollout.model.model_path=/path/to/model

**为其他任务派生配置**

复制 ``evaluations/robocasa/`` 下任意 YAML，把 ``defaults`` 中的条目改为 ``env/robocasa_<task>@env.eval``\ （位于 ``examples/embodiment/config/env/``\ ），并让 ``env.eval`` 中的时域与对应训练配方保持一致。

常见问题
--------

- **资产缺失或场景为空：** 执行一次 ``python -m robocasa.scripts.download_kitchen_assets``\ ，资产包约 5 GB。
- **渲染失败或卡住：** 把 ``MUJOCO_GL`` 与 ``PYOPENGL_PLATFORM`` 导出为\ **同一个**\ 后端（GPU 用 ``egl``\ ，软件渲染用 ``osmesa``\ ）。
- **Ray 杀掉 worker 且没有 Python 堆栈：** 这是主机 OOM killer，请降低 ``env.eval.total_num_envs``\ 。
- **启动校验失败：** ``max_steps_per_rollout_epoch`` 必须是 ``rollout.model.num_action_chunks``\ （XR-1 为 10）的整数倍，且 ``total_num_envs`` 能被 env rank 数整除。
- **``eval/success_at_end`` 远低于 ``eval/success_once``\ ：** 在 ``ignore_terminations: True`` 下属于预期，见上文\ `成功率指标`_\ 。
