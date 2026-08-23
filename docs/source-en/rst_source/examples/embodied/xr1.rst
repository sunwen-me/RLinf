RL on Xiaomi-Robotics-1 (XR-1)
==============================

.. TODO: swap for a pic/xr1.png in RLinf/misc once an architecture figure is available.

.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/robocasa.jpeg
   :align: center
   :width: 90%

   XR-1 is trained on the RoboCasa kitchen benchmark (image: `RoboCasa <https://robocasa.ai/>`__).

`Xiaomi-Robotics-1 <https://github.com/XiaomiRobotics/Xiaomi-Robotics-1>`__ (XR-1) is a
robot foundation model that couples a **Qwen3-VL** backbone to a **Diffusion Transformer**
action expert as a Mixture-of-Transformers: the DiT matches the VLM layer for layer, reuses
its KV cache, and decodes action chunks with a rectified-flow head. RLinf integrates it
**natively** — the HuggingFace checkpoint is loaded in RLinf's own memory space through
``trust_remote_code`` — and GRPO-fine-tunes it on RoboCasa's atomic kitchen tasks.

Overview
--------

GRPO-fine-tune XR-1's action expert on RoboCasa mobile-manipulation kitchen tasks.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Environments
      :text-align: center

      RoboCasa

   .. grid-item-card:: Algorithms
      :text-align: center

      GRPO · PPO

   .. grid-item-card:: Tasks
      :text-align: center

      9 atomic tasks

   .. grid-item-card:: Hardware
      :text-align: center

      1 node · 8 GPUs

| **You'll do:** install → download kitchen assets + the XR-1 checkpoint → launch ``run_embodiment.sh`` → watch ``env/success_once``.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · RoboCasa kitchen assets · the ``Xiaomi-Robotics-1-RoboCasa`` checkpoint (steps below).

Tasks
~~~~~

Every task ships as a triplet: a model-agnostic env config
(``examples/embodiment/config/env/robocasa_<task>.yaml``), a GRPO training config, and a
standalone evaluation config (``evaluations/robocasa/robocasa_<task>_xr1_eval.yaml``).
``max_episode_steps`` is chosen per task: short enough that part of every rollout group
stays unsolved, because GRPO's baseline is the group mean and a group that always
succeeds — or never does — yields a zero advantage.

.. list-table::
   :header-rows: 1
   :widths: 22 32 12 34

   * - Task
     - Config
     - ``max_episode_steps``
     - What it exercises
   * - ``CloseDrawer``
     - ``robocasa_closedrawer_grpo_xr1``
     - 200
     - Close a kitchen drawer with the PandaOmron mobile manipulator.
   * - ``OpenDrawer``
     - ``robocasa_opendrawer_grpo_xr1``
     - 500
     - Open a kitchen drawer with the PandaOmron mobile manipulator.
   * - ``CloseDoubleDoor``
     - ``robocasa_closedoubledoor_grpo_xr1``
     - 500
     - Close both doors of a two-door cabinet.
   * - ``TurnOnStove``
     - ``robocasa_turnonstove_grpo_xr1``
     - 500
     - Turn on the requested stove burner knob.
   * - ``TurnOffSinkFaucet``
     - ``robocasa_turnoffsinkfaucet_grpo_xr1``
     - 500
     - Turn off the sink faucet.
   * - ``TurnSinkSpout``
     - ``robocasa_turnsinkspout_grpo_xr1``
     - 500
     - Swivel the sink spout to the requested side.
   * - ``CoffeeSetupMug``
     - ``robocasa_coffeesetupmug_grpo_xr1``
     - 500
     - Place a mug under the coffee-machine dispenser.
   * - ``PnPCabToCounter``
     - ``robocasa_pnpcabtocounter_grpo_xr1``
     - 500
     - Pick an object from the cabinet and place it on the counter.
   * - ``PnPCounterToSink``
     - ``robocasa_pnpcountertosink_grpo_xr1``
     - 500
     - Pick an object from the counter and place it in the sink.
   * - Task suite
     - ``robocasa_atomic_suite_grpo_xr1``
     - 500
     - Multi-task GRPO over all 8 tasks; every rollout group stays on one task.

.. note::

   When ``task_names`` holds several tasks, the task index advances once per
   ``algorithm.group_size`` environments instead of once per environment, so a
   group-relative baseline never averages two different tasks. Keep
   ``env.train.total_num_envs`` a multiple of
   ``algorithm.group_size × len(task_names)`` for balanced coverage — RLinf logs a
   warning when it is not. RoboCasa cannot re-seed a scene on reset, so the episodes
   inside one group are different kitchen layouts of the same task.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

XR-1 expects the same interface its own RoboCasa evaluation uses, so the recipe overrides
the RoboCasa defaults: three camera views at 256×256, the 7-D end-effector action space,
and a state vector that ends in the raw arm joint positions.

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Field
     - Specification
   * - Observation
     - Three RGB views at 256×256 (``robot0_agentview_left``, ``robot0_agentview_right``,
       ``robot0_eye_in_hand``), fed to the VLM in that order, plus an 8-D state built from
       the 7 arm joint positions and the first gripper joint position.
   * - Action
     - 7-D end-effector control (3-D position delta, 3-D rotation delta, gripper). RLinf
       pads it to RoboCasa's 12-D action with ``ROBOCASA_DEFAULT_ACTION``, which keeps the
       mobile base fixed.
   * - Reward
     - Sparse task-completion reward.
   * - Prompt
     - Natural-language instruction generated by the RoboCasa task.

Interface Conventions
~~~~~~~~~~~~~~~~~~~~~

The adapter in ``rlinf/models/embodiment/xr1/`` reads a batch-first ``env_obs`` dict
(dimension 0 is batch size ``B``):

* ``main_images`` → ``robot0_agentview_left``, ``torch.uint8``, shape ``[B, H, W, 3]``.
* ``extra_view_images`` → ``robot0_agentview_right``, same dtype and shape.
* ``wrist_images`` → ``robot0_eye_in_hand``, same dtype and shape.
* ``states``: ``torch.float32``, shape ``[B, D_state]``; ``actor.model.xr1.state_indices``
  selects the 8 entries XR-1 consumes and the rest of the model's 60-D state slot is
  zero-padded, exactly as in the upstream evaluation client.
* ``task_descriptions``: ``list[str]`` of length ``B``.

Each forward pass denoises a chunk of ``num_action_chunks`` actions with
``num_steps`` Euler steps. For RL, the deterministic flow sampler is treated as an SDE so
every denoising step yields a Gaussian log-probability; ``noise_method`` selects the
variant (``flow_sde``, ``flow_cps``, ``flow_noise``) and ``noise_level`` sets its scale.

Installation
------------

.. include:: _setup_common.rst

**Option 1: Docker image** — image tag ``agentic-rlinf0.4-robocasa``:

.. code:: bash

   docker run -it --rm --gpus all \
      --shm-size 32g \
      --network host \
      --name rlinf \
      -v .:/workspace/RLinf \
      rlinf/rlinf:agentic-rlinf0.4-robocasa
      # Mainland China mirror: docker.1ms.run/rlinf/rlinf:agentic-rlinf0.4-robocasa

   # Inside the container, switch to the XR-1 virtual environment:
   source switch_env xr1

**Option 2: Custom environment** — install bundle ``--env robocasa``:

.. code:: bash

   # Add --use-mirror for faster downloads in mainland China.
   bash requirements/install.sh embodied --model xr1 --env robocasa
   source .venv/bin/activate

Download the kitchen assets after installing RoboCasa:

.. code:: bash

   python -m robocasa.scripts.download_kitchen_assets

.. warning::

   The RoboCasa kitchen assets are about 5 GB. Download them once before launching training.

Download the Model
~~~~~~~~~~~~~~~~~~

XR-1 ships its modelling code inside the checkpoint, so no extra repository is needed.
Use the RoboCasa SFT release ``Xiaomi-Robotics-1-RoboCasa`` -- RL fine-tuning starts from
the supervised policy, and that repository is the one packaged in HuggingFace format
(``config.json`` + sharded ``safetensors`` + ``modeling_mibot.py``). The
``Xiaomi-Robotics-1-5B`` release only holds a raw ``model_states.pt`` pre-training state
dict and cannot be loaded with ``from_pretrained``:

.. code:: bash

   # Method 1: git clone
   git lfs install
   git clone https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa

   # Method 2: huggingface-hub (set HF_ENDPOINT=https://hf-mirror.com in mainland China)
   uv pip install huggingface-hub
   hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa --local-dir ./Xiaomi-Robotics-1-RoboCasa

After downloading, point ``actor.model.model_path`` at the checkpoint. The recipe
interpolates ``rollout.model.model_path`` from it, so there is only one path to set:

.. code:: yaml

   actor:
     model:
       model_path: /path/to/Xiaomi-Robotics-1-RoboCasa

Run It
------

**1. Configuration**

Every recipe in the table above is
``examples/embodiment/config/robocasa_<task>_grpo_xr1.yaml`` and takes its task list and
horizon from the matching ``env/robocasa_<task>.yaml``; the model defaults live
in ``examples/embodiment/config/model/xr1.yaml``. Point the paths at your download and
keep the env spaces aligned with the checkpoint:

.. code:: yaml

   env:
     train:
       action_space: 7d          # 3-D position delta + 3-D rotation delta + gripper
       state_space: 32d          # 25-D end-effector block + 7 raw arm joint positions
       image_space: 3views
       include_joint_state: True # required by state_space: 32d
       init_params:
         camera_heights: 256
         camera_widths: 256

   actor:
     model:
       model_path: "/path/to/Xiaomi-Robotics-1-RoboCasa"
       num_action_chunks: 10     # fixed by the checkpoint's action horizon
       num_steps: 5              # Euler steps per chunk
       action_dim: 7
       add_value_head: False     # GRPO needs no critic
       rl_trainable_scope: "action_expert"   # "all" fine-tunes the VLM too
       policy_setup: ${env.train.action_space}
       xr1:
         robot_type: "robocasa_mg"
         image_size: 256
         state_indices: [25, 26, 27, 28, 29, 30, 31, 7]
         noise_method: "flow_sde"
         noise_level: 0.5

.. note::

   ``max_steps_per_rollout_epoch`` must stay a multiple of ``num_action_chunks`` (10) so
   every rollout epoch ends on a chunk boundary. For PPO instead of GRPO, set
   ``algorithm.adv_type: gae``, ``algorithm.loss_type: actor_critic``, and
   ``actor.model.add_value_head: True``.

**2. Launch**

.. code:: bash

   export MUJOCO_GL=egl

   # A single task
   bash examples/embodiment/run_embodiment.sh robocasa_closedrawer_grpo_xr1

   # All 8 500-step tasks in one run, one task per rollout group
   bash examples/embodiment/run_embodiment.sh robocasa_atomic_suite_grpo_xr1

Evaluation
----------

Every training recipe has a matching
``evaluations/robocasa/robocasa_<task>_xr1_eval.yaml`` that scores a checkpoint without
training. It keeps the observation and action spaces, the task list, and the horizon of the
training recipe but pins the initial states, so every run and every checkpoint is scored on
the same episodes:

.. code:: bash

   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   bash evaluations/run_eval.sh robocasa_closedrawer_xr1_eval \
      rollout.model.model_path=/path/to/Xiaomi-Robotics-1-RoboCasa

In eval-only mode the env and rollout workers read ``rollout.model``, so the model block
lives there instead of under ``actor``. To score a policy produced by RL, leave
``rollout.model.model_path`` on the SFT release and add the state dict the trainer wrote:

.. code:: bash

   bash evaluations/run_eval.sh robocasa_closedrawer_xr1_eval \
      rollout.model.model_path=/path/to/Xiaomi-Robotics-1-RoboCasa \
      runner.ckpt_path=/path/to/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt

The success rate is reported as ``eval/success_once`` and ``eval/success_at_end``; videos
are written to ``<runner.logger.log_path>/video/eval``.

.. note::

   ``run_eval.sh`` defaults ``MUJOCO_GL`` **and** ``PYOPENGL_PLATFORM`` to ``osmesa``, and
   robosuite refuses to start when only one of the two is switched to ``egl`` -- export
   both, as above, to render on the GPU.

Visualization and Results
-------------------------

Launch TensorBoard from the RLinf repo root:

.. code:: bash

   tensorboard --logdir ../results --port 6006

Watch **``env/success_once``** for the task success rate. For every logged metric, see
:doc:`Training metrics <../../reference/metrics>`.

Videos are saved through the env video config:

.. code:: yaml

   video_cfg:
     save_video: True
     video_base_dir: ${runner.logger.log_path}/video/eval
