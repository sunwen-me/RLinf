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

Fine-tune XR-1's action expert with GRPO or PPO on RoboCasa mobile-manipulation
kitchen tasks.

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
``CloseDrawer`` also ships a PPO recipe, ``robocasa_closedrawer_ppo_xr1.yaml``.
``max_episode_steps`` is the horizon RoboCasa itself allows for the task. The recipes keep
it unchanged, except for ``CloseDrawer``, which is shortened from 300 to 200 steps because
the released checkpoint closes the drawer in every 300-step episode.

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

Whether a horizon leaves room for GRPO depends on the checkpoint. A training episode ends
as soon as the task succeeds, so a trajectory's score is effectively "did this episode
succeed within ``max_episode_steps``" — ``success_once``, not ``success_at_end``, which the
evaluation configs let drift because they run with ``ignore_terminations: True``. GRPO then
subtracts the group mean, so a group whose episodes all succeed — or all fail — has an
identically zero advantage and contributes no gradient. The released SFT checkpoint scored
with the evaluation configs above, two independent sweeps of the same eight fixed-seed
episodes (the action expert samples stochastically, and eight episodes swing by up to 0.25
between sweeps — read the columns as a range, not a measurement):

.. list-table::
   :header-rows: 1
   :widths: 28 18 20 34

   * - Task
     - ``max_episode_steps``
     - ``success_once``, two sweeps
     - Shorter horizons probed
   * - ``CloseDrawer``
     - 200
     - 1.00 / 0.625
     - 150 → 0.00, 100 → 0.00
   * - ``OpenDrawer``
     - 500
     - 1.00 / 0.75
     - 350 → 0.875, 200 → 0.375
   * - ``CloseDoubleDoor``
     - 500
     - 1.00 / 0.75
     - 350 → 0.00, 200 → 0.00
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

``CloseDrawer`` is the one task the checkpoint solved in every episode at RoboCasa's own
horizon — 1.00 at both 300 and 500 steps — which is why its recipe shortens it to 200,
where the two sweeps read 1.00 and 0.625. ``OpenDrawer`` and
``CloseDoubleDoor`` sit at the top of the range, and a group of four all-successful
episodes is common there: the one-step GRPO run that happened to draw those two tasks
scored 4/4 in both groups and reported ``advantages_max``, ``advantages_mean``,
``advantages_min`` and ``actor/grad_norm`` all exactly 0, while the same step on
``TurnOnStove`` reached ``advantages_max`` 0.87 and ``actor/grad_norm`` 133.5. Shortening
``max_episode_steps`` is the first lever — 350 steps put ``OpenDrawer`` at 0.875 — but
success falls off a cliff rather than degrading smoothly: ``CloseDoubleDoor`` goes from
1.00 to 0.00 between 500 and 350 steps, ``CloseDrawer`` from 1.00 to 0.00 between 200 and
150, and at 200 steps the coffee and pick-and-place tasks reach 0. Re-measure whatever you
change, with more than eight episodes if the value matters.

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
   every rollout epoch ends on a chunk boundary. PPO is configured in
   ``robocasa_closedrawer_ppo_xr1.yaml``; see `PPO instead of GRPO`_.

**2. Launch**

.. code:: bash

   export MUJOCO_GL=egl

   # A single task
   bash examples/embodiment/run_embodiment.sh robocasa_closedrawer_grpo_xr1

   # All 8 500-step tasks in one run, one task per rollout group
   bash examples/embodiment/run_embodiment.sh robocasa_atomic_suite_grpo_xr1

   # PPO on the same task and horizon as the GRPO recipe
   bash examples/embodiment/run_embodiment.sh robocasa_closedrawer_ppo_xr1

.. note::

   Every RoboCasa environment runs in its own subprocess with a full MuJoCo scene and
   holds roughly 6--8 GB of host RAM, so ``total_num_envs`` is bounded by system memory
   long before GPU memory: a 115 GB host runs out well below 16 environments. The shipped
   values follow the scale of the existing ``robocasa_closedrawer_ppo_openpi`` recipe --
   lower them to fit your machine, and for a multi-task recipe keep the reduced value a
   multiple of ``algorithm.group_size x len(task_names)``.

.. note::

   Every XR-1 recipe ships with ``runner.save_interval: -1``, so no weights are written
   unless you ask for them: one consolidated XR-1 state dict is roughly 23 GB, and the
   default ``val_check_interval`` alone would fill a disk within a few hundred steps. Set
   it to a positive number of steps to keep intermediate checkpoints. The env video
   configs also write to ``<runner.logger.log_path>/video``; turn ``save_video`` off when
   space is tight.

RL hyper-parameters
~~~~~~~~~~~~~~~~~~~

XR-1's action head is a flow-matching sampler, so its RL hyper-parameters are taken from
the recipes this repository already ships for that model class
(``libero_spatial_grpo_evo1.yaml``, ``robotwin_*_grpo_lingbotvla.yaml``,
``libero_spatial_ppo_dexbotic_pi0.yaml``) rather than invented separately:

.. code:: yaml

   algorithm:
     logprob_type: token_level   # per-dim ratio (the executed 10x7 dims)
     filter_rewards: True        # drop all-success / all-fail groups
     rewards_lower_bound: 0.1
     rewards_upper_bound: 0.9
     clip_ratio_low: 0.2
     clip_ratio_high: 0.28       # clip-higher
     group_size: 8
     update_epoch: 2

   actor:
     optim:
       lr: 5.0e-6                # same as openpi / gr00t / dexbotic / evo1

``logprob_type`` is the one that matters most. ``chunk_level`` sums
``num_action_chunks x action_dim`` (10x7=70) Gaussian logprobs into a single number, so the
small numerical difference between the rollout and training passes reaches the PPO ratio
amplified by :math:`\sqrt{70}`. Measured on frozen actor weights (``actor/lr=0`` during
``critic_warmup_steps``, where both sides are provably the same weights and the true ratio
is exactly 1), ``actor/ratio_abs`` is 0.108 -- 54% of the 0.2 clip range -- with
``actor/clip_fraction`` between 6.7% and 10.0%. The clip fires on numerical noise before
the policy has moved at all. ``token_level`` makes every dimension its own ratio, which
leaves 0.013 of the same noise.
``tests/unit_tests/test_embodied_logprob_granularity.py`` pins that
:math:`\sqrt{\text{dims}}` relation as a contract test; Evo-1 (14x7 dims) and LingbotVLA
(50 action chunks) pick ``token_level`` for the same reason.

.. note::

   ``entropy_bonus`` has to stay 0: with ``noise_method: "flow_sde"`` the XR-1 action head
   returns an all-zero entropy (see ``get_log_prob_value`` in ``xr1_action_model.py``), the
   same as Evo-1. ``algorithm.kl_beta`` has no effect on the embodied path -- reference
   logprobs are only computed by the reasoning and Megatron actors.

PPO instead of GRPO
~~~~~~~~~~~~~~~~~~~

``robocasa_closedrawer_ppo_xr1.yaml`` trains the same task, horizon and observation spaces
as the GRPO recipe, and differs only in how the advantage is estimated:

.. code:: yaml

   algorithm:
     group_size: 1            # PPO scores a trajectory against the value head, not a group
     adv_type: gae
     loss_type: actor_critic
     gamma: 0.99              # sparse terminal reward, discounted as in the OpenPI recipe
     gae_lambda: 0.95

   actor:
     model:
       add_value_head: True   # required by loss_type: actor_critic
       xr1:
         value_after_vlm: False    # critic reads the mean-pooled DiT suffix
         detach_critic_input: True # the value loss never reaches the action expert
     optim:
       value_lr: 1.0e-4
       critic_warmup_steps: 0      # raise to train the fresh critic before the policy moves

The value head is an MLP on top of the DiT suffix; it is not part of the SFT release, so it
starts from random weights and its first predictions are meaningless. That is the trade-off
against GRPO: PPO keeps a usable gradient on a task the checkpoint already solves --- where
every episode in a group returns the same score and the GRPO advantage is identically
zero --- but it has to learn a critic from scratch first, and ``value_lr``,
``critic_warmup_steps`` and ``value_clip`` all matter early on.

Measured on one A800 at probe scale (``env.train.total_num_envs=8``,
``env.train.rollout_epoch=1``, ``actor.global_batch_size=16``,
``algorithm.update_epoch=1`` -- so one training step is exactly 10 optimizer steps),
starting from the SFT release. With ``critic_warmup_steps: 0`` the very first step is
already a real PPO update: ``advantages_max`` 2.561 and ``advantages_min`` -2.621 around a
zero mean, ``actor/grad_norm`` 109.0, ``actor/approx_kl`` 0.286, ``actor/clip_fraction``
0.190, ``critic/value_loss`` 0.069, on ``env/success_once`` 0.875. Repeating it with
``critic_warmup_steps: 10`` -- one full training step of critic-only updates -- behaves as
designed: step 1 logs ``actor/lr`` 0 and ``actor/policy_loss`` 0 while
``critic/value_loss`` drops to 0.226, and the first real policy update, in step 2, lands
gentler at ``actor/grad_norm`` 61.2, ``actor/approx_kl`` 0.224 and ``critic/value_loss``
0.063. The two runs sampled different episodes, and eight episodes are not a controlled
comparison -- read it as a direction, not a measurement. The recipe keeps
``critic_warmup_steps: 0``, as every other PPO recipe in the repository does.

.. note::

   ``critic_warmup_steps`` counts **optimizer** steps, not training steps::

      samples per training step = total_num_envs x rollout_epoch
                                  x max_steps_per_rollout_epoch / num_action_chunks
      optimizer steps           = samples / global_batch_size x update_epoch

   That is 40 optimizer steps per training step at the shipped values and 10 at the probe
   scale above, so a warmup shorter than one training step ends in the middle of one.

.. note::

   ``critic/explained_variance`` is unusable when a batch is near-saturated: with all eight
   episodes succeeding, the returns spanned only 0.920--1.079, and the metric read -88.9
   while ``critic/value_loss`` was in fact still improving. Watch ``critic/value_loss``
   instead, and read ``critic/value_clip_ratio`` as how far the critic moved since the
   rollout.

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
