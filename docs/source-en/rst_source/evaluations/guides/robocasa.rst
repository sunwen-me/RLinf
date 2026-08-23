RoboCasa Evaluation
===================

RoboCasa is a kitchen-scale mobile-manipulation benchmark built on robosuite (MuJoCo), with atomic tasks such as opening and closing drawers, turning knobs and faucets, and picking and placing objects. RLinf evaluates VLA policies on RoboCasa in parallel subprocesses and reports ``eval/success_once`` and ``eval/success_at_end``.

Related training doc: :doc:`../../examples/embodied/xr1`

Environment Setup
-----------------

**Install dependencies**

.. code-block:: bash

   bash requirements/install.sh embodied --model xr1 --env robocasa
   source .venv/bin/activate

The ``robocasa`` bundle is also available with ``--model openpi``; replace ``--model`` accordingly.

**Kitchen assets**

RoboCasa loads its kitchen scenes from a separate asset pack (about 5 GB). Download it once, after installing:

.. code-block:: bash

   python -m robocasa.scripts.download_kitchen_assets

**Docker (optional)**

The image ``rlinf/rlinf:agentic-rlinf0.4-robocasa`` ships the RoboCasa dependencies. Inside the container, pick the virtual environment for the model:

- XR-1: ``source switch_env xr1``
- OpenPI (π\ :sub:`0`\ / π\ :sub:`0.5`\ ): ``source switch_env openpi``

**Rendering**

``run_eval.sh`` defaults both ``MUJOCO_GL`` and ``PYOPENGL_PLATFORM`` to ``osmesa`` (software rendering). To render on the GPU, export **both** variables — robosuite refuses to start when only one of the two is switched:

.. code-block:: bash

   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

Example Configs
---------------

Available under ``evaluations/robocasa/``, one per training recipe:

.. list-table::
   :header-rows: 1
   :widths: 44 24 14 18

   * - Config file
     - Task
     - ``max_episode_steps``
     - Model
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
     - all 8 500-step tasks
     - 500
     - XR-1

Each horizon is the one RoboCasa itself allows for the task, except ``CloseDrawer``, which the XR-1 recipes shorten to 200 steps. Keeping the eval horizon equal to the training horizon is what makes ``eval/success_once`` comparable with the ``env/success_once`` logged during training.

For OpenPI on RoboCasa there is no config under ``evaluations/robocasa/`` yet; ``run_eval.sh`` falls back to the same config name under ``examples/embodiment/config/``, so a training recipe such as ``robocasa_closedrawer_ppo_openpi`` can be reused with ``runner.only_eval=True runner.task_type=embodied_eval``.

End-to-End Workflow
-------------------

**Step 1: Activate the environment**

.. code-block:: bash

   source .venv/bin/activate
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

**Step 2: Prepare the model**

XR-1 ships its modelling code inside the checkpoint. Use the RoboCasa SFT release `XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa <https://huggingface.co/XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa>`_ — it is the one packaged in HuggingFace format:

.. code-block:: bash

   hf download XiaomiRobotics/Xiaomi-Robotics-1-RoboCasa --local-dir ./Xiaomi-Robotics-1-RoboCasa

**Step 3: Edit the config**

Copy or edit the target YAML and set at least ``rollout.model.model_path``. In eval-only mode the env and rollout workers read ``rollout.model``, so the whole model block lives there instead of under ``actor``. See :doc:`../reference/configuration` (:ref:`env-eval-fields`) for the generic ``env.eval`` fields and :ref:`robocasa-eval-config` below for the RoboCasa protocol.

**Step 4: Launch evaluation**

.. code-block:: bash

   bash evaluations/run_eval.sh robocasa robocasa_closedrawer_xr1_eval \
     rollout.model.model_path=/path/to/Xiaomi-Robotics-1-RoboCasa

The benchmark argument can be omitted — ``run_eval.sh`` infers ``robocasa`` from the ``robocasa_`` prefix.

**Step 5: Check results**

The terminal prints ``eval/success_once`` and ``eval/success_at_end``; videos go to ``<runner.logger.log_path>/video/eval``. See :doc:`../reference/results`.

.. _robocasa-eval-config:

Evaluation Configuration
------------------------

Evaluation Protocol
~~~~~~~~~~~~~~~~~~~

RoboCasa cannot re-seed a scene through reset options, so ``RobocasaEnv`` fixes the episodes at construction time instead: parallel environment *i* is built with seed ``env.eval.seed + i``, which determines its kitchen layout, object placement, and language instruction. Two consequences:

- **Reproducibility comes from the seed, not from the reset-state flags.** ``use_fixed_reset_state_ids`` and ``use_ordered_reset_state_ids`` are inert for RoboCasa — the example configs set them for intent only. Repeated runs at the same ``total_num_envs`` and ``seed`` score the same episodes; the only variation left is the policy's own sampling.
- **Coverage scales with ``total_num_envs``.** Each parallel env contributes one episode per ``max_episode_steps``. To score more distinct layouts, raise ``total_num_envs`` or change ``env.eval.seed``; with ``auto_reset: True`` and a ``max_steps_per_rollout_epoch`` that is a multiple of ``max_episode_steps``, each env runs several consecutive episodes, whose layouts follow deterministically from its construction seed.

Multi-task configs (``robocasa_atomic_suite_xr1_eval``) assign one task per rollout group, numbered across env ranks, so ``total_num_envs`` should stay a multiple of ``group_size × len(task_names)``; RLinf warns when it is not.

Success Metrics
~~~~~~~~~~~~~~~

The example configs run with ``ignore_terminations: True``, so an episode keeps running after the task succeeds:

- ``eval/success_once`` — the fraction of episodes that succeeded at least once. This is the metric to compare against training, where an episode terminates on success.
- ``eval/success_at_end`` — success at the final step. Because the policy keeps acting after success, it can undo its own result (re-open a drawer, knock a mug over), so this number is usually the lower of the two and is not a drop-in replacement.

Parallelism and Host Memory
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Every RoboCasa environment runs in its own subprocess with a full MuJoCo scene and holds roughly 6--8 GB of host RAM, so ``total_num_envs`` is bounded by system memory long before GPU memory: a 115 GB host runs out well below 16 environments. ``total_num_envs`` must also be divisible by the number of env ranks in ``cluster.component_placement``.

.. code-block:: yaml

   env:
     eval:
       rollout_epoch: 1
       total_num_envs: 16        # 16 distinct kitchen layouts, one episode each
       max_episode_steps: 500
       max_steps_per_rollout_epoch: 500   # must be a multiple of num_action_chunks
       auto_reset: True
       ignore_terminations: True
       is_eval: True

Observation and Action Spaces
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

RoboCasa's own defaults are two 224×224 views, a 12-D action and a 25-D state. Model-specific settings must match the training recipe, or the policy is scored on inputs it was never trained on. For XR-1:

.. code-block:: yaml

   env:
     eval:
       action_space: 7d          # 3-D position delta + 3-D rotation delta + gripper
       state_space: 32d          # 25-D end-effector block + 7 raw arm joint positions
       image_space: 3views
       include_joint_state: True # required by state_space: 32d
       init_params:
         camera_heights: 256
         camera_widths: 256

Scoring an RL Checkpoint
~~~~~~~~~~~~~~~~~~~~~~~~

``rollout.model.model_path`` stays on the SFT release — it provides the modelling code and the config — and the trained weights are layered on top with ``runner.ckpt_path``:

.. code-block:: bash

   bash evaluations/run_eval.sh robocasa robocasa_closedrawer_xr1_eval \
     rollout.model.model_path=/path/to/Xiaomi-Robotics-1-RoboCasa \
     runner.ckpt_path=/path/to/checkpoints/global_step_10/actor/model_state_dict/full_weights.pt

Advanced Usage
--------------

**Adjust parallelism**

.. code-block:: bash

   bash evaluations/run_eval.sh robocasa robocasa_turnonstove_xr1_eval \
     env.eval.total_num_envs=8 \
     rollout.model.model_path=/path/to/model

**Probe a different horizon**

Success rates on RoboCasa fall off a cliff rather than degrading smoothly as the horizon shrinks (see the measured table in :doc:`../../examples/embodied/xr1`), so re-measure after any change:

.. code-block:: bash

   bash evaluations/run_eval.sh robocasa robocasa_opendrawer_xr1_eval \
     env.eval.max_episode_steps=350 \
     env.eval.max_steps_per_rollout_epoch=350 \
     rollout.model.model_path=/path/to/model

**Derive a config for another task**

Copy any YAML from ``evaluations/robocasa/``, point the ``defaults`` entry at ``env/robocasa_<task>@env.eval`` (under ``examples/embodiment/config/env/``), and keep the horizon in ``env.eval`` equal to the one in the matching training recipe.

FAQ
---

- **Missing assets or empty scenes:** run ``python -m robocasa.scripts.download_kitchen_assets`` once; the pack is about 5 GB.
- **Rendering fails or hangs:** export ``MUJOCO_GL`` and ``PYOPENGL_PLATFORM`` to the *same* backend (``egl`` for GPU, ``osmesa`` for software).
- **Ray kills workers with no Python traceback:** that is the host OOM killer — lower ``env.eval.total_num_envs``.
- **Startup validation fails:** ``max_steps_per_rollout_epoch`` must be a multiple of ``rollout.model.num_action_chunks`` (10 for XR-1), and ``total_num_envs`` divisible by the env rank count.
- **``eval/success_at_end`` is far below ``eval/success_once``:** expected with ``ignore_terminations: True`` — see `Success Metrics`_.
