Installation
============

Evaluation shares the same embodied environment installation flow as training. From the repository root:

.. code-block:: bash

   bash requirements/install.sh embodied --model <model> --env <env>
   source .venv/bin/activate

Choose ``<model>`` and ``<env>`` to match your target benchmark:

.. list-table::
   :header-rows: 1
   :widths: 22 28 50

   * - Benchmark
     - Recommended ``--model``
     - Recommended ``--env``
   * - LIBERO
     - ``openpi`` / ``openvla-oft`` / ``starvla`` / ``dreamzero`` / ``molmoact2``
     - ``maniskill_libero`` or ``libero``
   * - RoboTwin
     - ``openvla-oft`` / ``openpi`` / ``lingbotvla``
     - ``robotwin``
   * - RoboCasa
     - ``xr1`` / ``openpi``
     - ``robocasa``
   * - BEHAVIOR-1K
     - ``openpi``
     - ``behavior``
   * - ManiSkill OOD
     - ``openvla-oft``
     - ``maniskill_libero``
   * - RealWorld
     - ``openpi`` / ``dreamzero``
     - ``franka``
   * - PolaRiS
     - ``openpi``
     - ``polaris``

For more installation options, see :doc:`../../start/installation`.

Benchmark-Specific Environment Variables
----------------------------------------

Configure the following when your benchmark requires them:

**RoboTwin**

.. code-block:: bash

   export ROBOTWIN_PATH=/path/to/RoboTwin
   export ROBOT_PLATFORM=ALOHA

**RoboCasa**

``ROBOCASA_PATH`` is optional and only lets the installer reuse an existing RoboCasa clone instead
of cloning into the venv. Rendering, however, has to be set explicitly: ``run_eval.sh`` defaults
``MUJOCO_GL`` **and** ``PYOPENGL_PLATFORM`` to ``osmesa`` independently, and robosuite refuses to
start when only one of the two is switched, so export both to render on the GPU.

.. code-block:: bash

   export ROBOCASA_PATH=/path/to/robocasa   # optional
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl

**BEHAVIOR-1K**

Set ``OMNIGIBSON_DATA_PATH`` and related OmniGibson paths. See :doc:`../../examples/embodied/behavior`.

**DreamZero**

.. code-block:: bash

   export DREAMZERO_PATH=/path/to/DreamZero

**PolaRiS**

.. code-block:: bash

   export POLARIS_DATA_PATH=/path/to/dataset/PolaRiS-Hub

``run_eval.sh`` reads these variables automatically and configures ``PYTHONPATH``.

Next Steps
----------

After installation, follow :doc:`quick_tour` for your first evaluation, or see the benchmark guide: :doc:`../guides/index`.
