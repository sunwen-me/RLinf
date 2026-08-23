# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os
import sys

import hydra
import torch.multiprocessing as mp
from omegaconf.omegaconf import OmegaConf

from rlinf.config import validate_cfg
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.scheduler import Cluster
from rlinf.utils.logging import get_logger
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

mp.set_start_method("spawn", force=True)


def _drop_script_dir_from_sys_path() -> None:
    """Keep the per-benchmark config directories from shadowing simulator packages.

    Python prepends this script's directory to ``sys.path``, and the config
    directories sitting next to it are named after the simulators they
    configure (``evaluations/robocasa``, ``evaluations/libero``, ...).  Each is
    a PEP 420 namespace portion, which beats a simulator installed in editable
    mode: the editable install is served by a meta-path finder that never runs
    once the path scan has produced a namespace spec, so ``import robocasa``
    would return the YAML directory and no environment would be registered.
    Ray copies the driver's ``sys.path`` into its workers, so the entry has to
    go before any worker is created.
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path[:] = [
        path for path in sys.path if os.path.abspath(path or os.curdir) != script_dir
    ]


@hydra.main(
    version_base="1.1",
    config_path="libero",
    config_name="libero_spatial_starvla_eval",
)
def main(cfg) -> None:
    _drop_script_dir_from_sys_path()
    cfg.runner.task_type = "embodied_eval"
    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    component_placement = HybridComponentPlacement(cfg, cluster)

    # Create rollout worker group. Select the worker by ``rollout_backend``:
    # only ``sglang`` and ``huggingface`` are supported here (vllm is intentionally not wired in);
    rollout_placement = component_placement.get_strategy("rollout")
    rollout_backend = cfg.rollout.get("rollout_backend", "huggingface")
    # Default env worker; RTC on the huggingface path overrides it below.
    env_worker_cls = EnvWorker
    if rollout_backend == "sglang":
        from rlinf.workers.rollout.utils import get_rollout_backend_worker

        rollout_group = (
            get_rollout_backend_worker(cfg)
            .create_group(cfg, component_placement)
            .launch(
                cluster,
                name=cfg.rollout.group_name,
                placement_strategy=rollout_placement,
            )
        )
    elif rollout_backend == "huggingface":
        if cfg.runner.get("rtc", {}).get("enabled", False):
            from rlinf.workers.env.rtc_env_worker import RTCEnvWorker
            from rlinf.workers.rollout.hf.rtc_huggingface_worker import (
                RTCMultiStepRolloutWorker,
            )

            env_worker_cls = RTCEnvWorker
            rollout_worker_cls = RTCMultiStepRolloutWorker
        else:
            rollout_worker_cls = MultiStepRolloutWorker

        # Create rollout worker group
        rollout_placement = component_placement.get_strategy("rollout")
        rollout_group = rollout_worker_cls.create_group(cfg).launch(
            cluster, name=cfg.rollout.group_name, placement_strategy=rollout_placement
        )
    else:
        raise ValueError(f"Unsupported rollout backend: {rollout_backend}")
    # Create env worker group
    env_placement = component_placement.get_strategy("env")
    env_group = env_worker_cls.create_group(cfg).launch(
        cluster, name=cfg.env.group_name, placement_strategy=env_placement
    )

    # launch the sglang server
    if rollout_backend == "sglang":
        from rlinf.workers.rollout.sglang_server import (
            launch_sglang_router_and_server,
        )

        server_group, _ = launch_sglang_router_and_server(
            cfg,
            cluster,
            rollout_hardware_ranks=component_placement.get_hardware_ranks("rollout"),
            router_server_args=cfg.rollout.sglang,
        )
        _server_urls = list(server_group.get_server_url().wait())
        get_logger().info(
            f"[eval] launched {len(_server_urls)} sglang server(s): {_server_urls}"
        )
        rollout_group.set_sglang_server_urls(_server_urls).wait()

    runner = EmbodiedEvalRunner(
        cfg=cfg,
        rollout=rollout_group,
        env=env_group,
    )

    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
