# Copyright 2026 The RLinf Authors.
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

"""Access ``rlinf/envs/robocasa/utils.py`` without importing its package.

``rlinf.envs.robocasa.__init__`` eagerly imports ``RobocasaEnv``, which pulls in
the simulator stack (legacy ``gym``, robosuite). The helpers re-exported here are
plain NumPy, so the RoboCasa contract tests load the module straight from its
path and keep running on the dependency-light CPU CI runner.
"""

import importlib.util
from pathlib import Path

_UTILS_PATH = (
    Path(__file__).resolve().parents[2] / "rlinf" / "envs" / "robocasa" / "utils.py"
)

_spec = importlib.util.spec_from_file_location("robocasa_utils", _UTILS_PATH)
_utils = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_utils)

ROBOCASA_BASE_STATE_DIM = _utils.ROBOCASA_BASE_STATE_DIM
ROBOCASA_BASE_STATES = _utils.ROBOCASA_BASE_STATES
ROBOCASA_EXTRA_STATES = _utils.ROBOCASA_EXTRA_STATES
ROBOCASA_JOINT_STATE_DIM = _utils.ROBOCASA_JOINT_STATE_DIM
ROBOCASA_STATES = _utils.ROBOCASA_STATES
STATE_SPACE_STR_MAPPING = _utils.STATE_SPACE_STR_MAPPING
assign_task_ids = _utils.assign_task_ids
check_state_space = _utils._check_state_space
get_state_ids = _utils.get_state_ids
get_state_space = _utils.get_state_space

__all__ = [
    "ROBOCASA_BASE_STATES",
    "ROBOCASA_BASE_STATE_DIM",
    "ROBOCASA_EXTRA_STATES",
    "ROBOCASA_JOINT_STATE_DIM",
    "ROBOCASA_STATES",
    "STATE_SPACE_STR_MAPPING",
    "assign_task_ids",
    "check_state_space",
    "get_state_ids",
    "get_state_space",
]
