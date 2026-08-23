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

"""Contract tests for the RoboCasa multi-task environment assignment.

A multi-task ``task_names`` list is only usable with group-relative algorithms
if every rollout group stays on one task: GRPO averages the reward of
``group_size`` consecutive environments, so a group spanning several tasks would
score easy and hard episodes against a shared baseline. These tests pin that
property, and the global numbering that spreads the task list over env ranks.
"""

import numpy as np
import pytest
from _robocasa_utils import assign_task_ids

TASKS = [
    "CloseDrawer",
    "OpenDrawer",
    "OpenSingleDoor",
    "TurnOnSinkFaucet",
]


def test_single_task_is_always_index_zero():
    for group_size in (1, 4, 8):
        ids = assign_task_ids(
            num_envs=16, num_tasks=1, group_size=group_size, seed_offset=3
        )
        assert ids.tolist() == [0] * 16


def test_group_size_one_round_robins_over_tasks():
    ids = assign_task_ids(num_envs=8, num_tasks=len(TASKS), group_size=1)
    assert ids.tolist() == [0, 1, 2, 3, 0, 1, 2, 3]


def test_task_advances_once_per_group():
    ids = assign_task_ids(num_envs=16, num_tasks=len(TASKS), group_size=4)
    assert ids.tolist() == [0] * 4 + [1] * 4 + [2] * 4 + [3] * 4


@pytest.mark.parametrize("group_size", [2, 4, 8])
def test_every_group_runs_exactly_one_task(group_size):
    num_envs = 8 * group_size
    ids = assign_task_ids(
        num_envs=num_envs, num_tasks=len(TASKS), group_size=group_size, seed_offset=1
    )
    groups = ids.reshape(-1, group_size)
    assert all(len(set(group.tolist())) == 1 for group in groups)


def test_ranks_continue_the_task_list_instead_of_restarting_it():
    # Two env ranks, 8 envs each, four tasks and a group size of 4: the eight
    # rollout groups have to cover every task twice, not task 0 and 1 four times.
    per_rank = [
        assign_task_ids(
            num_envs=8, num_tasks=len(TASKS), group_size=4, seed_offset=rank
        )
        for rank in range(2)
    ]
    assert per_rank[0].tolist() == [0] * 4 + [1] * 4
    assert per_rank[1].tolist() == [2] * 4 + [3] * 4
    counts = np.bincount(np.concatenate(per_rank), minlength=len(TASKS))
    assert counts.tolist() == [4, 4, 4, 4]


def test_balanced_coverage_when_env_count_is_a_multiple_of_group_times_tasks():
    group_size, num_tasks, num_ranks = 8, len(TASKS), 2
    total_num_envs = group_size * num_tasks * num_ranks
    ids = np.concatenate(
        [
            assign_task_ids(
                num_envs=total_num_envs // num_ranks,
                num_tasks=num_tasks,
                group_size=group_size,
                seed_offset=rank,
            )
            for rank in range(num_ranks)
        ]
    )
    counts = np.bincount(ids, minlength=num_tasks)
    assert counts.tolist() == [group_size * num_ranks] * num_tasks


def test_task_ids_index_the_task_name_list():
    ids = assign_task_ids(num_envs=12, num_tasks=len(TASKS), group_size=3)
    names = [TASKS[i] for i in ids]
    assert names[:3] == ["CloseDrawer"] * 3
    assert set(names) == set(TASKS)
    assert ids.dtype == np.int64
