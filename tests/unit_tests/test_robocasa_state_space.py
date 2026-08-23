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

"""Contract tests for the RoboCasa state layout.

The optional ``robot0_joint_pos`` block (needed by XR-1) is appended *after* the
25 end-effector dims so that models slicing the state by index -- the OpenPI
``"16d"``/``"25d"`` recipes -- keep observing exactly the same numbers. These
tests pin that layout down.
"""

import types

import numpy as np
import pytest
from _robocasa_utils import (
    ROBOCASA_BASE_STATE_DIM,
    ROBOCASA_BASE_STATES,
    ROBOCASA_EXTRA_STATES,
    ROBOCASA_JOINT_STATE_DIM,
    ROBOCASA_STATES,
    STATE_SPACE_STR_MAPPING,
    check_state_space,
    get_state_ids,
    get_state_space,
)

PRESET_WIDTHS = {"16d": 16, "25d": 25, "32d": 32}


def test_base_block_is_contiguous_and_25_wide():
    ids = np.concatenate([ROBOCASA_BASE_STATES[name] for name in ROBOCASA_BASE_STATES])
    assert ids.tolist() == list(range(ROBOCASA_BASE_STATE_DIM))
    assert ROBOCASA_BASE_STATE_DIM == 25


def test_joint_block_is_appended_after_the_base_block():
    assert list(ROBOCASA_EXTRA_STATES) == ["robot0_joint_pos"]
    joint_ids = ROBOCASA_EXTRA_STATES["robot0_joint_pos"]
    assert ROBOCASA_JOINT_STATE_DIM == 7
    assert joint_ids.tolist() == list(range(25, 32))
    assert len(joint_ids) == ROBOCASA_JOINT_STATE_DIM
    assert joint_ids[0] == ROBOCASA_BASE_STATE_DIM


def test_state_registry_merges_without_shadowing_the_base_block():
    assert ROBOCASA_STATES.keys() == {*ROBOCASA_BASE_STATES, *ROBOCASA_EXTRA_STATES}
    for name, ids in ROBOCASA_BASE_STATES.items():
        assert ROBOCASA_STATES[name].tolist() == ids.tolist()
    all_ids = get_state_ids(list(ROBOCASA_STATES))
    assert all_ids == list(range(ROBOCASA_BASE_STATE_DIM + ROBOCASA_JOINT_STATE_DIM))


@pytest.mark.parametrize(("preset", "width"), sorted(PRESET_WIDTHS.items()))
def test_preset_widths_and_uniqueness(preset, width):
    state_space = get_state_space(preset)
    assert state_space == STATE_SPACE_STR_MAPPING[preset]
    assert check_state_space(state_space)
    ids = get_state_ids(state_space)
    assert len(ids) == width
    assert len(set(ids)) == width


@pytest.mark.parametrize("preset", ["16d", "25d"])
def test_end_effector_presets_never_reach_into_the_joint_block(preset):
    """Enabling ``include_joint_state`` must not move the OpenPI state slices."""
    ids = get_state_ids(get_state_space(preset))
    assert max(ids) < ROBOCASA_BASE_STATE_DIM
    assert "robot0_joint_pos" not in get_state_space(preset)


def test_get_state_space_passes_lists_through():
    state_space = ["robot0_eef_pos", "robot0_joint_pos"]
    assert get_state_space(state_space) is state_space


def test_get_state_space_returns_none_for_unregistered_presets():
    assert get_state_space("64d") is None


def _make_env(include_joint_state: bool):
    """A ``RobocasaEnv`` shell carrying only what state extraction reads.

    ``__init__`` needs a live robosuite simulator, so the two attributes it
    derives from the env config are set here the same way it derives them.
    """
    robocasa_env = pytest.importorskip("rlinf.envs.robocasa.robocasa_env")
    env = object.__new__(robocasa_env.RobocasaEnv)
    env.include_joint_state = include_joint_state
    env.state_dim = ROBOCASA_BASE_STATE_DIM + (
        ROBOCASA_JOINT_STATE_DIM if include_joint_state else 0
    )
    env.cfg = types.SimpleNamespace(robot_name="PandaOmron")
    return env


def _make_obs(joint_pos=None):
    obs = {
        name: np.arange(len(ids), dtype=np.float32) + 10 * offset
        for offset, (name, ids) in enumerate(ROBOCASA_BASE_STATES.items())
    }
    for image_key in (
        "robot0_agentview_left_image",
        "robot0_eye_in_hand_image",
        "robot0_agentview_right_image",
    ):
        obs[image_key] = np.arange(2 * 2 * 3, dtype=np.uint8).reshape(2, 2, 3)
    if joint_pos is not None:
        obs["robot0_joint_pos"] = np.asarray(joint_pos, dtype=np.float32)
    return obs


def test_extracted_state_matches_the_registry_indices():
    obs = _make_obs()
    extracted = _make_env(include_joint_state=False)._extract_image_and_state([obs])

    state = extracted["state"]
    assert state.shape == (1, ROBOCASA_BASE_STATE_DIM)
    for name, ids in ROBOCASA_BASE_STATES.items():
        assert state[0][ids] == pytest.approx(obs[name])
    # Images are flipped vertically: robosuite renders in OpenGL coordinates.
    assert np.array_equal(
        extracted["robot0_agentview_left_image"][0],
        obs["robot0_agentview_left_image"][::-1],
    )


def test_joint_state_only_appends_and_leaves_the_base_dims_untouched():
    joint_pos = np.arange(ROBOCASA_JOINT_STATE_DIM, dtype=np.float32) / 10.0
    base_state = _make_env(include_joint_state=False)._extract_image_and_state(
        [_make_obs()]
    )["state"]
    full_state = _make_env(include_joint_state=True)._extract_image_and_state(
        [_make_obs(joint_pos=joint_pos)]
    )["state"]

    assert full_state.shape == (1, ROBOCASA_BASE_STATE_DIM + ROBOCASA_JOINT_STATE_DIM)
    assert np.array_equal(full_state[:, :ROBOCASA_BASE_STATE_DIM], base_state)
    joint_ids = ROBOCASA_EXTRA_STATES["robot0_joint_pos"]
    assert full_state[0][joint_ids] == pytest.approx(joint_pos)


def test_missing_joint_pos_observable_is_reported():
    env = _make_env(include_joint_state=True)
    with pytest.raises(KeyError, match="robot0_joint_pos"):
        env._extract_image_and_state([_make_obs()])


def test_unexpected_joint_count_is_reported():
    env = _make_env(include_joint_state=True)
    obs = _make_obs(joint_pos=np.zeros(6, dtype=np.float32))
    with pytest.raises(ValueError, match="arm joint positions"):
        env._extract_image_and_state([obs])
