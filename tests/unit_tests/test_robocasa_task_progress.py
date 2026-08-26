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

"""Contract tests for the RoboCasa task-progress signal.

RoboCasa's own ``reward()`` returns ``0``, so the wrapper's only learning signal is
the bit from ``_check_success``. ``compute_task_progress`` recovers the continuous
fixture state that each ``_check_success`` thresholds. These tests pin the two
properties the shaped reward depends on: progress is monotone toward success, and
a *successful* state scores essentially the full potential -- otherwise shaping
could pay a near-miss more than a success.
"""

import types

import numpy as np
import pytest
from _robocasa_utils import compute_task_progress, shaped_potential

# Mirrors robocasa/environments/kitchen/single_stage/kitchen_stove.py, which reads
# ``0.35 <= abs(knob_value) <= 2 * pi - 0.35`` as "burner on".
STOVE_ON = 0.35
STOVE_ON_UPPER = 2 * np.pi - STOVE_ON
# Mirrors robocasa/models/fixtures/sink.py: ``0.40 < handle_joint_qpos < pi``.
SINK_ON = 0.40


def drawer_env(behavior, **door_state):
    """A ``ManipulateDrawer`` stand-in exposing only ``get_door_state``."""
    drawer = types.SimpleNamespace(get_door_state=lambda env: dict(door_state))
    return types.SimpleNamespace(behavior=behavior, drawer=drawer)


def door_env(behavior, **door_state):
    """A ``ManipulateDoor`` stand-in; the fixture is ``door_fxtr``, not ``drawer``."""
    fixture = types.SimpleNamespace(get_door_state=lambda env: dict(door_state))
    return types.SimpleNamespace(behavior=behavior, door_fxtr=fixture)


def stove_env(behavior, knob_value):
    """A ``ManipulateStoveKnob`` stand-in with one registered burner."""
    stove = types.SimpleNamespace(
        get_knobs_state=lambda env: {"front_left": knob_value}
    )
    return types.SimpleNamespace(behavior=behavior, stove=stove, knob="front_left")


def sink_env(behavior, handle_joint, spout_ori="center"):
    """A ``ManipulateSinkFaucet``/``TurnSinkSpout`` stand-in."""
    sink = types.SimpleNamespace(
        get_handle_state=lambda env: {
            "handle_joint": handle_joint,
            "water_on": SINK_ON < handle_joint < np.pi,
            "spout_ori": spout_ori,
        }
    )
    return types.SimpleNamespace(behavior=behavior, sink=sink)


@pytest.mark.parametrize(
    "door, expected",
    [(0.0, 0.0), (0.5, 0.5), (0.95, 0.95), (1.0, 1.0), (1.4, 1.0)],
)
def test_open_drawer_progress_tracks_the_slide_joint(door, expected):
    assert compute_task_progress(drawer_env("open", door=door)) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "door, expected",
    [(1.0, 0.0), (0.5, 0.5), (0.05, 0.95), (0.0, 1.0), (-0.2, 1.0)],
)
def test_close_drawer_progress_is_the_mirror_image(door, expected):
    assert compute_task_progress(drawer_env("close", door=door)) == pytest.approx(
        expected
    )


def test_a_successful_state_scores_essentially_the_whole_potential():
    # RoboCasa calls the drawer open at 0.95 and the door open at 0.90. Progress
    # must already be at least that high there, so no near-miss can out-earn a
    # success once the potential is weighted at most as much as the success bit.
    assert compute_task_progress(drawer_env("open", door=0.95)) >= 0.95
    both_hinges_open = door_env("open", left_door=0.90, right_door=0.90)
    assert compute_task_progress(both_hinges_open) >= 0.90


def test_double_door_progress_takes_the_worse_hinge():
    # ``_check_success`` requires *every* hinge past the threshold, so one hinge
    # left shut must not be averaged away by the other.
    env = door_env("open", left_door=0.98, right_door=0.20)
    assert compute_task_progress(env) == pytest.approx(0.20)


@pytest.mark.parametrize("behavior", ["open", "close"])
def test_progress_is_monotone_in_the_success_direction(behavior):
    states = np.linspace(0.0, 1.0, 21)
    values = [compute_task_progress(drawer_env(behavior, door=s)) for s in states]
    differences = np.diff(values)
    if behavior == "open":
        assert np.all(differences >= 0.0)
    else:
        assert np.all(differences <= 0.0)


@pytest.mark.parametrize(
    "knob_value, expected",
    [
        (0.0, 0.0),
        (STOVE_ON / 2, 0.5),
        (STOVE_ON, 1.0),
        (1.0, 1.0),
        # The knob turns either way; ``_check_success`` thresholds ``abs``.
        (-STOVE_ON, 1.0),
        (-STOVE_ON / 2, 0.5),
    ],
)
def test_turn_on_stove_progress_ramps_up_to_the_on_band(knob_value, expected):
    assert compute_task_progress(stove_env("turn_on", knob_value)) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "knob_value, expected",
    [
        (np.pi, 0.0),  # mid-band: as far from "off" as the knob gets
        (STOVE_ON, 1.0),  # at the lower edge of the on-band
        (STOVE_ON_UPPER, 1.0),  # ... or the upper one, whichever is nearer
        (0.1, 1.0),  # already off
        (STOVE_ON + (np.pi - STOVE_ON) / 2, 0.5),
    ],
)
def test_turn_off_stove_progress_measures_distance_to_the_nearer_edge(
    knob_value, expected
):
    assert compute_task_progress(stove_env("turn_off", knob_value)) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "behavior, handle_joint, expected",
    [
        ("turn_on", 0.0, 0.0),
        ("turn_on", SINK_ON / 2, 0.5),
        ("turn_on", SINK_ON, 1.0),
        ("turn_off", (SINK_ON + np.pi) / 2, 0.0),
        ("turn_off", SINK_ON, 1.0),
        ("turn_off", np.pi, 1.0),
    ],
)
def test_sink_faucet_progress_uses_the_handle_angle(behavior, handle_joint, expected):
    assert compute_task_progress(sink_env(behavior, handle_joint)) == pytest.approx(
        expected
    )


@pytest.mark.parametrize(
    "env",
    [
        # TurnSinkSpout shares the fixture but compares a categorical orientation.
        sink_env("left", 1.0, spout_ori="right"),
        # Pick-and-place succeeds on an object-in-receptacle check, not a threshold.
        types.SimpleNamespace(behavior=None),
        # A fixture whose joints could not be read at all.
        drawer_env("open"),
    ],
)
def test_tasks_without_a_scalar_success_signal_report_no_progress(env):
    assert compute_task_progress(env) is None


# ---------------------------------------------------------------------------
# The shaped reward built on top of the potential.
# ---------------------------------------------------------------------------

REWARD_COEF = 1.0
# Matches the dense mass of the repo's only other embodied dense reward, the
# staged ManiSkill one in ``rlinf/envs/maniskill/maniskill_env.py``: two 0.1
# milestones against a terminal 1.0.  Keeping the same 0.2 : 1.0 ratio also keeps
# a group's episode-summed reward inside the ``rewards_{lower,upper}_bound``
# window the RoboCasa configs filter on.
PROGRESS_COEF = 0.2


def episode_return(successes, progresses, progress_coef=PROGRESS_COEF):
    """Sum the per-step rewards the way ``RobocasaEnv`` accumulates them.

    Mirrors ``_calc_step_reward`` under ``use_rel_reward``: the reset seeds
    ``prev_step_reward`` with the head start already on the fixture, then every
    step contributes the *difference* of the shaped potential.
    """
    prev = shaped_potential(False, progresses[0], REWARD_COEF, progress_coef)
    total = 0.0
    for success, progress in zip(successes, progresses[1:]):
        reward = shaped_potential(success, progress, REWARD_COEF, progress_coef)
        total += float(reward - prev)
        prev = reward
    return total


def test_zero_coef_reproduces_the_sparse_reward():
    """The default ``progress_coef: 0.0`` must leave existing runs bit-identical."""
    for progress in (0.0, 0.37, 1.0):
        assert shaped_potential(False, progress, REWARD_COEF, 0.0) == 0.0
        assert shaped_potential(True, progress, REWARD_COEF, 0.0) == REWARD_COEF
    # And a whole episode of progress with no success still pays nothing.
    assert episode_return([False] * 4, [0.0, 0.3, 0.6, 0.9, 0.95], 0.0) == 0.0


def test_episode_return_telescopes_to_the_progress_gained():
    """Potential-based shaping: only the endpoints of the episode matter."""
    progresses = [0.0, 0.1, 0.45, 0.3, 0.62]
    got = episode_return([False] * 4, progresses)
    assert got == pytest.approx(PROGRESS_COEF * (progresses[-1] - progresses[0]))
    # A detour that returns to the same state earns the same as going straight there.
    straight = episode_return([False] * 2, [0.0, 0.31, 0.62])
    assert straight == pytest.approx(got)


def test_a_partial_pull_outranks_doing_nothing():
    """The reason the potential exists: the sparse bit scores these identically."""
    idle = episode_return([False] * 3, [0.0, 0.0, 0.0, 0.0])
    partial = episode_return([False] * 3, [0.0, 0.2, 0.45, 0.6])
    assert idle == 0.0
    assert partial > idle


def test_success_still_pays_more_than_any_near_miss():
    """``progress_coef <= reward_coef`` keeps the benchmark metric on top."""
    near_miss = episode_return([False] * 2, [0.0, 0.5, 0.999])
    success = episode_return([False, True], [0.0, 0.5, 1.0])
    assert success > near_miss
    # The gap is the full success payout, not a sliver of shaping.
    assert success - near_miss > 0.9 * REWARD_COEF


def test_reset_head_start_is_not_paid_out():
    """A fixture that starts part-open must not hand out free reward."""
    assert episode_return([False] * 3, [0.4, 0.4, 0.4, 0.4]) == pytest.approx(0.0)
    # Only the progress made *after* the reset is paid.
    assert episode_return([False] * 2, [0.4, 0.6, 0.7]) == pytest.approx(
        PROGRESS_COEF * 0.3
    )


def test_shaping_is_vectorised_over_the_env_batch():
    """``_calc_step_reward`` hands whole ``num_envs`` arrays through this."""
    success = np.array([True, False, False])
    progress = np.array([1.0, 0.6, 0.0])
    reward = shaped_potential(success, progress, REWARD_COEF, PROGRESS_COEF)
    np.testing.assert_allclose(reward, [1.2, 0.12, 0.0])
