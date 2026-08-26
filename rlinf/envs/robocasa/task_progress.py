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

"""Continuous task-progress signals for RoboCasa tasks.

RoboCasa ships no reward function of its own: ``Kitchen.reward()`` returns a
literal ``0`` and the ``reward_shaping`` constructor argument is annotated as one
of the "currently unused variables", so the only learning signal a RoboCasa task
exposes is the single bit returned by ``_check_success``.

Each atomic task's ``_check_success`` is, however, a *threshold on a continuous
fixture state*: a drawer's slide joint, a door's hinges, a stove knob's angle, a
sink handle's angle.  This module recovers that underlying quantity and rescales
it to ``[0, 1]``, where ``1`` means "at or past the success condition".  Used as a
potential it turns the one-bit benchmark into a graded one, in the same spirit as
the staged dense reward ManiSkill ships in ``rlinf/envs/maniskill/maniskill_env.py``.

The extractors read ``env.sim``, so they run inside the environment subprocess;
see ``rlinf/envs/robocasa/venv.py``.  Task families whose success condition is not
a threshold on a scalar -- the pick-and-place family, and the sink-spout
orientation, which compares against a categorical -- return ``None`` so the caller
keeps the sparse success bit.
"""

from typing import Optional

import numpy as np

# Bands copied from the RoboCasa sources, which threshold a raw joint angle in
# radians instead of a normalised fraction:
#   robocasa/environments/kitchen/single_stage/kitchen_stove.py
#       knob_on = 0.35 <= np.abs(knob_value) <= 2 * np.pi - 0.35
#   robocasa/models/fixtures/sink.py
#       handle_state["water_on"] = 0.40 < handle_joint_qpos < np.pi
_STOVE_KNOB_ON_BAND = (0.35, 2 * np.pi - 0.35)
_SINK_WATER_ON_BAND = (0.40, np.pi)


def _ramp(value: float, zero_at: float, one_at: float) -> float:
    """Map ``value`` onto ``[0, 1]``, reading ``0`` at ``zero_at`` and ``1`` at ``one_at``.

    ``one_at`` may lie below ``zero_at`` for a descending ramp.  The result is
    clipped, so anything past ``one_at`` saturates at ``1``.
    """
    span = one_at - zero_at
    if span == 0.0:
        return float(value >= one_at)
    return float(np.clip((value - zero_at) / span, 0.0, 1.0))


def _progress_into_band(value: float, band: tuple) -> float:
    """Progress toward entering ``band`` from below its lower edge."""
    lower_edge, _ = band
    return _ramp(value, 0.0, lower_edge)


def _progress_out_of_band(value: float, band: tuple) -> float:
    """Progress toward leaving ``band`` through whichever edge is nearer."""
    lower_edge, upper_edge = band
    half_width = 0.5 * (upper_edge - lower_edge)
    distance_to_edge = min(value - lower_edge, upper_edge - value)
    return _ramp(distance_to_edge, half_width, 0.0)


def _door_state_progress(fixture, env, behavior: str) -> Optional[float]:
    """Reduce a ``get_door_state`` dict to one progress value.

    ``get_door_state`` already reports "a percentage of how open they are", and
    ``_check_success`` requires *every* joint to pass its threshold, so the joints
    are combined with ``min``.  The percentage is used as-is rather than rescaled
    by the task's own 0.90/0.95 cutoff: a potential only has to be monotone in
    progress, and re-deriving the cutoff here would duplicate a constant that
    lives in RoboCasa.
    """
    door_state = fixture.get_door_state(env=env)
    if not door_state:
        return None
    if behavior == "open":
        return min(_ramp(value, 0.0, 1.0) for value in door_state.values())
    return min(_ramp(value, 1.0, 0.0) for value in door_state.values())


def compute_task_progress(env) -> Optional[float]:
    """Return how far ``env`` is toward its success condition, in ``[0, 1]``.

    Args:
        env: A RoboCasa ``Kitchen`` environment, unwrapped, inside the subprocess
            that owns its MuJoCo simulation.

    Returns:
        The progress fraction, or ``None`` when the task family exposes no scalar
        progress signal.  Callers should then fall back to the sparse success bit.
    """
    behavior = getattr(env, "behavior", None)

    # Drawer (``self.drawer``) and door (``self.door_fxtr``) tasks threshold the
    # fractions returned by ``get_door_state``.
    for attribute in ("drawer", "door_fxtr"):
        fixture = getattr(env, attribute, None)
        if fixture is not None and hasattr(fixture, "get_door_state"):
            if behavior in ("open", "close"):
                return _door_state_progress(fixture, env, behavior)
            return None

    # Stove knob tasks threshold the absolute knob angle.
    stove = getattr(env, "stove", None)
    knob = getattr(env, "knob", None)
    if stove is not None and knob is not None and hasattr(stove, "get_knobs_state"):
        knobs_state = stove.get_knobs_state(env=env)
        if knob not in knobs_state:
            return None
        knob_value = abs(float(knobs_state[knob]))
        if behavior == "turn_on":
            return _progress_into_band(knob_value, _STOVE_KNOB_ON_BAND)
        if behavior == "turn_off":
            return _progress_out_of_band(knob_value, _STOVE_KNOB_ON_BAND)
        return None

    # Sink faucet tasks threshold the handle angle to decide ``water_on``.  The
    # sink-spout task shares the fixture but compares an orientation label, so it
    # falls through to ``None``.
    sink = getattr(env, "sink", None)
    if sink is not None and hasattr(sink, "get_handle_state"):
        handle_state = sink.get_handle_state(env=env)
        if "handle_joint" not in handle_state:
            return None
        handle_value = float(handle_state["handle_joint"])
        if behavior == "turn_on":
            return _progress_into_band(handle_value, _SINK_WATER_ON_BAND)
        if behavior == "turn_off":
            return _progress_out_of_band(handle_value, _SINK_WATER_ON_BAND)
        return None

    return None


def shaped_potential(success, task_progress, reward_coef: float, progress_coef: float):
    """The un-differenced reward that :class:`RobocasaEnv` telescopes into a return.

    ``RobocasaEnv._calc_step_reward`` differences this quantity between steps when
    ``use_rel_reward`` is set, so an episode's return collapses to
    ``reward_coef * (success_T - success_0) + progress_coef * (progress_T -
    progress_0)`` -- classic potential-based shaping, which leaves the optimal
    policy unchanged while giving a partial pull on the drawer a nonzero score.

    ``progress_coef = 0.0`` reproduces the sparse one-bit reward exactly, which is
    what every RoboCasa config did before the potential existed.
    """
    reward = reward_coef * np.asarray(success, dtype=float)
    if progress_coef != 0.0 and task_progress is not None:
        reward = reward + progress_coef * np.asarray(task_progress, dtype=float)
    return reward
