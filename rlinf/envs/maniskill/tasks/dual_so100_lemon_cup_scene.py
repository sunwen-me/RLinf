"""
DualSO100LemonCupScene-v1
==========================
Dual-arm SO100/SO101 scene: two robots side-by-side, a cup on the table,
a lemon slice near the right arm, and a target marker on the cup rim.
The goal is to bring the lemon slice's slot onto the cup rim.

Cameras (XR-0 format): ego, wrist_left, wrist_right

Adapted from maniskill_so101_env for RLinf integration.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.base_agent import Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.multi_agent import MultiAgent
from mani_skill.agents.registration import register_agent
from mani_skill.agents.robots.so100.so_100 import SO100
from mani_skill.envs.sapien_env import BaseEnv
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils import sapien_utils
from mani_skill.utils.building import actors
from mani_skill.utils.geometry.rotation_conversions import quaternion_apply
from mani_skill.utils.registration import register_env
from mani_skill.utils.scene_builder.table import TableSceneBuilder
from mani_skill.utils.structs.pose import Pose
from mani_skill.utils.structs.types import GPUMemoryConfig, SimConfig

# Import the SO101 agent from the same tasks directory
from rlinf.envs.maniskill.tasks.so101_agent import SO101WristCam


# ======================================================================
# SO100 variant with a wrist camera (follows panda_wristcam pattern)
# ======================================================================

@register_agent()
class SO100WristCam(SO100):
    """SO100 arm with a camera mounted on the Wrist_Pitch_Roll link."""

    uid = "so100_wristcam"

    _WRIST_CAM_LOCAL_P = [-0.05, -0.12, 0.0]
    _WRIST_CAM_LOCAL_Q = euler2quat(np.pi / 2, np.pi / 2, np.pi)

    @property
    def _sensor_configs(self):
        wrist_cam_local = sapien.Pose(
            p=self._WRIST_CAM_LOCAL_P,
            q=self._WRIST_CAM_LOCAL_Q,
        )
        return [
            CameraConfig(
                uid="wrist_camera",
                pose=wrist_cam_local,
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["Wrist_Pitch_Roll"],
            )
        ]

    def _after_loading_articulation(self):
        super()._after_loading_articulation()


# ======================================================================
# Main environment
# ======================================================================

@register_env("DualSO100LemonCupScene-v1", max_episode_steps=200)
class DualSO100LemonCupScene(BaseEnv):
    """
    **Task Description:**
    Two SO100/SO101 robots side-by-side (24 cm apart). A cup sits in front of
    them, a lemon slice is placed near the right arm, and a target marker sits
    on the cup rim. The goal is to bring the lemon slice onto the cup rim.

    **Cameras (XR-0 format):**
    - ``ego``: fixed external view covering the whole workspace
    - ``wrist_left``: attached to the left arm's wrist link
    - ``wrist_right``: attached to the right arm's wrist link

    **Success Conditions:**
    - slot center within 1.5 cm of rim target
    - slot height within 8 mm of rim height
    - slot forward axis anti-parallel to rim normal, angle < 20 deg
    - lemon is stable (low velocity)
    - cup is not knocked away
    """

    SUPPORTED_ROBOTS = [
        ("so100_wristcam", "so100_wristcam"),
        ("so101_wristcam", "so101_wristcam"),
        ("so100_wristcam", "so101_wristcam"),
        ("so101_wristcam", "so100_wristcam"),
    ]
    agent: MultiAgent

    # ---- Lemon geometry ----
    lemon_radius: float = 0.03
    lemon_half_height: float = 0.002
    slot_depth: float = 0.012
    slot_width: float = 0.005

    # ---- Cup geometry ----
    cup_radius: float = 0.035
    cup_height: float = 0.08
    cup_wall_thickness: float = 0.008
    cup_bottom_thickness: float = 0.002

    # ---- Robot layout ----
    robot_base_x: float = -0.35
    left_robot_y: float = -0.12
    right_robot_y: float = 0.12

    # ---- Slot frame: local-frame constants (lemon body frame) ----
    _SLOT_CENTER_LOCAL = None
    _SLOT_FORWARD_LOCAL = None
    _PLANE_NORMAL_LOCAL = None

    # ---- Rim frame: local-frame constants (cup body frame) ----
    _RIM_TARGET_POS_LOCAL = None
    _RIM_NORMAL_LOCAL = None
    _RIM_TANGENT_LOCAL = None
    _RIM_Z_LOCAL = None

    _lemon_mesh_path: str = "/tmp/maniskill_lemon_slot.obj"
    _slot_mesh_available: bool = False

    def __init__(
        self,
        *args,
        robot_uids=("so100_wristcam", "so100_wristcam"),
        robot_init_qpos_noise=0.02,
        easy_grasp_debug=False,
        **kwargs,
    ):
        self.robot_init_qpos_noise = robot_init_qpos_noise
        self.easy_grasp_debug = easy_grasp_debug

        if easy_grasp_debug:
            self.lemon_half_height = 0.006
            self._easy_grasp_friction = dict(
                lemon_static=4.0, lemon_dynamic=3.0,
                gripper_static=3.0, gripper_dynamic=2.5,
            )
        else:
            self._easy_grasp_friction = None

        # Compute slot frame local constants
        r = self.lemon_radius
        sd = self.slot_depth
        self._SLOT_CENTER_LOCAL = torch.tensor([r - sd, 0.0, 0.0])
        self._SLOT_FORWARD_LOCAL = torch.tensor([1.0, 0.0, 0.0])
        self._PLANE_NORMAL_LOCAL = torch.tensor([0.0, 0.0, 1.0])

        # Compute rim frame local constants
        cr = self.cup_radius
        ch = self.cup_height
        self._RIM_TARGET_POS_LOCAL = torch.tensor([0.0, cr, ch])
        self._RIM_NORMAL_LOCAL = torch.tensor([0.0, 1.0, 0.0])
        self._RIM_TANGENT_LOCAL = torch.tensor([1.0, 0.0, 0.0])
        self._RIM_Z_LOCAL = torch.tensor([0.0, 0.0, 1.0])

        super().__init__(*args, robot_uids=robot_uids, **kwargs)

    # ------------------------------------------------------------------
    # Configuration properties
    # ------------------------------------------------------------------

    @property
    def _default_sim_config(self):
        return SimConfig(
            gpu_memory_config=GPUMemoryConfig(
                found_lost_pairs_capacity=2**25,
                max_rigid_patch_count=2**19,
                max_rigid_contact_count=2**21,
            )
        )

    @property
    def _default_sensor_configs(self):
        ego_pose = sapien_utils.look_at([-0.35, 0.0, 0.55], [-0.10, 0.0, 0.0])
        return [
            CameraConfig("base_camera", ego_pose, 128, 128, np.pi / 2, 0.01, 100),
        ]

    @property
    def _default_human_render_camera_configs(self):
        pose = sapien_utils.look_at([0.5, -0.8, 0.6], [-0.1, 0.0, 0.15])
        return CameraConfig("render_camera", pose, 512, 512, 1, 0.01, 100)

    # ------------------------------------------------------------------
    # Scene loading
    # ------------------------------------------------------------------

    def _load_agent(self, options: dict):
        super()._load_agent(
            options,
            [
                sapien.Pose(p=[self.robot_base_x, self.left_robot_y, 0]),
                sapien.Pose(p=[self.robot_base_x, self.right_robot_y, 0]),
            ],
        )

    _WRIST_CAM_RENAME = {
        "so100_wristcam-0-wrist_camera": "wrist_left",
        "so100_wristcam-1-wrist_camera": "wrist_right",
        "so101_wristcam-0-wrist_camera": "wrist_left",
        "so101_wristcam-1-wrist_camera": "wrist_right",
    }

    def _setup_sensors(self, options: dict):
        super()._setup_sensors(options)
        for old_uid, new_uid in self._WRIST_CAM_RENAME.items():
            if old_uid in self._sensors:
                self._sensors[new_uid] = self._sensors.pop(old_uid)
            if old_uid in self._sensor_configs:
                self._sensor_configs[new_uid] = self._sensor_configs.pop(old_uid)
            if old_uid in self._agent_sensor_configs:
                self._agent_sensor_configs[new_uid] = self._agent_sensor_configs.pop(
                    old_uid
                )
            if hasattr(self.agent, "_sensor_config_agent_map"):
                m = self.agent._sensor_config_agent_map
                if old_uid in m:
                    m[new_uid] = m.pop(old_uid)

    def _load_scene(self, options: dict):
        """Build table, cup, lemon slice (mesh), target marker, and camera markers."""
        # Table + ground plane
        self.table_scene = TableSceneBuilder(
            env=self, robot_init_qpos_noise=self.robot_init_qpos_noise
        )
        self.table_scene.build()

        # ---- Ego camera visual marker ----
        ego_marker = actors.build_box(
            self.scene,
            half_sizes=[0.015, 0.015, 0.01],
            color=[0.0, 0.8, 0.0, 1],
            name="ego_camera_marker",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(p=[-0.35, 0.0, 0.55]),
        )
        self._hidden_objects.append(ego_marker)

        # ---- Wrist camera visual markers (small cyan boxes) ----
        self.cam_marker_left = actors.build_box(
            self.scene,
            half_sizes=[0.01, 0.01, 0.0075],
            color=[0, 0.8, 0.8, 1],
            name="cam_marker_left",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self.cam_marker_right = actors.build_box(
            self.scene,
            half_sizes=[0.01, 0.01, 0.0075],
            color=[0, 0.8, 0.8, 1],
            name="cam_marker_right",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )

        # ---- Cup (kinematic, composite shape) ----
        self.cup = self._build_cup()

        # ---- Lemon slice (dynamic, mesh with slot) ----
        self.lemon_slice = self._build_lemon_actor()

        # ---- Easy grasp support: two small blocks flanking the lemon ----
        if self.easy_grasp_debug:
            support_color = [0.6, 0.4, 0.2, 1]
            self.lemon_support_left = actors.build_box(
                self.scene,
                half_sizes=[0.003, 0.02, 0.006],
                color=support_color,
                name="lemon_support_left",
                body_type="kinematic",
                add_collision=True,
                initial_pose=sapien.Pose(p=[-0.035, 0.06, 0.006]),
            )
            self.lemon_support_right = actors.build_box(
                self.scene,
                half_sizes=[0.003, 0.02, 0.006],
                color=support_color,
                name="lemon_support_right",
                body_type="kinematic",
                add_collision=True,
                initial_pose=sapien.Pose(p=[0.035, 0.06, 0.006]),
            )

        # ---- Target marker (kinematic, green sphere at cup rim) ----
        self.target_marker = actors.build_sphere(
            self.scene,
            radius=0.008,
            color=[0, 1, 0, 1],
            name="target_marker",
            body_type="kinematic",
            add_collision=False,
            initial_pose=sapien.Pose(),
        )
        self._hidden_objects.append(self.target_marker)

        if self.easy_grasp_debug:
            self._set_gripper_friction()

    def _set_gripper_friction(self):
        """Set high friction on gripper finger collision shapes for easy_grasp_debug."""
        f = self._easy_grasp_friction
        mat = sapien.physx.PhysxMaterial(
            static_friction=f["gripper_static"],
            dynamic_friction=f["gripper_dynamic"],
            restitution=0.0,
        )
        # Try both SO100 and SO101 link names.
        # SO100: "Fixed_Jaw" / "Moving_Jaw"
        # SO101: "gripper" / "jaw"
        finger_links = ["Fixed_Jaw", "Moving_Jaw", "gripper", "jaw"]
        for agent in [self.left_agent, self.right_agent]:
            for link_name in finger_links:
                link = agent.robot.links_map.get(link_name)
                if link is not None:
                    try:
                        for body in link._bodies:
                            for cs in body.collision_shapes:
                                cs.set_physical_material(mat)
                    except Exception:
                        pass

    # ------------------------------------------------------------------
    # Cup builder
    # ------------------------------------------------------------------

    def _build_cup(self):
        """Build a hollow cylindrical cup using trimesh mesh."""
        import trimesh

        r_out = self.cup_radius
        r_in = r_out - self.cup_wall_thickness
        h = self.cup_height
        t_b = self.cup_bottom_thickness

        bottom = trimesh.creation.cylinder(radius=r_out, height=t_b)
        bottom.apply_translation([0, 0, t_b / 2])

        outer_wall = trimesh.creation.cylinder(radius=r_out, height=h - t_b)
        outer_wall.apply_translation([0, 0, t_b + (h - t_b) / 2])

        inner_cut = trimesh.creation.cylinder(radius=r_in, height=h - t_b + 0.001)
        inner_cut.apply_translation([0, 0, t_b + (h - t_b) / 2])

        cup_mesh = trimesh.util.concatenate([bottom, outer_wall])
        cup_mesh = cup_mesh.difference(inner_cut)

        cup_mesh_path = "/tmp/maniskill_cup.obj"
        cup_mesh.export(cup_mesh_path)

        builder = self.scene.create_actor_builder()

        cup_color = sapien.render.RenderMaterial(base_color=[0.9, 0.9, 0.95, 1])
        builder.add_visual_from_file(cup_mesh_path, material=cup_color)
        builder.add_multiple_convex_collisions_from_file(
            cup_mesh_path,
            decomposition="coacd",
            material=sapien.physx.PhysxMaterial(
                static_friction=2.0, dynamic_friction=1.5, restitution=0.0
            ),
        )

        builder.initial_pose = sapien.Pose(p=[0, 0, 0])
        cup = builder.build(name="cup")
        cup.set_mass(0.5)
        try:
            for body in cup._bodies:
                body.set_linear_damping(2.0)
                body.set_angular_damping(2.0)
        except Exception:
            pass
        return cup

    # ------------------------------------------------------------------
    # Lemon mesh builder
    # ------------------------------------------------------------------

    def _build_lemon_mesh(self):
        """Create a trimesh for the lemon slice with a slot cut from the +X edge."""
        import trimesh

        lemon = trimesh.creation.cylinder(
            radius=self.lemon_radius,
            height=self.lemon_half_height * 2,
        )

        try:
            slot_box = trimesh.creation.box(
                extents=[
                    self.slot_depth,
                    self.slot_width,
                    self.lemon_half_height * 2 + 0.001,
                ]
            )
            slot_box.apply_translation(
                [self.lemon_radius - self.slot_depth / 2, 0, 0]
            )
            lemon_with_slot = lemon.difference(slot_box)
            if lemon_with_slot is not None and len(lemon_with_slot.faces) > 0:
                return lemon_with_slot, True
        except Exception:
            pass

        import warnings
        warnings.warn(
            "Trimesh boolean not available; using plain cylinder for lemon."
        )
        return lemon, False

    def _build_lemon_actor(self):
        """Build the lemon slice actor from procedural mesh."""
        builder = self.scene.create_actor_builder()

        mesh, has_slot = self._build_lemon_mesh()
        self._slot_mesh_available = has_slot
        mesh.export(self._lemon_mesh_path)

        yellow = sapien.render.RenderMaterial(base_color=[1.0, 0.95, 0.2, 1])
        builder.add_visual_from_file(self._lemon_mesh_path, material=yellow)

        if self.easy_grasp_debug and self._easy_grasp_friction is not None:
            f = self._easy_grasp_friction
            phys_mat = sapien.physx.PhysxMaterial(
                static_friction=f["lemon_static"],
                dynamic_friction=f["lemon_dynamic"],
                restitution=0.0,
            )
            builder.add_multiple_convex_collisions_from_file(
                self._lemon_mesh_path,
                decomposition="coacd",
                material=phys_mat,
            )
        else:
            builder.add_multiple_convex_collisions_from_file(
                self._lemon_mesh_path,
                decomposition="coacd",
            )

        builder.initial_pose = sapien.Pose(p=[0.0, 0.06, self.lemon_half_height])
        lemon = builder.build(name="lemon_slice")
        lemon.set_mass(0.01)
        try:
            for body in lemon._bodies:
                body.set_linear_damping(1.0)
                body.set_angular_damping(1.0)
        except Exception:
            pass
        return lemon

    # ------------------------------------------------------------------
    # Frame computations (batched, GPU-safe)
    # ------------------------------------------------------------------

    def _get_slot_frame(self):
        """Get the lemon slot frame in world coordinates."""
        lemon_q = self.lemon_slice.pose.q
        lemon_p = self.lemon_slice.pose.p

        sc = self._SLOT_CENTER_LOCAL.to(self.device)
        sf = self._SLOT_FORWARD_LOCAL.to(self.device)
        pn = self._PLANE_NORMAL_LOCAL.to(self.device)

        b = lemon_p.shape[0]
        sc = sc.unsqueeze(0).expand(b, -1)
        sf = sf.unsqueeze(0).expand(b, -1)
        pn = pn.unsqueeze(0).expand(b, -1)

        slot_center = lemon_p + quaternion_apply(lemon_q, sc)
        slot_forward = quaternion_apply(lemon_q, sf)
        plane_normal = quaternion_apply(lemon_q, pn)

        return slot_center, slot_forward, plane_normal

    def _get_rim_frame(self):
        """Get the cup rim target frame in world coordinates."""
        cup_q = self.cup.pose.q
        cup_p = self.cup.pose.p

        rp = self._RIM_TARGET_POS_LOCAL.to(self.device)
        rn = self._RIM_NORMAL_LOCAL.to(self.device)
        rt = self._RIM_TANGENT_LOCAL.to(self.device)
        rz = self._RIM_Z_LOCAL.to(self.device)

        b = cup_p.shape[0]
        rp = rp.unsqueeze(0).expand(b, -1)
        rn = rn.unsqueeze(0).expand(b, -1)
        rt = rt.unsqueeze(0).expand(b, -1)
        rz = rz.unsqueeze(0).expand(b, -1)

        rim_pos = cup_p + quaternion_apply(cup_q, rp)
        rim_normal = quaternion_apply(cup_q, rn)
        rim_tangent = quaternion_apply(cup_q, rt)
        rim_z = quaternion_apply(cup_q, rz)

        return rim_pos, rim_normal, rim_tangent, rim_z

    # ------------------------------------------------------------------
    # Step: update wrist camera markers
    # ------------------------------------------------------------------

    def step(self, action):
        # RLinf env worker sends flat array/tensor (B, 12), but MultiAgent needs dict.
        # Split flat action into per-agent dict: left 6D + right 6D.
        if not isinstance(action, dict):
            action_dim = action.shape[-1]
            if action_dim == 12:
                left_uid = self.agent.agents[0].uid + "-0"
                right_uid = self.agent.agents[1].uid + "-1"
                action = {
                    left_uid: action[..., :6],
                    right_uid: action[..., 6:],
                }
        obs, reward, terminated, truncated, info = super().step(action)
        self._update_camera_markers()
        return obs, reward, terminated, truncated, info

    def _get_cam_link(self, agent):
        """Get the camera link: handeye_cam if available, else Wrist_Pitch_Roll."""
        links = agent.robot.links_map
        for name in ["handeye_cam", "Wrist_Pitch_Roll", "wrist"]:
            if name in links:
                return links[name]
        return list(links.values())[0]

    def _update_camera_markers(self):
        self.cam_marker_left.set_pose(self._get_cam_link(self.left_agent).pose)
        self.cam_marker_right.set_pose(self._get_cam_link(self.right_agent).pose)

    # ------------------------------------------------------------------
    # Episode initialization
    # ------------------------------------------------------------------

    def _initialize_episode(self, env_idx: torch.Tensor, options: dict):
        with torch.device(self.device):
            b = len(env_idx)

            self.table_scene.initialize(env_idx)
            self._reset_robots(env_idx, b)

            # Cup at origin
            self.cup.set_pose(Pose.create_from_pq(p=torch.tensor([[0.0, 0.0, 0.0]])))

            # Lemon slice near right arm
            lemon_xy = torch.zeros((b, 2))
            lemon_xy[:, 0] = torch.rand((b,)) * 0.06 - 0.03
            lemon_xy[:, 1] = 0.04 + torch.rand((b,)) * 0.04
            lemon_xyz = torch.zeros((b, 3))
            lemon_xyz[:, :2] = lemon_xy

            if self.easy_grasp_debug:
                lemon_xyz[:, 0] = 0.0
                lemon_xyz[:, 1] = 0.06
                lemon_xyz[:, 2] = self.lemon_half_height
            else:
                lemon_xyz[:, 2] = self.lemon_half_height

            self.lemon_slice.set_pose(Pose.create_from_pq(p=lemon_xyz))

            # Target marker at the rim target position
            rim_target = self._RIM_TARGET_POS_LOCAL.to(self.device).unsqueeze(0).expand(b, -1)
            self.target_marker.set_pose(Pose.create_from_pq(p=rim_target))

        self._update_camera_markers()

    # SO101 URDF joint limits with safety margin (must match so101_agent.py keyframes)
    _JOINT_LOWER = np.array([-1.9199, -1.7453, -1.7453, -1.6581, -1.3700, -0.2000])
    _JOINT_UPPER = np.array([1.9199, 1.7453, 1.5708, 1.6581, 4.4907, 1.7453])
    _JOINT_MARGIN = 0.05

    def _reset_robots(self, env_idx: torch.Tensor, b: int):
        # Rest qpos: all values strictly within URDF limits with margin.
        # joint[2]=1.50 (was 1.5708=AT limit), joint[5]=0.50 (was -1.1=BELOW limit)
        rest_qpos = np.array([0, -1.5708, 1.50, 0.66, 0, 0.5])
        yaw = np.pi / 2

        left_qpos = rest_qpos + self._episode_rng.normal(
            0, self.robot_init_qpos_noise, (b, len(rest_qpos))
        )
        # Clip to safe range: [lower + margin, upper - margin]
        lo = self._JOINT_LOWER + self._JOINT_MARGIN
        hi = self._JOINT_UPPER - self._JOINT_MARGIN
        left_qpos = np.clip(left_qpos, lo, hi)

        self.left_agent.reset(left_qpos)
        self.left_agent.robot.set_pose(
            sapien.Pose(
                p=[self.robot_base_x, self.left_robot_y, 0],
                q=euler2quat(0, 0, yaw),
            )
        )

        right_qpos = rest_qpos + self._episode_rng.normal(
            0, self.robot_init_qpos_noise, (b, len(rest_qpos))
        )
        right_qpos = np.clip(right_qpos, lo, hi)

        self.right_agent.reset(right_qpos)
        self.right_agent.robot.set_pose(
            sapien.Pose(
                p=[self.robot_base_x, self.right_robot_y, 0],
                q=euler2quat(0, 0, yaw),
            )
        )

        # Assert qpos is within safe bounds (first env only, to avoid perf overhead)
        self._assert_qpos_in_bounds(self.left_agent, "left")
        self._assert_qpos_in_bounds(self.right_agent, "right")

    def _assert_qpos_in_bounds(self, agent, label: str):
        """Assert all qpos are within [lower + margin, upper - margin]."""
        qpos = agent.robot.get_qpos()[0].cpu().numpy()
        lo = self._JOINT_LOWER + self._JOINT_MARGIN
        hi = self._JOINT_UPPER - self._JOINT_MARGIN
        for i in range(len(qpos)):
            if qpos[i] < lo[i] or qpos[i] > hi[i]:
                joint_name = agent.robot.active_joints[i].name
                raise AssertionError(
                    f"[{label}] joint '{joint_name}' (idx {i}): "
                    f"qpos={qpos[i]:.4f} out of safe range "
                    f"[{lo[i]:.4f}, {hi[i]:.4f}] "
                    f"(URDF [{self._JOINT_LOWER[i]:.4f}, {self._JOINT_UPPER[i]:.4f}], "
                    f"margin={self._JOINT_MARGIN})"
                )

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def left_agent(self):
        return self.agent.agents[0]

    @property
    def right_agent(self):
        return self.agent.agents[1]

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self):
        slot_center, slot_forward, _ = self._get_slot_frame()
        rim_pos, rim_normal, _, _ = self._get_rim_frame()

        slot_to_rim = torch.linalg.norm(slot_center - rim_pos, axis=1)
        slot_near_rim = slot_to_rim < 0.015

        height_error = torch.abs(slot_center[:, 2] - rim_pos[:, 2])
        height_ok = height_error < 0.008

        cos_angle = torch.sum(slot_forward * rim_normal, axis=1)
        abs_cos = torch.abs(cos_angle)
        angle_error = torch.arccos(torch.clamp(abs_cos, -1.0, 1.0))
        orientation_ok = angle_error < (20.0 * np.pi / 180.0)

        lemon_vel = torch.linalg.norm(self.lemon_slice.linear_velocity, axis=1)
        lemon_stable = lemon_vel < 0.1

        cup_xy = self.cup.pose.p[..., :2]
        cup_stable = torch.linalg.norm(cup_xy, axis=1) < 0.05

        success = slot_near_rim & height_ok & orientation_ok & lemon_stable & cup_stable

        return {
            "success": success,
            "slot_near_rim": slot_near_rim,
            "height_ok": height_ok,
            "orientation_ok": orientation_ok,
            "lemon_stable": lemon_stable,
            "cup_stable": cup_stable,
            "lemon_to_target_dist": slot_to_rim,
            "slot_alignment_error": angle_error,
        }

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------

    def _get_obs_extra(self, info: dict):
        obs = dict(
            left_tcp=self.left_agent.tcp_pose.raw_pose,
            right_tcp=self.right_agent.tcp_pose.raw_pose,
        )
        if "state" in self.obs_mode:
            slot_center, slot_forward, _ = self._get_slot_frame()
            rim_pos, rim_normal, _, _ = self._get_rim_frame()

            obs.update(
                cup_pose=self.cup.pose.raw_pose,
                lemon_pose=self.lemon_slice.pose.raw_pose,
                target_pose=self.target_marker.pose.raw_pose,
                left_tcp_to_lemon=self.lemon_slice.pose.p - self.left_agent.tcp_pose.p,
                right_tcp_to_lemon=self.lemon_slice.pose.p - self.right_agent.tcp_pose.p,
                lemon_to_target=rim_pos - slot_center,
                slot_center=slot_center,
                slot_forward=slot_forward,
                rim_target_pos=rim_pos,
                rim_normal=rim_normal,
            )
        return obs

    # ------------------------------------------------------------------
    # Reward
    # ------------------------------------------------------------------

    def compute_dense_reward(self, obs: Any, action: torch.Tensor, info: dict):
        slot_center, slot_forward, _ = self._get_slot_frame()
        rim_pos, rim_normal, _, _ = self._get_rim_frame()

        tcp_to_lemon = torch.linalg.norm(
            self.lemon_slice.pose.p - self.right_agent.tcp_pose.p, axis=1
        )
        reaching_reward = 1 - torch.tanh(5 * tcp_to_lemon)

        slot_to_rim = torch.linalg.norm(slot_center - rim_pos, axis=1)
        positioning_reward = 1 - torch.tanh(5 * slot_to_rim)

        cos_angle = torch.sum(slot_forward * rim_normal, axis=1)
        abs_cos = torch.abs(cos_angle)
        alignment_reward = abs_cos

        height_error = torch.abs(slot_center[:, 2] - rim_pos[:, 2])
        height_reward = 1 - torch.tanh(20 * height_error)

        cup_xy = self.cup.pose.p[..., :2]
        cup_knock_penalty = torch.linalg.norm(cup_xy, axis=1)

        success_bonus = info["success"].float() * 10.0

        reward = (
            1.0 * reaching_reward
            + 2.0 * positioning_reward
            + 1.0 * alignment_reward
            + 1.0 * height_reward
            - 0.5 * cup_knock_penalty
            + success_bonus
        )

        return reward

    def compute_normalized_dense_reward(
        self, obs: Any, action: torch.Tensor, info: dict
    ):
        return self.compute_dense_reward(obs=obs, action=action, info=info) / 15.0
