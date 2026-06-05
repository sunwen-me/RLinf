"""
SO101 robot agent for ManiSkill with wrist camera.

SO101 has different link names than SO100:
  - gripper (instead of Fixed_Jaw)
  - jaw (instead of Moving_Jaw)
  - gripper_camera_mount (camera mount link)
  - camera_module, handeye_cam (camera links)

URDF joints are numbered "1"-"6":
  1: shoulder_pan   (base → shoulder)
  2: shoulder_lift  (shoulder → upper_arm)
  3: elbow_flex     (upper_arm → lower_arm)
  4: wrist_flex     (lower_arm → wrist)
  5: wrist_roll     (wrist → gripper)
  6: gripper        (gripper → jaw)
"""

import os

import numpy as np
import sapien
import torch
from transforms3d.euler import euler2quat

from mani_skill.agents.base_agent import BaseAgent, Keyframe
from mani_skill.agents.controllers import *
from mani_skill.agents.registration import register_agent
from mani_skill.sensors.camera import CameraConfig
from mani_skill.utils.structs.pose import Pose


def deepcopy_dict(d):
    import copy
    return {k: copy.deepcopy(v) for k, v in d.items()}


# SO101 URDF path (relative to this file)
_SO101_URDF = os.path.join(
    os.path.dirname(__file__), "..", "assets", "robots", "so100", "so101_fixed.urdf"
)


@register_agent()
class SO101(BaseAgent):
    """SO101 robot base class."""

    uid = "so101"
    urdf_path = _SO101_URDF

    # SO101 link names
    FINGER1_LINK = "gripper"        # equivalent to SO100's Fixed_Jaw
    FINGER2_LINK = "jaw"            # equivalent to SO100's Moving_Jaw
    FINGER1_TIP = "Fixed_Jaw_tip"   # added to URDF manually
    FINGER2_TIP = "Moving_Jaw_tip"  # added to URDF manually
    CAMERA_MOUNT_LINK = "gripper_camera_mount"

    keyframes = dict(
        rest=Keyframe(
            # All qpos must be strictly within URDF joint limits with margin >= 0.05:
            #   joint '1' (idx 0): [-1.9199, 1.9199]  rest=0.00   margin=1.92
            #   joint '2' (idx 1): [-1.7453, 1.7453]  rest=-1.57  margin=0.17
            #   joint '3' (idx 2): [-1.7453, 1.5708]  rest=1.50   margin=0.07  (was 1.5708 = AT limit)
            #   joint '4' (idx 3): [-1.6581, 1.6581]  rest=0.66   margin=1.00
            #   joint '5' (idx 4): [-1.3700, 4.4907]  rest=0.00   margin=1.37
            #   joint '6' (idx 5): [-0.2000, 1.7453]  rest=0.50   margin=0.70  (was -1.1 = BELOW limit)
            qpos=np.array([0, -1.5708, 1.50, 0.66, 0, 0.5]),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
        zero=Keyframe(
            qpos=np.array([0.0] * 6),
            pose=sapien.Pose(q=euler2quat(0, 0, np.pi / 2)),
        ),
    )

    # URDF joint names (numeric "1"-"6")
    arm_joint_names = ["1", "2", "3", "4", "5"]
    gripper_joint_names = ["6"]

    @property
    def _controller_configs(self):
        pd_joint_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            lower=None, upper=None,
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            normalize_action=False,
        )
        pd_joint_delta_pos = PDJointPosControllerConfig(
            [joint.name for joint in self.robot.active_joints],
            [-0.05, -0.05, -0.05, -0.05, -0.05, -0.2],
            [0.05, 0.05, 0.05, 0.05, 0.05, 0.2],
            stiffness=[1e3] * 6,
            damping=[1e2] * 6,
            force_limit=100,
            use_delta=True,
            use_target=False,
        )
        import copy
        pd_joint_target_delta_pos = copy.deepcopy(pd_joint_delta_pos)
        pd_joint_target_delta_pos.use_target = True

        # ------------------------------------------------------------------ #
        # EE-space controllers (for XR0 and similar VLA models)
        # ------------------------------------------------------------------ #
        arm_pd_ee_delta_pose = PDEEPoseControllerConfig(
            joint_names=self.arm_joint_names,
            pos_lower=-0.1,
            pos_upper=0.1,
            rot_lower=-0.1,
            rot_upper=0.1,
            stiffness=[1e3] * 5,
            damping=[1e2] * 5,
            force_limit=100,
            ee_link="Fixed_Jaw_tip",
            urdf_path=self.urdf_path,
            frame="body_translation:body_aligned_body_rotation",
        )
        arm_pd_ee_target_delta_pose = copy.deepcopy(arm_pd_ee_delta_pose)
        arm_pd_ee_target_delta_pose.use_target = True

        gripper_pd_joint_pos = PDJointPosControllerConfig(
            self.gripper_joint_names,
            lower=None,
            upper=None,
            stiffness=[1e3],
            damping=[1e2],
            force_limit=100,
            normalize_action=False,
        )

        return deepcopy_dict(dict(
            pd_joint_delta_pos=pd_joint_delta_pos,
            pd_joint_pos=pd_joint_pos,
            pd_joint_target_delta_pos=pd_joint_target_delta_pos,
            pd_ee_delta_pose=dict(arm=arm_pd_ee_delta_pose, gripper=gripper_pd_joint_pos),
            pd_ee_target_delta_pose=dict(arm=arm_pd_ee_target_delta_pose, gripper=gripper_pd_joint_pos),
        ))

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
        self.finger1_link = self.robot.links_map[self.FINGER1_LINK]
        self.finger2_link = self.robot.links_map[self.FINGER2_LINK]
        self.finger1_tip = self.robot.links_map[self.FINGER1_TIP]
        self.finger2_tip = self.robot.links_map[self.FINGER2_TIP]

    @property
    def tcp_pos(self):
        return (self.finger1_tip.pose.p + self.finger2_tip.pose.p) / 2

    @property
    def tcp_pose(self):
        return Pose.create_from_pq(self.tcp_pos, self.finger1_link.pose.q)

    def is_grasping(self, object, min_force=0.5, max_angle=110):
        from mani_skill.utils.structs.actor import Actor
        from mani_skill.utils import common
        l_contact_forces = self.scene.get_pairwise_contact_forces(self.finger1_link, object)
        r_contact_forces = self.scene.get_pairwise_contact_forces(self.finger2_link, object)
        lforce = torch.linalg.norm(l_contact_forces, axis=1)
        rforce = torch.linalg.norm(r_contact_forces, axis=1)
        ldirection = self.finger1_link.pose.to_transformation_matrix()[..., :3, 1]
        rdirection = -self.finger2_link.pose.to_transformation_matrix()[..., :3, 1]
        langle = common.compute_angle_between(ldirection, l_contact_forces)
        rangle = common.compute_angle_between(rdirection, r_contact_forces)
        lflag = torch.logical_and(lforce >= min_force, torch.rad2deg(langle) <= max_angle)
        rflag = torch.logical_and(rforce >= min_force, torch.rad2deg(rangle) <= max_angle)
        return torch.logical_and(lflag, rflag)

    def is_static(self, threshold=0.2):
        qvel = self.robot.get_qvel()[:, :-1]
        return torch.max(torch.abs(qvel), 1)[0] <= threshold


@register_agent()
class SO101WristCam(SO101):
    """SO101 with wrist camera mounted on the URDF handeye_cam link."""

    uid = "so101_wristcam"

    _WRIST_CAM_LOCAL_P = [0.0, 0.0, 0.0]
    _WRIST_CAM_LOCAL_Q = euler2quat(0, -np.pi / 2, 0)

    @property
    def _sensor_configs(self):
        return [
            CameraConfig(
                uid="wrist_camera",
                pose=sapien.Pose(p=self._WRIST_CAM_LOCAL_P, q=self._WRIST_CAM_LOCAL_Q),
                width=128,
                height=128,
                fov=np.pi / 2,
                near=0.01,
                far=100,
                mount=self.robot.links_map["handeye_cam"],
            )
        ]

    def _after_loading_articulation(self):
        super()._after_loading_articulation()
