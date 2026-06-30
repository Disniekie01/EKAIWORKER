# Copyright 2025 ROBOTIS CO., LTD.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DDS teleoperation SDK for FFW-SH5 dexterous-hand recording."""

from __future__ import annotations

import threading
import time

from cyclonedds.core import Policy, Qos
from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_

from dds_sdk.ffw_sg2_sdk import FFWSG2Sdk


def _trajectory_qos() -> Qos:
    """Match sh5_dds_bringup / cyclo_motion_controller BestEffort trajectory QoS."""
    return Qos(
        Policy.Reliability.BestEffort,
        Policy.Durability.Volatile,
        Policy.History.KeepLast(10),
    )

SH5_POLICY_JOINT_NAMES = (
    [f"arm_l_joint{i}" for i in range(1, 8)]
    + [f"finger_l_joint{i}" for i in range(1, 21)]
    + [f"arm_r_joint{i}" for i in range(1, 8)]
    + [f"finger_r_joint{i}" for i in range(1, 21)]
    + ["head_joint1", "head_joint2", "lift_joint"]
)

LEFT_HAND_TOPIC = "/leader/joint_trajectory_command_broadcaster_left_hand/joint_trajectory"
RIGHT_HAND_TOPIC = "/leader/joint_trajectory_command_broadcaster_right_hand/joint_trajectory"


class FFWSH5Sdk(FFWSG2Sdk):
    """FFW-SH5 SDK: arms/head/lift from SG2 topics + finger joints on hand topics."""

    TRAJECTORY_QOS = _trajectory_qos()

    def __init__(self, env, mode: str):
        self.left_hand_trajectory_cmd = None
        self.right_hand_trajectory_cmd = None
        super().__init__(env, mode)
        # IMPORTANT: the action tensor is consumed by the ActionManager in term
        # registration order, and within each JointPositionAction the values are
        # applied in the articulation's joint-index order (find_joints with
        # preserve_order=False), NOT the numeric order of the regex. With 20
        # finger joints per hand, using numeric order scrambles the fingers.
        # Derive the true order straight from the action manager so get_action's
        # tensor lines up exactly with how the env applies it.
        self.joint_names = self._resolve_action_joint_order()

        from robotis_dds_python.tools.topic_manager import TopicManager

        topic_manager = TopicManager(domain_id=self.domain_id)
        trajectory_qos = self.TRAJECTORY_QOS
        self.left_hand_joint_trajectory_reader = topic_manager.topic_reader(
            topic_name=LEFT_HAND_TOPIC,
            topic_type=JointTrajectory_,
            qos=trajectory_qos,
        )
        self.right_hand_joint_trajectory_reader = topic_manager.topic_reader(
            topic_name=RIGHT_HAND_TOPIC,
            topic_type=JointTrajectory_,
            qos=trajectory_qos,
        )

        self.left_hand_thread = threading.Thread(target=self._left_hand_subscriber_loop, daemon=True)
        self.right_hand_thread = threading.Thread(target=self._right_hand_subscriber_loop, daemon=True)
        self.left_hand_thread.start()
        self.right_hand_thread.start()
        print(f"[SH5] Hand DDS topics: {LEFT_HAND_TOPIC}, {RIGHT_HAND_TOPIC}")
        self._keyboard_controls()

    def _resolve_action_joint_order(self) -> list:
        """Build the flat joint-name order the ActionManager actually applies.

        Falls back to the static SH5_POLICY_JOINT_NAMES if introspection fails.
        """
        try:
            am = self.env.action_manager
            order: list[str] = []
            for term_name in am.active_terms:
                term = am.get_term(term_name)
                joint_names = getattr(term, "_joint_names", None)
                if not joint_names:
                    raise RuntimeError(f"action term '{term_name}' exposes no joint names")
                order.extend(joint_names)
            if not order:
                raise RuntimeError("action manager produced empty joint order")
            print(f"[SH5] Action joint order resolved from action manager ({len(order)} joints).")
            return order
        except Exception as exc:  # pragma: no cover - defensive
            print(f"[SH5] WARNING: falling back to static joint order ({exc}).")
            return list(SH5_POLICY_JOINT_NAMES)

    def _hand_subscriber_loop(self, reader, attr_name: str, label: str) -> None:
        try:
            while self.running:
                for msg in reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            current = getattr(self, attr_name) or {}
                            current.update(joint_dict)
                            setattr(self, attr_name, current)
                time.sleep(0.001)
        except Exception as e:
            print(f"{label} hand subscriber thread exception:", e)
        finally:
            try:
                reader.Close()
            except Exception as e:
                print(f"Error closing {label} hand subscriber: {e}")
            print(f"{label} hand subscriber closed")

    def _left_hand_subscriber_loop(self):
        self._hand_subscriber_loop(
            self.left_hand_joint_trajectory_reader,
            "left_hand_trajectory_cmd",
            "Left",
        )

    def _right_hand_subscriber_loop(self):
        self._hand_subscriber_loop(
            self.right_hand_joint_trajectory_reader,
            "right_hand_trajectory_cmd",
            "Right",
        )

    def _get_device_state(self):
        joint_state = super()._get_device_state()
        with self.lock:
            if self.left_hand_trajectory_cmd:
                joint_state.update(self.left_hand_trajectory_cmd)
            if self.right_hand_trajectory_cmd:
                joint_state.update(self.right_hand_trajectory_cmd)
        return joint_state

    def shutdown(self):
        self.running = False
        self.left_hand_thread.join()
        self.right_hand_thread.join()
        for reader in (
            self.left_hand_joint_trajectory_reader,
            self.right_hand_joint_trajectory_reader,
        ):
            try:
                reader.Close()
            except Exception:
                pass
        super().shutdown()
        print("FFWSH5Sdk shutdown complete")

    def _keyboard_controls(self):
        print("\n[Control] Press keys to control the FFW-SH5 robot:")
        if self.mode == "record":
            print("[N / Right Joystick Button] Save successful episode and proceed to the next one")
            print("[R / Left Joystick Button] Skip failed episode (not saved) and proceed to the next one")
            print("[B / Right Joystick Button] Start recording the current episode")
            print(f"[L] Smoothly rotate robot to face the {self._l_motion_label}, then move forward")
        elif self.mode == "inference":
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start robot control")
            print(f"[L] Smoothly rotate robot to face the {self._l_motion_label}, then move forward")
