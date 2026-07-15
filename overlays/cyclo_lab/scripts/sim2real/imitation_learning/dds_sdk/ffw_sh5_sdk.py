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

import torch
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

_DEFAULT_LIFT_KEYBOARD_STEP = 0.03
_DEFAULT_LIFT_MIN = -0.40
_DEFAULT_LIFT_MAX = 0.0
_SH5_FINGER_HOLD_ALPHA = 0.42
_SH5_FINGER_HOLD_ALPHA_FIRM = 0.68
_SH5_FINGER_HOLD_ALPHA_CARRY = 0.82


class FFWSH5Sdk(FFWSG2Sdk):
    """FFW-SH5 SDK: arms/head/lift from SG2 topics + finger joints on hand topics."""

    TRAJECTORY_QOS = _trajectory_qos()

    def __init__(self, env, mode: str):
        self.left_hand_trajectory_cmd = None
        self.right_hand_trajectory_cmd = None
        self._lift_keyboard_cmd: float | None = None
        self._lift_keyboard_step = _DEFAULT_LIFT_KEYBOARD_STEP
        self._lift_keyboard_min = _DEFAULT_LIFT_MIN
        self._lift_keyboard_max = _DEFAULT_LIFT_MAX
        self._sh5_finger_hold_alpha = _SH5_FINGER_HOLD_ALPHA
        self._sh5_finger_hold_alpha_firm = _SH5_FINGER_HOLD_ALPHA_FIRM
        self._sh5_finger_hold_alpha_carry = _SH5_FINGER_HOLD_ALPHA_CARRY
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
        self._apply_sh5_teleop_config()
        self._keyboard_controls()

    def _apply_sh5_teleop_config(self) -> None:
        """SH5-only teleop options (lift keyboard jog limits)."""
        cfg = getattr(self.env, "cfg", None)
        if cfg is None:
            return
        self._lift_keyboard_step = float(
            getattr(cfg, "teleop_lift_keyboard_step", self._lift_keyboard_step)
        )
        self._lift_keyboard_min = float(
            getattr(cfg, "teleop_lift_min", self._lift_keyboard_min)
        )
        self._lift_keyboard_max = float(
            getattr(cfg, "teleop_lift_max", self._lift_keyboard_max)
        )
        self._sh5_finger_hold_alpha = float(
            getattr(cfg, "teleop_sh5_finger_hold_alpha", self._sh5_finger_hold_alpha)
        )
        self._sh5_finger_hold_alpha_firm = float(
            getattr(cfg, "teleop_sh5_finger_hold_alpha_firm", self._sh5_finger_hold_alpha_firm)
        )
        self._sh5_finger_hold_alpha_carry = float(
            getattr(cfg, "teleop_sh5_finger_hold_alpha_carry", self._sh5_finger_hold_alpha_carry)
        )

    def after_env_step(self) -> None:
        """Keep dexterous finger poses stiff while grasping or during L-motion."""
        super().after_env_step()
        self._hold_sh5_fingers()

    def _should_hold_sh5_fingers(self) -> bool:
        return self._check_box_gripped() or self._is_l_motion_active() or self._carry_box

    def _hold_sh5_fingers(self) -> None:
        if not self._should_hold_sh5_fingers():
            return

        robot = self.env.scene["robot"]
        finger_names = [f"finger_l_joint{i}" for i in range(1, 21)] + [
            f"finger_r_joint{i}" for i in range(1, 21)
        ]
        with self.lock:
            left_cmd = dict(self.left_hand_trajectory_cmd or {})
            right_cmd = dict(self.right_hand_trajectory_cmd or {})

        if self._is_l_motion_active() or self._carry_box:
            alpha = self._sh5_finger_hold_alpha_carry
        else:
            alpha = self._sh5_finger_hold_alpha_firm

        device = self.env.device
        env_ids = torch.tensor([0], device=device)
        joint_pos = robot.data.joint_pos.clone()
        joint_vel = robot.data.joint_vel.clone()
        finger_ids: list[int] = []

        for name in finger_names:
            if name not in robot.joint_names:
                continue
            finger_id = robot.joint_names.index(name)
            finger_ids.append(finger_id)
            if name in left_cmd:
                target = float(left_cmd[name])
            elif name in right_cmd:
                target = float(right_cmd[name])
            else:
                target = float(joint_pos[0, finger_id].item())
            current = float(joint_pos[0, finger_id].item())
            blended = current + alpha * (target - current)
            joint_pos[0, finger_id] = blended
            joint_vel[0, finger_id] = 0.0

        if not finger_ids:
            return

        robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
        pos_target = robot.data.joint_pos_target.clone()
        vel_target = robot.data.joint_vel_target.clone()
        for finger_id in finger_ids:
            pos_target[:, finger_id] = joint_pos[0, finger_id]
            vel_target[:, finger_id] = 0.0
        robot.set_joint_position_target(pos_target)
        robot.set_joint_velocity_target(vel_target)

    def _current_lift_position(self) -> float:
        if self._lift_keyboard_cmd is not None:
            return self._lift_keyboard_cmd
        with self.lock:
            if self.lift_joint_trajectory_cmd and "lift_joint" in self.lift_joint_trajectory_cmd:
                return float(self.lift_joint_trajectory_cmd["lift_joint"])
        robot = self.env.scene["robot"]
        idx = robot.joint_names.index("lift_joint")
        return float(robot.data.joint_pos[0, idx].item())

    def _jog_lift_keyboard(self, delta: float) -> None:
        pos = self._current_lift_position() + delta
        pos = max(self._lift_keyboard_min, min(self._lift_keyboard_max, pos))
        with self.lock:
            self._lift_keyboard_cmd = pos
            self.lift_joint_trajectory_cmd = self.lift_joint_trajectory_cmd or {}
            self.lift_joint_trajectory_cmd["lift_joint"] = pos
        print(f"[Control] lift_joint -> {pos:.3f} m")

    def _on_press(self, key):
        try:
            if hasattr(key, "char") and key.char in ("i", "I"):
                self._jog_lift_keyboard(self._lift_keyboard_step)
                return
            if hasattr(key, "char") and key.char in ("o", "O"):
                self._jog_lift_keyboard(-self._lift_keyboard_step)
                return
        except AttributeError:
            pass
        super()._on_press(key)

    def _clear_teleop_command_state(self) -> None:
        """Clear SG2 caches plus SH5 hand/lift keyboard leftovers."""
        super()._clear_teleop_command_state()
        self._lift_keyboard_cmd = None
        self.left_hand_trajectory_cmd = None
        self.right_hand_trajectory_cmd = None

    def _hand_subscriber_loop(self, reader, attr_name: str, label: str) -> None:
        try:
            while self.running:
                for msg in reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            if not self._accept_teleop_cmds:
                                continue
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
        l_motion = (
            "swerve-drive rotate + forward"
            if self._use_swerve_l_motion and self._swerve_controller is not None
            else "smooth rotate + forward (root teleport)"
        )
        if self.mode == "record":
            print("[N / Right Joystick Button] Save successful episode and proceed to the next one")
            print("[R / Left Joystick Button] Skip failed episode (not saved) and proceed to the next one")
            print("[B / Right Joystick Button] Start recording the current episode")
            print(f"[L] {l_motion} toward the {self._l_motion_label}")
        elif self.mode == "inference":
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start robot control")
            print(f"[L] {l_motion} toward the {self._l_motion_label}")
        print(
            f"[I] Lift up (+{self._lift_keyboard_step:.2f} m)    "
            f"[O] Lift down (-{self._lift_keyboard_step:.2f} m)  "
            f"range [{self._lift_keyboard_min:.2f}, {self._lift_keyboard_max:.2f}] m"
        )
