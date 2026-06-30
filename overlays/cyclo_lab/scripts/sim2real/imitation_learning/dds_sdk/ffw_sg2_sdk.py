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
#
# Author: Taehyeong Kim

import os
import math
import threading
import torch
import cv2
import time
from pynput.keyboard import Listener
from collections.abc import Callable
from datetime import datetime

import isaaclab.utils.math as math_utils

from robotis_dds_python.idl.trajectory_msgs.msg import JointTrajectory_
from robotis_dds_python.idl.sensor_msgs.msg import JointState_
from robotis_dds_python.idl.sensor_msgs.msg import CompressedImage_
from robotis_dds_python.idl.std_msgs.msg import Header_
from robotis_dds_python.idl.std_msgs.msg import String_
from robotis_dds_python.idl.builtin_interfaces.msg import Time_

from robotis_dds_python.tools.topic_manager import TopicManager


class FFWSG2Sdk:
    """FFWSG2Sdk class for DDS teleoperation and publishing humanoid robot state/images."""

    # Subclasses (e.g. FFWSH5Sdk) override with BestEffort to match cyclo_motion_controller.
    TRAJECTORY_QOS = None

    def __init__(self, env, mode: str):
        self.env = env
        self.mode = mode  # 'record' or 'inference'
        self.running = True
        self.domain_id = int(os.getenv("ROS_DOMAIN_ID", 0))
        self.left_arm_trajectory_cmd = None
        self.right_arm_trajectory_cmd = None
        self.head_joint_trajectory_cmd = None
        self.lift_joint_trajectory_cmd = None
        self._started = False
        self._reset_state = False
        self._additional_callbacks = {}
        self._first_episode = True  # Track if this is the first episode
        self._episode_phase = "idle"  # Current state: "idle" (waiting) or "recording" (active episode)
        self._face_left_yaw = math.pi / 2.0
        self._l_motion_label = "left table"
        self._pending_face_left = False
        self._rotation_active = False
        self._rotation_start_yaw = 0.0
        self._rotation_target_yaw = 0.0
        self._rotation_start_time = 0.0
        self._rotation_duration_s = 3.0
        self._forward_active = False
        self._forward_start_pos = None
        self._forward_end_pos = None
        self._forward_yaw = 0.0
        self._forward_start_time = 0.0
        self._forward_duration_s = 2.0
        self._forward_distance_m = 0.30
        self._pending_reset_pose = False
        self._home_root_pose = None  # (1, 7) tensor captured on first publish
        # While the L (face-left) motion teleports the robot root, the box is
        # carried rigidly with it via this relative transform so it does not
        # slip out of the grippers.
        self._carry_box = False
        self._box_rel_pos = None  # (1, 3) box pos in root frame
        self._box_rel_quat = None  # (1, 4) box quat in root frame
        self._box_asset_name = "cardboard_box"
        # Swerve-drive L-motion (SH5): uses cmd_vel-style base motion instead of root teleport.
        self._use_swerve_l_motion = False
        self._swerve_controller = None
        self._swerve_phase = None
        self._swerve_steering_joint_ids: list[int] = []
        self._swerve_wheel_joint_ids: list[int] = []
        self._swerve_forward_start_xy = None
        self._swerve_forward_dir_xy = None
        self._last_swerve_update_time = time.monotonic()
        self._swerve_yaw_tol = 0.08
        self._swerve_max_angular_z = 0.6
        self._swerve_max_linear_x = 0.25
        self.lock = threading.Lock()  # Protect shared state

        # Initialize current joint state - will be updated only when commands are received
        self.current_joint_state = {}

        # Define joint names for FFW_SG2 humanoid robot
        self.joint_names = [
            "arm_l_joint1", "arm_l_joint2", "arm_l_joint3", "arm_l_joint4", "arm_l_joint5", "arm_l_joint6", "arm_l_joint7", "gripper_l_joint1",
            "arm_r_joint1", "arm_r_joint2", "arm_r_joint3", "arm_r_joint4", "arm_r_joint5", "arm_r_joint6", "arm_r_joint7", "gripper_r_joint1",
            "lift_joint", "head_joint1", "head_joint2"
        ]

        # DDS Topic Manager
        topic_manager = TopicManager(domain_id=self.domain_id)
        trajectory_qos = self.TRAJECTORY_QOS

        # Subscribers for both arms
        self.left_arm_joint_trajectory_reader = topic_manager.topic_reader(
            topic_name="/leader/joint_trajectory_command_broadcaster_left/joint_trajectory",
            topic_type=JointTrajectory_,
            qos=trajectory_qos,
        )
        self.right_arm_joint_trajectory_reader = topic_manager.topic_reader(
            topic_name="/leader/joint_trajectory_command_broadcaster_right/joint_trajectory",
            topic_type=JointTrajectory_,
            qos=trajectory_qos,
        )
        self.head_joint_trajectory_reader = topic_manager.topic_reader(
            topic_name="/leader/joystick_controller_left/joint_trajectory",
            topic_type=JointTrajectory_,
            qos=trajectory_qos,
        )
        self.lift_joint_trajectory_reader = topic_manager.topic_reader(
            topic_name="/leader/joystick_controller_right/joint_trajectory",
            topic_type=JointTrajectory_,
            qos=trajectory_qos,
        )
        self.joystick_track_trigger_reader = topic_manager.topic_reader(
            topic_name="/leader/joystick_controller/tact_trigger",
            topic_type=String_
        )

        # Publishers
        self.joint_state_writer = topic_manager.topic_writer(
            topic_name="joint_states",
            topic_type=JointState_
        )
        self.head_cam_writer = topic_manager.topic_writer(
            topic_name="/zed/zed_node/left/image_rect_color/compressed",
            topic_type=CompressedImage_
        )
        self.right_wrist_cam_writer = topic_manager.topic_writer(
            topic_name="/camera_right/camera_right/color/image_rect_raw/compressed",
            topic_type=CompressedImage_
        )
        self.left_wrist_cam_writer = topic_manager.topic_writer(
            topic_name="/camera_left/camera_left/color/image_rect_raw/compressed",
            topic_type=CompressedImage_
        )

        # Start subscriber threads for both arms
        self.left_thread = threading.Thread(target=self._left_arm_subscriber_loop, daemon=True)
        self.right_thread = threading.Thread(target=self._right_arm_subscriber_loop, daemon=True)
        self.lift_thread = threading.Thread(target=self._lift_joint_subscriber_loop, daemon=True)
        self.head_thread = threading.Thread(target=self._head_joint_subscriber_loop, daemon=True)
        self.joystick_thread = threading.Thread(target=self._joystick_subscriber_loop, daemon=True)

        self.left_thread.start()
        self.right_thread.start()
        self.lift_thread.start()
        self.head_thread.start()
        self.joystick_thread.start()

        # Keyboard listener
        self.listener = Listener(on_press=self._on_press)
        self.listener.start()

        self._apply_env_teleop_config()
        self._keyboard_controls()

    def _apply_env_teleop_config(self) -> None:
        """Read optional per-task L-motion settings from ``env.cfg``."""
        cfg = getattr(self.env, "cfg", None)
        if cfg is None:
            return
        self._face_left_yaw = float(getattr(cfg, "teleop_l_yaw", self._face_left_yaw))
        self._forward_distance_m = float(getattr(cfg, "teleop_l_forward_m", self._forward_distance_m))
        self._forward_duration_s = float(
            getattr(cfg, "teleop_l_forward_duration_s", self._forward_duration_s)
        )
        self._rotation_duration_s = float(
            getattr(cfg, "teleop_l_rotation_duration_s", self._rotation_duration_s)
        )
        self._l_motion_label = str(getattr(cfg, "teleop_l_target_label", self._l_motion_label))
        self._use_swerve_l_motion = bool(getattr(cfg, "teleop_l_use_swerve", self._use_swerve_l_motion))

    # ----------------------
    # Keyboard controls
    # ----------------------
    def _keyboard_controls(self):
        print("\n[Control] Press keys to control the FFW_SG2 robot:")
        if self.mode == 'record':
            print("[N / Right Joystick Button] Save successful episode and proceed to the next one")
            print("[R / Left Joystick Button] Skip failed episode (not saved) and proceed to the next one")
            print("[B / Right Joystick Button] Start recording the current episode")
            print(f"[L] Smoothly rotate robot to face the {self._l_motion_label}, then move forward")
        elif self.mode == 'inference':
            print("[R] Skip failed episode (not saved) and proceed to the next one")
            print("[B] Start robot control")
            print(f"[L] Smoothly rotate robot to face the {self._l_motion_label}, then move forward")

    def _on_press(self, key):
        try:
            if hasattr(key, "char") and key.char == "l":
                with self.lock:
                    self._pending_face_left = True
                return
            if self.mode == 'record':
                if key.char == 'b':
                    self._started = True
                    self._reset_state = False
                    # Update episode tracking when manually starting
                    if self._first_episode:
                        self._first_episode = False
                    self._episode_phase = "recording"  # Now recording
                elif key.char == 'r':
                    self._started = False
                    self._reset_state = True
                    self._request_pose_reset()
                    self._call_callback("R")
                    # If resetting while recording before first episode was saved, go back to first episode state
                    if self._episode_phase == "recording" and not self._first_episode:
                        self._first_episode = True
                        self._episode_phase = "idle"
                elif key.char == 'n':
                    self._started = False
                    self._reset_state = True
                    self._call_callback("N")
                    # After saving, go back to idle state
                    self._episode_phase = "idle"
            elif self.mode == 'inference':
                if key.char == 'b':
                    self._started = True
                    self._reset_state = False
                elif key.char == 'r':
                    self._started = False
                    self._reset_state = True
                    self._request_pose_reset()
                    self._call_callback("R")
        except AttributeError:
            pass

    def _call_callback(self, key):
        if key in self._additional_callbacks:
            self._additional_callbacks[key]()

    def _request_pose_reset(self):
        """Flag a robot root-pose reset and cancel any in-progress L motion.

        Sets booleans only (no sim access); the actual write to sim happens on
        the sim thread in ``publish_observations``. Safe to call whether or not
        ``self.lock`` is already held.
        """
        self._pending_reset_pose = True
        self._rotation_active = False
        self._forward_active = False
        self._carry_box = False

    # ----------------------
    # Subscriber loops for both arms
    # ----------------------
    def _left_arm_subscriber_loop(self):
        """Continuously read joint trajectory commands from the leader."""
        try:
            while self.running:
                for msg in self.left_arm_joint_trajectory_reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            # Merge instead of replace: arm IK and gripper commands
                            # share this topic but arrive as separate partial
                            # messages, so replacing would erase the gripper.
                            self.left_arm_trajectory_cmd = self.left_arm_trajectory_cmd or {}
                            self.left_arm_trajectory_cmd.update(joint_dict)
                time.sleep(0.001)  # 1ms sleep to reduce CPU load
        except Exception as e:
            print("Left arm subscriber thread exception:", e)
        finally:
            try:
                self.left_arm_joint_trajectory_reader.Close()
            except Exception as e:
                print(f"Error closing left arm subscriber: {e}")
            print("Left arm subscriber closed")

    def _right_arm_subscriber_loop(self):
        """Continuously read right arm joint trajectory commands from the leader."""
        try:
            while self.running:
                for msg in self.right_arm_joint_trajectory_reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            # Merge instead of replace: arm IK and gripper commands
                            # share this topic but arrive as separate partial
                            # messages, so replacing would erase the gripper.
                            self.right_arm_trajectory_cmd = self.right_arm_trajectory_cmd or {}
                            self.right_arm_trajectory_cmd.update(joint_dict)
                time.sleep(0.001)  # 1ms sleep to reduce CPU load
        except Exception as e:
            print("Right arm subscriber thread exception:", e)
        finally:
            try:
                self.right_arm_joint_trajectory_reader.Close()
            except:
                pass
            print("Right arm subscriber closed")

    def _lift_joint_subscriber_loop(self):
        """Continuously read lift joint trajectory commands from the leader."""
        try:
            while self.running:
                for msg in self.lift_joint_trajectory_reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            # Update only the lift joint command
                            self.lift_joint_trajectory_cmd = self.lift_joint_trajectory_cmd or {}
                            self.lift_joint_trajectory_cmd.update(joint_dict)
                time.sleep(0.001)  # 1ms sleep to reduce CPU load
        except Exception as e:
            print("Lift joint subscriber thread exception:", e)
        finally:
            try:
                self.lift_joint_trajectory_reader.Close()
            except:
                pass
            print("Lift joint subscriber closed")

    def _head_joint_subscriber_loop(self):
        """Continuously read head joint trajectory commands from the leader."""
        try:
            while self.running:
                for msg in self.head_joint_trajectory_reader.take_iter():
                    if msg and msg.points:
                        joint_dict = dict(zip(msg.joint_names, msg.points[-1].positions))
                        with self.lock:
                            # Update only the head joint commands
                            self.head_joint_trajectory_cmd = self.head_joint_trajectory_cmd or {}
                            self.head_joint_trajectory_cmd.update(joint_dict)
                time.sleep(0.001)  # 1ms sleep to reduce CPU load
        except Exception as e:
            print("Head joint subscriber thread exception:", e)
        finally:
            try:
                self.head_joint_trajectory_reader.Close()
            except:
                pass
            print("Head joint subscriber closed")

    def _joystick_subscriber_loop(self):
        """Continuously read joystick track trigger commands from the leader."""
        try:
            while self.running:
                for msg in self.joystick_track_trigger_reader.take_iter():
                    # Only process joystick triggers in record mode
                    if self.mode != 'record':
                        continue

                    joystick_trigger = msg.data
                    if joystick_trigger == 'right':
                        with self.lock:
                            if self._first_episode:
                                # First episode: only start recording
                                self._started = True
                                self._reset_state = False
                                self._first_episode = False
                                self._episode_phase = "recording"  # Now recording
                            elif self._episode_phase == "recording":
                                # Currently recording: save episode and go back to idle
                                self._started = False
                                self._reset_state = True
                                self._call_callback("N")
                                self._episode_phase = "idle"  # Now idle, waiting for next start
                            elif self._episode_phase == "idle":
                                # Currently idle: start new episode
                                self._started = True
                                self._reset_state = False
                                self._episode_phase = "recording"  # Now recording
                    elif joystick_trigger == 'left':
                        with self.lock:
                            # Reset current episode (don't save)
                            self._started = False
                            self._reset_state = True
                            self._request_pose_reset()
                            self._call_callback("R")
                            # If resetting while recording before first episode was saved, go back to first episode state
                            if self._episode_phase == "recording" and not self._first_episode:
                                # We started recording but haven't saved yet - reset to first episode
                                self._first_episode = True
                                self._episode_phase = "idle"
                time.sleep(0.001)  # 1ms sleep to reduce CPU load
        except Exception as e:
            print("Joystick subscriber thread exception:", e)
        finally:
            try:
                self.joystick_track_trigger_reader.Close()
            except:
                pass
            print("Joystick subscriber closed")

    # ----------------------
    # Publishers
    # ----------------------
    def _publish_joint_states(self):
        """Publish current joint states over DDS."""
        now = datetime.now()
        stamp = Time_(sec=int(now.timestamp()), nanosec=now.microsecond * 1000)
        header = Header_(stamp=stamp, frame_id="base_link")

        obs_joint_name = self.env.scene["robot"].data.joint_names
        all_positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()
        all_velocities = self.env.scene["robot"].data.joint_vel.squeeze(0).tolist()
        all_efforts = [0.0] * len(all_positions)

        # Flatten nested lists if necessary
        if isinstance(all_positions[0], list):
            all_positions = [p for sub in all_positions for p in sub]
        if isinstance(all_velocities[0], list):
            all_velocities = [v for sub in all_velocities for v in sub]

        # Get indices of the joints we care about
        indices = [obs_joint_name.index(name) for name in self.joint_names if name in obs_joint_name]

        positions = [all_positions[i] for i in indices]
        velocities = [all_velocities[i] for i in indices]
        efforts = [all_efforts[i] for i in indices]

        joint_state = JointState_(
            header=header,
            name=[self.joint_names[i] for i in range(len(indices))],
            position=positions,
            velocity=velocities,
            effort=efforts
        )

        try:
            self.joint_state_writer.write(joint_state)
        except Exception as e:
            print("[Writer] write error:", e)

    def _publish_camera(self, cam_name: str):
        """Publish camera image as DDS compressed image."""
        try:
            cam_data = self.env.scene[cam_name].data
            img = cam_data.output['rgb'][0].cpu().numpy()  # Convert tensor to numpy (RGB format)
            
            # Convert RGB to BGR for OpenCV encoding
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            _, buffer = cv2.imencode('.jpg', img_bgr)
            jpeg_bytes = buffer.tobytes()

            now = datetime.now()
            stamp = Time_(sec=int(now.timestamp()), nanosec=now.microsecond * 1000)
            header = Header_(stamp=stamp, frame_id="camera_frame")

            msg = CompressedImage_(header=header, format="jpeg", data=jpeg_bytes)
            
            # Map camera names to publishers for FFW_SG2
            if cam_name == "cam_wrist_right":
                self.right_wrist_cam_writer.write(msg)
            elif cam_name == "cam_wrist_left":
                self.left_wrist_cam_writer.write(msg)
            elif cam_name == "cam_head":
                self.head_cam_writer.write(msg)
        except Exception as e:
            print(f"Camera publish error for {cam_name}:", e)

    def _compute_action_state(self):
        """Compute current action dictionary based on keyboard input and subscriber."""
        state = {'reset': self._reset_state, 'started': self._started}
        if state['reset']:
            self._reset_state = False
            return state
        state['joint_state'] = self._get_device_state()
        return state

    def _get_device_state(self):
        """Return latest joint positions, starting with current robot state and updating with received commands."""
        with self.lock:
            # Start with current robot joint positions
            obs_joint_name = self.env.scene["robot"].data.joint_names
            all_positions = self.env.scene["robot"].data.joint_pos.squeeze(0).tolist()
            
            # Flatten nested lists if necessary
            if isinstance(all_positions[0], list):
                all_positions = [p for sub in all_positions for p in sub]

            # Build joint state from current robot state
            joint_state = {}
            for name in self.joint_names:
                if name in obs_joint_name:
                    idx = obs_joint_name.index(name)
                    joint_state[name] = all_positions[idx]
                else:
                    joint_state[name] = 0.0  # Fallback only if joint not found in robot
            
            # Update with left arm commands if available
            if self.left_arm_trajectory_cmd:
                joint_state.update(self.left_arm_trajectory_cmd)
            
            # Update with right arm commands if available
            if self.right_arm_trajectory_cmd:
                joint_state.update(self.right_arm_trajectory_cmd)

            if self.head_joint_trajectory_cmd:
                joint_state.update(self.head_joint_trajectory_cmd)

            if self.lift_joint_trajectory_cmd:
                joint_state.update(self.lift_joint_trajectory_cmd)
            
            return joint_state

    def get_action(self):
        """Return action tensor for robot control."""
        action = self._compute_action_state()
        if action['reset']:
            return {"reset": True}
        if not action['started']:
            return None

        joint_state = action['joint_state']
        positions = [joint_state.get(name, 0.0) for name in self.joint_names]
        return torch.tensor(positions, device=self.env.device, dtype=torch.float32).unsqueeze(0)

    def publish_observations(self):
        """Publish joint states and camera images."""
        # Capture the robot's home root pose once, before any L motion edits it.
        if self._home_root_pose is None:
            robot = self.env.scene["robot"]
            self._home_root_pose = robot.data.root_state_w[0:1, 0:7].clone()

        with self.lock:
            pending_face_left = self._pending_face_left
            self._pending_face_left = False
            pending_reset_pose = self._pending_reset_pose
            self._pending_reset_pose = False
        # Pose reset (key 'R') takes priority over a queued face-left request.
        if pending_reset_pose:
            self._restore_home_pose()
        elif pending_face_left:
            self.face_left_table()
        if self._use_swerve_l_motion and self._swerve_controller is not None:
            self._step_swerve_l_motion()
        else:
            self._step_yaw_rotation()
            self._step_forward_motion()
        self._publish_joint_states()
        self._publish_camera("cam_head")
        # self._publish_camera("cam_wrist_right")
        # self._publish_camera("cam_wrist_left")

    # ----------------------
    # Utility
    # ----------------------
    def shutdown(self):
        """Stop threads and close DDS publishers/subscribers."""
        self.running = False
        self.left_thread.join()
        self.right_thread.join()
        self.lift_thread.join()
        self.head_thread.join()
        self.joystick_thread.join()
        
        for obj in [self.left_arm_joint_trajectory_reader, self.right_arm_joint_trajectory_reader,
                    self.joint_state_writer, self.head_cam_writer, 
                    self.right_wrist_cam_writer, self.left_wrist_cam_writer]:
            try:
                obj.Close()
            except:
                pass
        print("FFWSG2Sdk shutdown complete")

    def reset(self):
        self._reset_state = False
        self._rotation_active = False
        self._forward_active = False
        self._pending_reset_pose = False
        self._carry_box = False
        self._swerve_phase = None

    def _restore_home_pose(self):
        """Restore the robot root pose (position + orientation) to its home/start
        pose, undoing any rotation/translation applied by the L (face-left) action."""
        self._swerve_phase = None
        if self._swerve_controller is not None:
            self._apply_swerve_cmd_vel(0.0, 0.0, 0.0)
        if self._home_root_pose is None:
            return
        robot = self.env.scene["robot"]
        device = self.env.device
        env_ids = torch.tensor([0], device=device)
        root_pose = self._home_root_pose.to(device)
        robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)
        print("[Control] Robot root pose reset to home")

    def face_left_table(self):
        """Begin a smooth rotation toward the task target, then move forward."""
        if self._use_swerve_l_motion and self._swerve_controller is not None:
            self._forward_active = False
            self._rotation_active = False
            self._swerve_phase = "rotate"
            self._rotation_target_yaw = self._face_left_yaw
            self._swerve_forward_start_xy = None
            self._swerve_forward_dir_xy = None
            print(
                f"[Control] Swerve: rotating to face the {self._l_motion_label}, "
                f"then driving forward {self._forward_distance_m:.2f} m"
            )
            return
        self._forward_active = False
        self._capture_box_carry()
        self._begin_yaw_rotation(self._face_left_yaw)
        print(f"[Control] Rotating robot to face the {self._l_motion_label}")

    def _capture_box_carry(self):
        """Record the box pose relative to the robot root so it can be carried
        rigidly with the robot during the L motion (prevents it slipping out)."""
        try:
            robot = self.env.scene["robot"]
            box = self.env.scene[self._box_asset_name]
        except KeyError:
            self._carry_box = False
            return

        root_pos = robot.data.root_pos_w[0:1].clone()
        root_quat = robot.data.root_quat_w[0:1].clone()
        box_pos = box.data.root_pos_w[0:1].clone()
        box_quat = box.data.root_quat_w[0:1].clone()

        inv_root_quat = math_utils.quat_inv(root_quat)
        self._box_rel_pos = math_utils.quat_apply(inv_root_quat, box_pos - root_pos)
        self._box_rel_quat = math_utils.quat_mul(inv_root_quat, box_quat)
        self._carry_box = True

    def _carry_box_with_root(self):
        """Re-impose the captured box->root relative transform on the box."""
        if not self._carry_box or self._box_rel_pos is None:
            return
        try:
            robot = self.env.scene["robot"]
            box = self.env.scene[self._box_asset_name]
        except KeyError:
            return

        device = self.env.device
        env_ids = torch.tensor([0], device=device)
        root_pos = robot.data.root_pos_w[0:1]
        root_quat = robot.data.root_quat_w[0:1]
        box_pos = root_pos + math_utils.quat_apply(root_quat, self._box_rel_pos.to(device))
        box_quat = math_utils.quat_mul(root_quat, self._box_rel_quat.to(device))
        box_pose = torch.cat([box_pos, box_quat], dim=-1)
        box.write_root_pose_to_sim(box_pose, env_ids=env_ids)
        box.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)

    def _get_robot_yaw(self) -> float:
        robot = self.env.scene["robot"]
        _, _, yaw = math_utils.euler_xyz_from_quat(robot.data.root_quat_w[0:1])
        return float(yaw.item())

    def _begin_yaw_rotation(self, target_yaw: float):
        self._rotation_start_yaw = self._get_robot_yaw()
        self._rotation_target_yaw = target_yaw
        self._rotation_start_time = time.monotonic()
        self._rotation_active = True

    def _begin_forward_motion(self):
        robot = self.env.scene["robot"]
        device = self.env.device
        self._forward_start_pos = robot.data.root_pos_w[0:1].clone()
        quat = robot.data.root_quat_w[0:1].clone()
        offset = torch.tensor(
            [[self._forward_distance_m, 0.0, 0.0]],
            device=device,
        )
        self._forward_end_pos = self._forward_start_pos + math_utils.quat_apply(quat, offset)
        self._forward_yaw = self._get_robot_yaw()
        self._forward_start_time = time.monotonic()
        self._forward_active = True

    def _smoothstep(self, alpha: float) -> float:
        return alpha * alpha * (3.0 - 2.0 * alpha)

    def _lerp_yaw(self, start: float, end: float, alpha: float) -> float:
        delta = (end - start + math.pi) % (2.0 * math.pi) - math.pi
        return start + alpha * delta

    def _set_robot_pose(self, pos: torch.Tensor, yaw: float):
        robot = self.env.scene["robot"]
        device = self.env.device
        env_ids = torch.tensor([0], device=device)
        quat = math_utils.quat_from_euler_xyz(
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.tensor([yaw], device=device),
        )
        root_pose = torch.cat([pos, quat], dim=-1)
        robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)

    def _set_robot_yaw(self, yaw: float):
        robot = self.env.scene["robot"]
        self._set_robot_pose(robot.data.root_pos_w[0:1].clone(), yaw)

    def _step_yaw_rotation(self):
        if not self._rotation_active:
            return

        elapsed = time.monotonic() - self._rotation_start_time
        alpha = self._smoothstep(min(elapsed / self._rotation_duration_s, 1.0))
        yaw = self._lerp_yaw(self._rotation_start_yaw, self._rotation_target_yaw, alpha)
        self._set_robot_yaw(yaw)
        self._carry_box_with_root()

        if alpha >= 1.0:
            self._rotation_active = False
            self._begin_forward_motion()
            print(f"[Control] Robot finished rotating; moving forward toward the {self._l_motion_label}")

    def _step_forward_motion(self):
        if not self._forward_active:
            return

        elapsed = time.monotonic() - self._forward_start_time
        alpha = self._smoothstep(min(elapsed / self._forward_duration_s, 1.0))
        pos = self._forward_start_pos + alpha * (self._forward_end_pos - self._forward_start_pos)
        self._set_robot_pose(pos, self._forward_yaw)
        self._carry_box_with_root()

        if alpha >= 1.0:
            self._forward_active = False
            self._carry_box = False
            print(f"[Control] Robot finished moving forward toward the {self._l_motion_label}")

    def _init_swerve_drive(self) -> None:
        """Set up the SH5 swerve controller (same stack as sh5_dds_bringup)."""
        import sys
        from pathlib import Path

        bringup_dir = Path(__file__).resolve().parents[2] / "bringup"
        if str(bringup_dir) not in sys.path:
            sys.path.insert(0, str(bringup_dir))

        from common import robotis_config as bringup_cfg
        from common.swerve_drive import SwerveDriveController, SwerveModule
        from cyclo_lab.assets.robots.FFW_SH5 import (
            SH5_SWERVE_MODULE_ANGLE_OFFSETS,
            SH5_SWERVE_MODULE_X_OFFSETS,
            SH5_SWERVE_MODULE_Y_OFFSETS,
            SH5_SWERVE_STEERING_JOINTS,
            SH5_SWERVE_WHEEL_JOINTS,
            SH5_SWERVE_WHEEL_RADIUS,
        )

        robot = self.env.scene["robot"]
        joint_names = list(robot.data.joint_names)
        name_to_id = {name: idx for idx, name in enumerate(joint_names)}

        modules = [
            SwerveModule(
                steering_joint=steering_joint,
                wheel_joint=wheel_joint,
                x_offset=SH5_SWERVE_MODULE_X_OFFSETS[index],
                y_offset=SH5_SWERVE_MODULE_Y_OFFSETS[index],
                angle_offset=SH5_SWERVE_MODULE_ANGLE_OFFSETS[index],
                steering_limit_lower=bringup_cfg.AI_WORKER_SWERVE_STEERING_LIMIT_LOWER,
                steering_limit_upper=bringup_cfg.AI_WORKER_SWERVE_STEERING_LIMIT_UPPER,
                wheel_speed_limit_lower=bringup_cfg.AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_LOWER,
                wheel_speed_limit_upper=bringup_cfg.AI_WORKER_SWERVE_WHEEL_SPEED_LIMIT_UPPER,
            )
            for index, (steering_joint, wheel_joint) in enumerate(
                zip(SH5_SWERVE_STEERING_JOINTS, SH5_SWERVE_WHEEL_JOINTS)
            )
        ]

        missing = [
            joint
            for module in modules
            for joint in (module.steering_joint, module.wheel_joint)
            if joint not in name_to_id
        ]
        if missing:
            print(f"[SH5] Swerve joints missing from articulation {missing}; L-motion uses root teleport.")
            self._use_swerve_l_motion = False
            return

        self._swerve_steering_joint_ids = [name_to_id[m.steering_joint] for m in modules]
        self._swerve_wheel_joint_ids = [name_to_id[m.wheel_joint] for m in modules]
        self._swerve_controller = SwerveDriveController(modules, SH5_SWERVE_WHEEL_RADIUS)
        self._use_swerve_l_motion = True
        print("[SH5] Swerve-drive L-motion enabled (cmd_vel-style base control).")

    def _yaw_error(self, target_yaw: float) -> float:
        current = self._get_robot_yaw()
        return (target_yaw - current + math.pi) % (2.0 * math.pi) - math.pi

    def _apply_swerve_cmd_vel(self, linear_x: float, linear_y: float, angular_z: float) -> None:
        if self._swerve_controller is None:
            return

        robot = self.env.scene["robot"]
        now = time.monotonic()
        dt = max(now - self._last_swerve_update_time, 1.0e-3)
        self._last_swerve_update_time = now

        current_steering = [
            float(v)
            for v in robot.data.joint_pos[0, self._swerve_steering_joint_ids].detach().cpu().tolist()
        ]
        current_wheel_velocities = [
            float(v)
            for v in robot.data.joint_vel[0, self._swerve_wheel_joint_ids].detach().cpu().tolist()
        ]

        module_commands = self._swerve_controller.compute_commands(
            linear_x,
            linear_y,
            angular_z,
            current_steering_positions=current_steering,
            current_wheel_velocities=current_wheel_velocities,
            dt=dt,
        )

        position_target = robot.data.joint_pos_target.clone()
        velocity_target = robot.data.joint_vel_target.clone()
        for module_command, steering_id, wheel_id in zip(
            module_commands,
            self._swerve_steering_joint_ids,
            self._swerve_wheel_joint_ids,
        ):
            position_target[:, steering_id] = module_command.steering_position
            velocity_target[:, wheel_id] = module_command.wheel_velocity
        robot.set_joint_position_target(position_target)
        robot.set_joint_velocity_target(velocity_target)

        # The recording env spawns the SH5 with gravity disabled, so the wheels
        # spin without ground traction and the base would never move from
        # friction alone. Integrate the commanded body twist into the root pose
        # so the base actually drives, while the steering/wheel joint targets
        # above still produce realistic swerve values in the recorded dataset.
        if (
            abs(linear_x) > 1.0e-4
            or abs(linear_y) > 1.0e-4
            or abs(angular_z) > 1.0e-4
        ):
            self._integrate_swerve_root(linear_x, linear_y, angular_z, dt)

    def _integrate_swerve_root(
        self, linear_x: float, linear_y: float, angular_z: float, dt: float
    ) -> None:
        robot = self.env.scene["robot"]
        device = self.env.device
        env_ids = torch.tensor([0], device=device)

        yaw = self._get_robot_yaw()
        new_yaw = yaw + angular_z * dt
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        dx = (linear_x * cos_y - linear_y * sin_y) * dt
        dy = (linear_x * sin_y + linear_y * cos_y) * dt

        pos = robot.data.root_pos_w[0:1].clone()
        pos[0, 0] += dx
        pos[0, 1] += dy
        quat = math_utils.quat_from_euler_xyz(
            torch.zeros(1, device=device),
            torch.zeros(1, device=device),
            torch.tensor([new_yaw], device=device),
        )
        root_pose = torch.cat([pos, quat], dim=-1)
        robot.write_root_pose_to_sim(root_pose, env_ids=env_ids)
        robot.write_root_velocity_to_sim(torch.zeros(1, 6, device=device), env_ids=env_ids)
        self._carry_box_with_root()

    def _step_swerve_l_motion(self) -> None:
        if self._swerve_phase is None:
            self._apply_swerve_cmd_vel(0.0, 0.0, 0.0)
            return

        if self._swerve_phase == "rotate":
            yaw_err = self._yaw_error(self._rotation_target_yaw)
            if abs(yaw_err) < self._swerve_yaw_tol:
                self._swerve_phase = "forward"
                robot = self.env.scene["robot"]
                pos = robot.data.root_pos_w[0].detach().cpu().tolist()
                yaw = self._get_robot_yaw()
                self._swerve_forward_start_xy = (pos[0], pos[1])
                self._swerve_forward_dir_xy = (math.cos(yaw), math.sin(yaw))
                print(f"[Control] Swerve: rotation done; driving toward the {self._l_motion_label}")
                return

            angular_z = max(
                -self._swerve_max_angular_z,
                min(self._swerve_max_angular_z, 2.5 * yaw_err),
            )
            self._apply_swerve_cmd_vel(0.0, 0.0, angular_z)
            return

        if self._swerve_phase == "forward":
            robot = self.env.scene["robot"]
            pos = robot.data.root_pos_w[0].detach().cpu().tolist()
            if self._swerve_forward_start_xy is None or self._swerve_forward_dir_xy is None:
                self._swerve_phase = None
                self._apply_swerve_cmd_vel(0.0, 0.0, 0.0)
                return

            dx = pos[0] - self._swerve_forward_start_xy[0]
            dy = pos[1] - self._swerve_forward_start_xy[1]
            traveled = dx * self._swerve_forward_dir_xy[0] + dy * self._swerve_forward_dir_xy[1]
            if traveled >= self._forward_distance_m:
                self._swerve_phase = None
                self._apply_swerve_cmd_vel(0.0, 0.0, 0.0)
                print(f"[Control] Swerve: finished driving toward the {self._l_motion_label}")
                return

            linear_x = min(self._swerve_max_linear_x, max(0.05, 0.5 * (self._forward_distance_m - traveled)))
            self._apply_swerve_cmd_vel(linear_x, 0.0, 0.0)

    def add_callback(self, key: str, func: Callable):
        """Add callback function for a specific key."""
        self._additional_callbacks[key] = func
