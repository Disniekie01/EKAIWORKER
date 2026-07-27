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

"""``FFW_SG2_MOBILE`` regression check: does it stand / roll / move holonomically.

    cd /workspace/cyclo_lab
    ./third_party/IsaacLab/isaaclab.sh -p scripts/tools/check_ffw_sg2_mobile.py --headless
    ./third_party/IsaacLab/isaaclab.sh -p scripts/tools/check_ffw_sg2_mobile.py   # visual

Running this on the stock FFW_SG2 is expected to fail everything (the base is welded to the world).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="FFW_SG2_MOBILE regression check")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sim import SimulationContext
from isaaclab.utils.math import euler_xyz_from_quat

from cyclo_lab.assets.robots.FFW_SG2 import SG2_SWERVE_WHEEL_RADIUS
from cyclo_lab.assets.robots.FFW_SG2_MOBILE import FFW_SG2_MOBILE_CFG
from cyclo_lab.controllers import SwerveController

DT = 1.0 / 120.0
# Stock wheel stop: +/-1080 deg = 18.85 rad = stops after 1.63 m. The conversion must exceed this.
STOCK_LIMIT_RAD = 1080.0 * math.pi / 180.0
STOCK_LIMIT_M = STOCK_LIMIT_RAD * SG2_SWERVE_WHEEL_RADIUS

results: list[tuple[str, bool, str]] = []


def info(*args):
    print(*args, flush=True)  # Kit swallows the buffer, so flushing is required.


def record(name: str, passed: bool, detail: str):
    results.append((name, passed, detail))
    info(f"    {'O PASS' if passed else 'X FAIL'}  {detail}")


def main() -> int:
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args_cli.device))
    sim.set_camera_view(eye=(4.0, 4.0, 3.0), target=(0.0, 0.0, 0.5))
    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func(
        "/World/light", sim_utils.DomeLightCfg(intensity=2500.0)
    )

    robot = Articulation(FFW_SG2_MOBILE_CFG.replace(prim_path="/World/Robot"))
    sim.reset()
    swerve = SwerveController(robot)

    info("=" * 88)
    info("FFW_SG2_MOBILE regression check")
    info("=" * 88)
    info(f"  bodies={robot.num_bodies} joints={robot.num_joints}")
    info(f"  module order: {swerve.module_keys}")

    def step(n: int, twist=None):
        for _ in range(n):
            if twist is None:
                swerve.stop()
            else:
                swerve.apply(*twist)
            robot.write_data_to_sim()
            sim.step()
            robot.update(DT)

    # Remember the actual settled pose and return to it here.
    # Do not use default_root_state: it is init_state.pos (0, 0, 0.01), but the root body's actual
    # height is 1.4396 (a 1.43 m offset inside the USD). Writing it verbatim buries the robot 1.43 m
    # underground so the wheels spin uselessly.
    home_pose: list[torch.Tensor] = []

    def reset_pose():
        robot.write_root_pose_to_sim(home_pose[0].clone())
        robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=robot.device))
        step(180)  # until it settles again

    def align_steer(twist):
        """Align only the steering before driving. Without this the wheels push while the steer is
        turning and the robot wobbles."""
        angles, _ = swerve.compute(*twist)
        for _ in range(180):
            robot.set_joint_position_target(angles, joint_ids=swerve._steer_ids)
            robot.set_joint_velocity_target(
                torch.zeros((1, len(swerve._drive_ids)), device=robot.device),
                joint_ids=swerve._drive_ids,
            )
            robot.write_data_to_sim()
            sim.step()
            robot.update(DT)

    def yaw() -> float:
        _, _, y = euler_xyz_from_quat(robot.data.root_quat_w)
        return y[0].item()

    # [1] Is the base free? ---------------------------------------------------
    info("\n[1] Base fixing released")
    record("fixing released", not robot.is_fixed_base, f"is_fixed_base={robot.is_fixed_base} (must be False)")

    # [2] Does it stand on its wheels? ----------------------------------------
    info("\n[2] Settling stability under gravity (3 s)")
    step(360)
    z = robot.data.root_pos_w[0, 2].item()
    speed = torch.norm(robot.data.root_lin_vel_w[0]).item()
    record("settling", speed < 0.2 and 0.5 < z < 2.5, f"root_z={z:.4f} residual speed={speed:.3f} m/s")

    # Save the settled pose as the reference point (reset_pose returns here).
    home_pose.append(torch.cat([robot.data.root_pos_w[:, :3], robot.data.root_quat_w], dim=-1).clone())

    # [3] Does it drive past the stock wheel stop? ----------------------------
    info(f"\n[3] Straight drive for 10 s — does it break the stock limit {STOCK_LIMIT_M:.2f} m")
    reset_pose()
    align_steer((0.865, 0.0, 0.0))
    start = robot.data.root_pos_w[0, :3].clone()
    angle_start = robot.data.joint_pos[0, swerve._drive_ids].clone()
    step(1200, twist=(0.865, 0.0, 0.0))  # ~10 rad/s
    moved = torch.norm((robot.data.root_pos_w[0, :3] - start)[:2]).item()
    spun = (robot.data.joint_pos[0, swerve._drive_ids] - angle_start).mean().item()
    record(
        "straight drive",
        moved > 2.0 and abs(spun) > STOCK_LIMIT_RAD,
        f"moved {moved:.2f} m, wheels spun {spun:.1f} rad ({spun * 57.2958:.0f} deg)",
    )

    # [4] Holonomic maneuvers -------------------------------------------------
    # Reset the pose before each maneuver. Accumulated spin makes the robot-frame command and the
    # world-frame measurement diverge, misjudging correct behavior as a failure.
    info("\n[4] Holonomic maneuvers (reset pose before each)")

    info("\n  [4-1] Crab vy=+0.5 (expect: move in +y, no rotation)")
    reset_pose()
    align_steer((0.0, 0.5, 0.0))
    start = robot.data.root_pos_w[0, :3].clone()
    y0 = yaw()
    step(360, twist=(0.0, 0.5, 0.0))
    d = robot.data.root_pos_w[0, :3] - start
    dyaw = math.degrees(yaw() - y0)
    record(
        "crab",
        d[1].item() > 1.0 and abs(d[0].item()) < 0.3,
        f"dx={d[0].item():+.3f} dy={d[1].item():+.3f} dyaw={dyaw:+.1f} deg",
    )

    info("\n  [4-2] Spin in place omega=+0.8 (expect: rotation only, no translation)")
    reset_pose()
    align_steer((0.0, 0.0, 0.8))
    start = robot.data.root_pos_w[0, :3].clone()
    y0 = yaw()
    step(360, twist=(0.0, 0.0, 0.8))
    d = robot.data.root_pos_w[0, :3] - start
    dyaw = math.degrees(yaw() - y0)
    drift = math.hypot(d[0].item(), d[1].item())
    record("spin in place", abs(dyaw) > 60.0 and drift < 0.4, f"dyaw={dyaw:+.0f} deg drift={drift:.3f} m")

    info("\n  [4-3] Diagonal vx=vy=+0.35 (expect: +x +y together)")
    reset_pose()
    align_steer((0.35, 0.35, 0.0))
    start = robot.data.root_pos_w[0, :3].clone()
    step(360, twist=(0.35, 0.35, 0.0))
    d = robot.data.root_pos_w[0, :3] - start
    record(
        "diagonal",
        d[0].item() > 0.7 and d[1].item() > 0.7,
        f"dx={d[0].item():+.3f} dy={d[1].item():+.3f}",
    )

    # Summary -----------------------------------------------------------------
    info("\n" + "=" * 88)
    passed = sum(1 for _, ok, _ in results if ok)
    info(f"Result: {passed}/{len(results)} passed")
    info("=" * 88)
    for name, ok, detail in results:
        info(f"  {'O' if ok else 'X'}  {name:12s} {detail}")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    code = main()
    simulation_app.close()
