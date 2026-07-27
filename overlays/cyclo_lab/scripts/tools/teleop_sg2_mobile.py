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

"""Drive the FFW_SG2 mobile base with the keyboard (3-module holonomic swerve).

    cd /workspace/cyclo_lab
    ./third_party/IsaacLab/isaaclab.sh -p scripts/tools/teleop_sg2_mobile.py

Click the Isaac Sim window to give it focus, then press keys.

    ↑ / Numpad 8   forward           Z / Numpad 7   turn left
    ↓ / Numpad 2   backward          X / Numpad 9   turn right
    ← / Numpad 6   crab left         L              stop
    → / Numpad 4   crab right

    E  speed +25%     Q  speed -25%     R  reset robot pose
    C  toggle robot-frame / world-frame

Press forward and crab together to go diagonally (holonomic).
"""

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Keyboard driving for the FFW_SG2 mobile base")
parser.add_argument("--speed", type=float, default=0.5, help="Initial speed scale (default 0.5)")
parser.add_argument("--world_frame", action="store_true", help="Start in world frame")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Driving needs a window.
args_cli.headless = False
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.sim import SimulationContext

from cyclo_lab.assets.robots.FFW_SG2_MOBILE import FFW_SG2_MOBILE_CFG, SG2_MOBILE_SPAWN_HEIGHT
from cyclo_lab.controllers import SwerveController

DT = 1.0 / 120.0


class Teleop:
    """Keyboard state -> swerve command."""

    def __init__(self, robot, speed: float, world_frame: bool):
        self.robot = robot
        self.swerve = SwerveController(robot)
        self.speed = speed
        self.world_frame = world_frame

        # Keep sensitivity at 1.0 and multiply the scale here (it must change at runtime).
        self.keyboard = Se2Keyboard(
            Se2KeyboardCfg(v_x_sensitivity=1.0, v_y_sensitivity=1.0, omega_z_sensitivity=1.5)
        )
        self.keyboard.add_callback("E", self._faster)
        self.keyboard.add_callback("Q", self._slower)
        self.keyboard.add_callback("C", self._toggle_frame)
        self.keyboard.add_callback("R", self._reset_pose)
        self._banner()

    def _banner(self):
        print("\n" + "=" * 68, flush=True)
        print(self.keyboard, flush=True)
        print("\t----------------------------------------------", flush=True)
        print("\tSpeed up: E   Speed down: Q", flush=True)
        print("\tToggle robot/world frame: C   Reset pose: R", flush=True)
        print("=" * 68, flush=True)
        self._status()

    def _status(self):
        frame = "world frame" if self.world_frame else "robot frame"
        print(
            f"  [speed x{self.speed:.2f}]  forward max {self.speed:.2f} m/s  |  {frame}",
            flush=True,
        )

    def _faster(self):
        self.speed = min(self.speed * 1.25, 4.0)
        self._status()

    def _slower(self):
        self.speed = max(self.speed / 1.25, 0.05)
        self._status()

    def _toggle_frame(self):
        self.world_frame = not self.world_frame
        self._status()

    def set_home(self):
        """Save the settled pose as the reset reference point.

        Do not use default_root_state: it is init_state.pos (0, 0, 0.01), but the root body's actual
        height is 1.44 (a 1.43 m offset inside the USD). Writing it verbatim buries the robot
        underground so the wheels spin uselessly.
        """
        self._home = torch.cat(
            [self.robot.data.root_pos_w[:, :3], self.robot.data.root_quat_w], dim=-1
        ).clone()

    def _reset_pose(self):
        self.robot.write_root_pose_to_sim(self._home.clone())
        self.robot.write_root_velocity_to_sim(torch.zeros((1, 6), device=self.robot.device))
        print("  [reset] robot back to start position", flush=True)

    def step(self):
        cmd = self.keyboard.advance()
        vx = float(cmd[0]) * self.speed
        vy = float(cmd[1]) * self.speed
        omega = float(cmd[2]) * self.speed

        if abs(vx) < 1e-4 and abs(vy) < 1e-4 and abs(omega) < 1e-4:
            self.swerve.stop()
        elif self.world_frame:
            self.swerve.apply_world(vx, vy, omega)
        else:
            self.swerve.apply(vx, vy, omega)


def main():
    sim = SimulationContext(sim_utils.SimulationCfg(dt=DT, device=args_cli.device))
    sim.set_camera_view(eye=(3.5, 3.5, 2.5), target=(0.0, 0.0, 0.5))

    sim_utils.GroundPlaneCfg().func("/World/ground", sim_utils.GroundPlaneCfg())
    sim_utils.DomeLightCfg(intensity=2500.0).func("/World/light", sim_utils.DomeLightCfg(intensity=2500.0))

    robot = Articulation(FFW_SG2_MOBILE_CFG.replace(prim_path="/World/Robot"))
    sim.reset()

    teleop = Teleop(robot, speed=args_cli.speed, world_frame=args_cli.world_frame)

    # Wait a moment until it settles onto the wheels (a fall of SG2_MOBILE_SPAWN_HEIGHT).
    for _ in range(120):
        robot.write_data_to_sim()
        sim.step()
        robot.update(DT)
    teleop.set_home()  # the R key returns here.
    print(f"  settled (root_z={robot.data.root_pos_w[0, 2].item():.4f}) — driving enabled\n", flush=True)

    while simulation_app.is_running():
        teleop.step()
        robot.write_data_to_sim()
        sim.step()
        robot.update(DT)


if __name__ == "__main__":
    main()
    simulation_app.close()
