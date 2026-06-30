# AIWORKER

Portable overlay for AI Worker Isaac Sim VR teleoperation tasks.

This repo is intentionally small. It does not vendor the full ROBOTIS repositories and it does not include recorded datasets. Instead, `setup.sh` clones the required upstream repos at pinned commits, initializes `cyclo_lab` submodules, and copies the AIWORKER overlay files on top.

## What This Adds

- `cyclo_lab` dashboard for launching the stack and selecting tasks.
- SG2 L-table, box-stack, single-box-far, and thick-box variants.
- SH5 hand versions of the same tasks.
- SH5 DDS recorder support for VR hand teleoperation.
- Task-specific table/box assets and teleop motion settings.
- Minor `robotis_applications` VR publisher update used by this setup.

## Upstream Pins

- `cyclo_lab`: `85b237bf22068da18999bacbda5652f201594d11`
- `ai_worker`: `e8c2eacb612e47473cdf03e44bee6d527c00b4f9`
- `robotis_applications`: `7ef0aabc748174cb91013866b2e4142122ef475c`

## Install On A New Machine

```bash
git clone https://github.com/Disniekie01/EKAIWORKER.git AIWORKER
cd AIWORKER
./setup.sh ~/AIWORKER
```

The install directory can be any path. The command above creates:

```text
~/AIWORKER/
  cyclo_lab/
  ai_worker/
  robotis_applications/
```

## Start Containers

```bash
cd ~/AIWORKER/cyclo_lab/docker
./container.sh start

cd ~/AIWORKER/robotis_applications/docker
./container.sh start

cd ~/AIWORKER/ai_worker/docker
./container.sh start
```

## Start Dashboard

```bash
cd ~/AIWORKER/cyclo_lab
python3 sg2_ltable_dashboard.py
```

Open the dashboard at:

```text
http://localhost:8765
```

Select the task, then launch the stack from the dashboard.

## VR Notes

For SH5 hand tasks, the dashboard launches:

- `robotis_vuer vr.launch.py model:=sh5`
- `cyclo_motion_controller_ros ai_worker_controller.launch.py controller_type:=vr hand:=true`

The headset page is printed by the dashboard. It normally looks like:

```text
https://<host-ip>:8012
```

For hand tasks, VR publishing starts disabled by default. Enable it with the SH5 gesture or publish the override:

```bash
docker exec -it robotis-applications bash
export ROS_DOMAIN_ID=30
source /opt/ros/jazzy/setup.bash
source /root/ros2_ws/install/setup.bash
ros2 topic pub --once /vr/reactivate std_msgs/msg/Bool "{data: true}"
```

## Datasets

Recorded `.hdf5` datasets are intentionally excluded. Re-record demonstrations on the target machine using the dashboard.

## Notes

- This overlay assumes NVIDIA Docker and Isaac Sim container requirements are already satisfied.
- `cyclo_lab/docker/.env` is included to pin Isaac Sim 5.1 settings used during development.
- If you change upstream commits, re-test the overlay because task registration and IsaacLab APIs may move.
