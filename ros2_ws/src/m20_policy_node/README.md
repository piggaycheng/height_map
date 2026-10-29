# m20_policy_node

External ROS 2 inference node for the M20 policy trained in `locomotion_training`
(`MyLocomotionTraining-Velocity-Flat-M20` or `...-Rough-M20`). It loads the TorchScript `policy.pt` exported by `isaaclab play`
(`logs/rsl_rl/m20_{flat,rough}/<run>/exported/policy.pt`) and is independent of Isaac Sim, so the same node can drive a real robot.

| Direction | Topic | Type | Content |
|-----------|-------|------|---------|
| in | `/m20/joint_states` | `sensor_msgs/JointState` | position/velocity of the 16 joints by name; **each message triggers one inference** |
| in | `/m20/imu` | `sensor_msgs/Imu` | orientation (wxyz base→world) and angular velocity in the base frame |
| in | `/m20/base_velocity` | `geometry_msgs/TwistStamped` | linear velocity in the base frame |
| in | `/cmd_vel` | `geometry_msgs/Twist` | vx, vy, yaw rate (treated as zero after `command_timeout_s`) |
| out | `/m20/joint_commands` | `sensor_msgs/JointState` | leg `position` targets (rad) and wheel `velocity` targets (rad/s), NaN where unused; stamp copied from the triggering joint state |

The launch file defaults to `use_sim_time:=true` (follows `/clock`, which `m20_standalone.py --external` publishes); pass `use_sim_time:=false` on the real robot.
**Rough policy (247-dim obs):** the 17x11 height scan (1.6 x 1.0 m, 0.1 m, yaw-aligned, `base_z - ground_z - 0.5` clipped to +-1) is
sampled from the elevation_mapping_cupy GridMap (`elevation_map_topic`, layer `height_layer`, map -> `base_frame` via TF).
`drift_free_height` (default true) takes the base height as the standing height above the median scan terrain, so LiDAR-odometry z drift does not corrupt the scan. `use_height_scan:=false` gives the 60-dim flat obs. Until the map and TF are available the node publishes the default pose.

Controls (GUI): Up/Down forward/back, Left/Right strafe, N/M (or Q/E) turn left/right, Space stop; `--command VX VY WZ`, `--demo` (scripted loop) or `--headless` (zero command) otherwise. RViz starts with the script (`--no-rviz` to skip). The IMU/TF graph starts after 3 s of sim time so FAST-LIO does not initialise on the spawn jolt (it diverged otherwise). The LiDAR sits on the body at z=0.15 (`--lidar-offset`), its mesh is hidden and returns closer than `--lidar-near-range` (default 1.0 m, `nearRangeM`) are dropped so the robot body does not show up in the cloud.

Scene options of `scripts/m20_standalone.py`: `--environment office|grid` (default office); `--office-ground mesh|plane` (default `mesh`: floors and stairs get triangle-mesh colliders so you can descend below z=0; `plane` keeps a collision plane at z=0 plus stair colliders). Debug: `--flat-scan` (local mode) and `FLAT_HEIGHT_SCAN=true scripts/run_m20_policy.sh` (node param `flat_height_scan`) fill the height scan with flat ground to isolate policy vs. map problems.
The launch file also starts FAST-LIO2 + elevation mapping (`height_map:=false` to skip). The sim side needs the LiDAR/IMU
mounted by `m20_standalone.py --external` (`--no-lidar` to disable). LiDAR odometry degrades on the featureless default
grid ground; use a scene with structure for long runs.

Publish IMU, base velocity and command before `joint_states`. Wheel contact is not measured (assumed 1.0).

```bash
docker compose run --rm ros build
ros2 launch m20_policy_node m20_policy.launch.py policy_path:=/workspace/path/to/policy.pt  # use_sim_time:=false for hardware
# Isaac Sim side (host), using the ROS 2 DDS settings of run_height_map.sh:
scripts/m20_standalone.py --external
# or everything at once (rebuild with --build after node changes):
scripts/run_m20_policy.sh --policy /path/to/exported/policy.pt
```
