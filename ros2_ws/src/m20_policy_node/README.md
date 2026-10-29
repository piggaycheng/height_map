# m20_policy_node

External ROS 2 inference node for the M20 policy trained in `locomotion_training`
(`MyLocomotionTraining-Velocity-Flat-M20`). It loads the TorchScript `policy.pt` exported by `isaaclab play`
(`logs/rsl_rl/m20_flat/<run>/exported/policy.pt`) and is independent of Isaac Sim, so the same node can drive a real robot.

| Direction | Topic | Type | Content |
|-----------|-------|------|---------|
| in | `/m20/joint_states` | `sensor_msgs/JointState` | position/velocity of the 16 joints by name; **each message triggers one inference** |
| in | `/m20/imu` | `sensor_msgs/Imu` | orientation (wxyz base→world) and angular velocity in the base frame |
| in | `/m20/base_velocity` | `geometry_msgs/TwistStamped` | linear velocity in the base frame |
| in | `/cmd_vel` | `geometry_msgs/Twist` | vx, vy, yaw rate (treated as zero after `command_timeout_s`) |
| out | `/m20/joint_commands` | `sensor_msgs/JointState` | leg `position` targets (rad) and wheel `velocity` targets (rad/s), NaN where unused; stamp copied from the triggering joint state |

The launch file defaults to `use_sim_time:=true` (follows `/clock`, which `m20_standalone.py --external` publishes); pass `use_sim_time:=false` on the real robot.
Publish IMU, base velocity and command before `joint_states`. Wheel contact is not measured (assumed 1.0).

```bash
docker compose run --rm ros build
ros2 launch m20_policy_node m20_policy.launch.py policy_path:=/workspace/path/to/policy.pt  # use_sim_time:=false for hardware
# Isaac Sim side (host), using the ROS 2 DDS settings of run_height_map.sh:
scripts/m20_standalone.py --external
```
