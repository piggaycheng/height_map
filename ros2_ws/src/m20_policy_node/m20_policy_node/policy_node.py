#!/usr/bin/env python3
"""ROS 2 node running the M20 locomotion policy outside the simulator / robot process.

Subscribes (all topic names are parameters):
  /m20/joint_states    sensor_msgs/JointState   position + velocity of the 16 M20 joints, by name.
                                                 Every message triggers one inference, so publish it last.
  /m20/imu             sensor_msgs/Imu          orientation (base -> world) and angular velocity in the base frame
  /m20/base_velocity   geometry_msgs/TwistStamped  base linear velocity in the base frame
  /cmd_vel             geometry_msgs/Twist      velocity command (vx, vy, yaw rate)
Publishes:
  /m20/joint_commands  sensor_msgs/JointState   name = 16 joints; position = leg position targets (rad),
                                                 velocity = wheel velocity targets (rad/s); NaN where unused.
                                                 header.stamp is copied from the joint_states that triggered it.
"""

import numpy as np
import rclpy
from geometry_msgs.msg import Twist, TwistStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu, JointState

from . import m20_policy as m20


class M20PolicyNode(Node):
    def __init__(self) -> None:
        super().__init__("m20_policy_node")
        self.declare_parameter("policy_path", "")
        self.declare_parameter("device", "cpu")
        self.declare_parameter("joint_states_topic", "/m20/joint_states")
        self.declare_parameter("imu_topic", "/m20/imu")
        self.declare_parameter("base_velocity_topic", "/m20/base_velocity")
        self.declare_parameter("cmd_vel_topic", "/cmd_vel")
        self.declare_parameter("joint_commands_topic", "/m20/joint_commands")
        # a command older than this is treated as zero (0 disables the check)
        self.declare_parameter("command_timeout_s", 0.5)
        # no contact sensing on the robot yet: 1.0 assumes every wheel touches the ground
        self.declare_parameter("assume_wheel_contact", True)

        policy_path = self.get_parameter("policy_path").value
        if not policy_path:
            raise RuntimeError("Parameter 'policy_path' (TorchScript policy.pt) is required")
        self._policy = m20.M20Policy(policy_path, self.get_parameter("device").value)
        self._command_timeout = float(self.get_parameter("command_timeout_s").value)
        self._wheel_contact = np.full(4, 1.0 if self.get_parameter("assume_wheel_contact").value else 0.0)

        self._orientation_wxyz = None
        self._ang_vel = None
        self._lin_vel = None
        self._command = np.zeros(3)
        self._command_time = None
        self._warned_missing = False
        self._last_stamp_ns = None

        param = self.get_parameter
        self._pub = self.create_publisher(JointState, param("joint_commands_topic").value, 1)
        self.create_subscription(Imu, param("imu_topic").value, self._on_imu, 1)
        self.create_subscription(TwistStamped, param("base_velocity_topic").value, self._on_base_velocity, 1)
        self.create_subscription(Twist, param("cmd_vel_topic").value, self._on_cmd_vel, 1)
        self.create_subscription(JointState, param("joint_states_topic").value, self._on_joint_states, 1)
        self.get_logger().info(f"Loaded policy {policy_path}")

    def _on_imu(self, msg: Imu) -> None:
        q = msg.orientation
        self._orientation_wxyz = np.array([q.w, q.x, q.y, q.z])
        w = msg.angular_velocity
        self._ang_vel = np.array([w.x, w.y, w.z])

    def _on_base_velocity(self, msg: TwistStamped) -> None:
        v = msg.twist.linear
        self._lin_vel = np.array([v.x, v.y, v.z])

    def _on_cmd_vel(self, msg: Twist) -> None:
        self._command = np.array([msg.linear.x, msg.linear.y, msg.angular.z])
        self._command_time = self.get_clock().now()

    def _active_command(self) -> np.ndarray:
        if self._command_timeout <= 0.0:
            return self._command
        if self._command_time is None:
            return np.zeros(3)
        # uses the node clock (/clock with use_sim_time); a negative age means the clock was reset
        age = (self.get_clock().now() - self._command_time).nanoseconds * 1e-9
        return self._command if 0.0 <= age <= self._command_timeout else np.zeros(3)

    def _on_joint_states(self, msg: JointState) -> None:
        if self._orientation_wxyz is None or self._lin_vel is None:
            if not self._warned_missing:
                self.get_logger().warn("Waiting for IMU and base velocity before running the policy")
                self._warned_missing = True
            return
        stamp_ns = msg.header.stamp.sec * 1_000_000_000 + msg.header.stamp.nanosec
        if self._last_stamp_ns is not None and stamp_ns < self._last_stamp_ns:
            self._policy.reset()  # the state source restarted
        self._last_stamp_ns = stamp_ns

        index = {name: i for i, name in enumerate(msg.name)}
        missing = [name for name in m20.JOINT_NAMES if name not in index]
        if missing or len(msg.position) != len(msg.name) or len(msg.velocity) != len(msg.name):
            self.get_logger().error(f"Invalid joint state (missing joints: {missing}); skipping inference")
            return
        order = [index[name] for name in m20.JOINT_NAMES]
        pos = np.asarray(msg.position)[order]
        vel = np.asarray(msg.velocity)[order]

        obs = m20.build_observation(
            self._lin_vel,
            self._ang_vel,
            self._orientation_wxyz,
            self._active_command(),
            pos[:12],
            vel[:12],
            vel[12:],
            self._policy.last_action,
            self._wheel_contact,
        )
        if not np.all(np.isfinite(obs)):
            self.get_logger().error("Non-finite observation; skipping inference")
            return
        leg_targets, wheel_targets = m20.action_to_targets(self._policy.infer(obs))

        out = JointState()
        out.header.stamp = msg.header.stamp
        out.name = list(m20.JOINT_NAMES)
        out.position = [float(v) for v in leg_targets] + [float("nan")] * 4
        out.velocity = [float("nan")] * 12 + [float(v) for v in wheel_targets]
        self._pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = M20PolicyNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
