#!/home/user/isaacsim-6.2.0/python.sh
"""Isaac Sim 6.2 standalone: Deep Robotics M20 driven by a policy trained in locomotion_training.

Modeled on standalone_examples/api/isaacsim.robot.policy.examples/go2_standalone.py, but the M20
(wheel-legged) is not a bundled policy spec, so observations and actions are built here to match
my_locomotion_training "Velocity-Flat-M20":
  obs (60) = base lin vel(3) | base ang vel(3) | projected gravity(3) | command(3) | leg pos rel(12)
             | leg vel rel *0.05 (12) | last action(16) | wheel vel *0.05 (4) | wheel contact(4)
  act (16) = 12 leg position offsets (*0.25 rad, added to default pose, clipped +-1.5)
             | 4 wheel velocities (*20 rad/s, clipped +-60)
The policy is the TorchScript file exported by `isaaclab play` (logs/rsl_rl/m20_flat/<run>/exported/policy.pt).
Without --policy the robot only holds its default pose (zero actions), which is useful as a smoke test.

With --external the policy runs in the ROS 2 node `m20_policy_node` instead: each control step this script
publishes /clock (simulation time), /m20/imu, /m20/base_velocity, /cmd_vel and finally /m20/joint_states, then waits (up to
--external-timeout) for the matching /m20/joint_commands and applies it. See ros2_ws/src/m20_policy_node.
"""

import argparse
import re

from isaacsim import SimulationApp

M20_ASSET_PATH = "/Isaac/Robots/DeepRobotics/M20/M20.usd"
M20_PRIM_PATH = "/World/M20"
VARIANTS = {"Physics": "PhysX", "Sensor": "None", "Robot": "Robot"}
PHYSICS_DT = 0.005
DECIMATION = 4
STANDING_HEIGHT = 0.49
LEG_JOINTS = re.compile(r".*_(hipx|hipy|knee)_joint")
WHEEL_JOINTS = re.compile(r".*_wheel_joint")
DEFAULT_POS = [
    (re.compile(r".*_hipx_joint"), 0.0),
    (re.compile(r"f[lr]_hipy_joint"), -0.65),
    (re.compile(r"h[lr]_hipy_joint"), 0.65),
    (re.compile(r"f[lr]_knee_joint"), 1.3),
    (re.compile(r"h[lr]_knee_joint"), -1.3),
]
LEG_ACTION_SCALE = 0.25
LEG_TARGET_LIMIT = 1.5
WHEEL_ACTION_SCALE = 20.0
WHEEL_TARGET_LIMIT = 60.0
LEG_STIFFNESS, LEG_DAMPING, LEG_MAX_EFFORT, LEG_MAX_VELOCITY = 80.0, 2.0, 76.4, 22.4
WHEEL_STIFFNESS, WHEEL_DAMPING, WHEEL_MAX_EFFORT, WHEEL_MAX_VELOCITY = 0.0, 2.0, 21.6, 79.3

# scripted velocity commands: (seconds, [vx, vy, yaw_rate]); training ranges are vx +-1.5, vy +-0.3, yaw +-1.0
COMMAND_PHASES = [
    (3.0, [0.0, 0.0, 0.0]),
    (5.0, [0.8, 0.0, 0.0]),
    (4.0, [0.5, 0.0, 0.6]),
    (3.0, [0.0, 0.0, 0.0]),
    (4.0, [-0.5, 0.0, 0.0]),
]

parser = argparse.ArgumentParser(description="Run the M20 with an exported locomotion_training policy.")
parser.add_argument("--policy", type=str, default=None, help="TorchScript policy.pt exported by isaaclab play.")
parser.add_argument("--external", action="store_true", help="Run inference in the external m20_policy_node via ROS 2.")
parser.add_argument("--external-timeout", type=float, default=0.2, help="Seconds to wait for each external action.")
parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim GUI.")
parser.add_argument("--test", action="store_true", help="Run the command script once and exit.")
parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda", help="Simulation device.")
parser.add_argument("--command", type=float, nargs=3, metavar=("VX", "VY", "WZ"), help="Constant command instead of the script.")
args, _ = parser.parse_known_args()

simulation_app = SimulationApp({"headless": args.headless})

import time

import carb
import numpy as np
import omni.timeline
import omni.usd
from isaacsim.core.experimental.prims import Articulation
from isaacsim.core.experimental.utils.stage import define_prim, set_stage_units, set_stage_up_axis
from isaacsim.core.rendering_manager import RenderingManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.core.simulation_manager.impl.isaac_events import IsaacEvents
from isaacsim.storage.native import get_assets_root_path

if args.external and args.policy:
    parser.error("--policy and --external are mutually exclusive")
policy = None
if args.policy:
    import torch

    policy = torch.jit.load(args.policy, map_location="cpu").eval()
print(f"Policy: {'external ROS 2 node' if args.external else args.policy or 'none (zero actions)'}")


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Express a world-frame vector in the frame described by ``quat_wxyz``."""
    w, x, y, z = quat_wxyz
    q_vec = np.array([-x, -y, -z])
    t = 2.0 * np.cross(q_vec, vec)
    return vec + w * t + np.cross(q_vec, t)


robot = None
leg_idx = wheel_idx = None
default_leg_pos = None
last_action = np.zeros(16, dtype=np.float32)
step_count = 0
needs_init = True
reset_needed = False
failed = False
base_command = np.zeros(3, dtype=np.float32)


def build_observation() -> np.ndarray:
    positions, orientations = robot.get_world_poses()
    quat = orientations.numpy().reshape(-1)
    lin_vel, ang_vel = robot.get_velocities()
    dof_pos = robot.get_dof_positions().numpy().reshape(-1)
    dof_vel = robot.get_dof_velocities().numpy().reshape(-1)
    return np.concatenate(
        [
            quat_rotate_inverse(quat, lin_vel.numpy().reshape(-1)),
            quat_rotate_inverse(quat, ang_vel.numpy().reshape(-1)),
            quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0])),
            base_command,
            dof_pos[leg_idx] - default_leg_pos,
            dof_vel[leg_idx] * 0.05,
            last_action,
            dof_vel[wheel_idx] * 0.05,
            np.ones(len(wheel_idx)),  # wheel contact flags: assumes flat ground contact
        ]
    ).astype(np.float32)


def apply_action(action: np.ndarray) -> None:
    leg_targets = np.clip(default_leg_pos + LEG_ACTION_SCALE * action[:12], -LEG_TARGET_LIMIT, LEG_TARGET_LIMIT)
    wheel_targets = np.clip(WHEEL_ACTION_SCALE * action[12:], -WHEEL_TARGET_LIMIT, WHEEL_TARGET_LIMIT)
    apply_targets(leg_targets, wheel_targets)


def apply_targets(leg_targets: np.ndarray, wheel_targets: np.ndarray) -> None:
    robot.set_dof_position_targets(leg_targets.reshape(1, -1), dof_indices=leg_idx)
    robot.set_dof_velocity_targets(wheel_targets.reshape(1, -1), dof_indices=wheel_idx)


class ExternalPolicyBridge:
    """Publishes the robot state over ROS 2 and applies the actions returned by m20_policy_node."""

    def __init__(self, timeout: float) -> None:
        import isaacsim.core.experimental.utils.app as app_utils

        app_utils.enable_extension("isaacsim.ros2.bridge")
        simulation_app.update()
        import rclpy
        from geometry_msgs.msg import Twist, TwistStamped
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import Imu, JointState

        self._rclpy, self._Twist, self._TwistStamped = rclpy, Twist, TwistStamped
        self._Imu, self._JointState, self._Clock = Imu, JointState, Clock
        rclpy.init()
        self._node = rclpy.create_node("m20_isaac_sim")
        self._clock_pub = self._node.create_publisher(Clock, "/clock", 1)
        self._imu_pub = self._node.create_publisher(Imu, "/m20/imu", 1)
        self._vel_pub = self._node.create_publisher(TwistStamped, "/m20/base_velocity", 1)
        self._cmd_pub = self._node.create_publisher(Twist, "/cmd_vel", 1)
        self._joint_pub = self._node.create_publisher(JointState, "/m20/joint_states", 1)
        self._node.create_subscription(JointState, "/m20/joint_commands", self._on_command, 1)
        self._timeout = timeout
        self._pending_stamp = None
        self._received = None
        self.timeouts = 0

    def _on_command(self, msg) -> None:
        self._received = msg

    def close(self) -> None:
        self._node.destroy_node()
        self._rclpy.shutdown()

    def step(self) -> None:
        stamp_ns = int(round(step_count * PHYSICS_DT * 1e9))
        # simulation time since the last (re)start, like Isaac Sim's own clock; also published on /clock
        clock = self._Clock()
        clock.clock.sec, clock.clock.nanosec = divmod(stamp_ns, 1_000_000_000)
        stamp = clock.clock
        pos, quat = robot.get_world_poses()
        quat = quat.numpy().reshape(-1)
        lin_vel, ang_vel = robot.get_velocities()
        lin_b = quat_rotate_inverse(quat, lin_vel.numpy().reshape(-1))
        ang_b = quat_rotate_inverse(quat, ang_vel.numpy().reshape(-1))

        imu = self._Imu()
        imu.header.stamp, imu.header.frame_id = stamp, "base_link"
        imu.orientation.w, imu.orientation.x, imu.orientation.y, imu.orientation.z = (float(v) for v in quat)
        imu.angular_velocity.x, imu.angular_velocity.y, imu.angular_velocity.z = (float(v) for v in ang_b)
        vel = self._TwistStamped()
        vel.header.stamp, vel.header.frame_id = stamp, "base_link"
        vel.twist.linear.x, vel.twist.linear.y, vel.twist.linear.z = (float(v) for v in lin_b)
        cmd = self._Twist()
        cmd.linear.x, cmd.linear.y, cmd.angular.z = (float(v) for v in base_command)
        joints = self._JointState()
        joints.header.stamp = stamp
        joints.name = list(dof_names)
        joints.position = [float(v) for v in robot.get_dof_positions().numpy().reshape(-1)]
        joints.velocity = [float(v) for v in robot.get_dof_velocities().numpy().reshape(-1)]

        self._received = None
        self._clock_pub.publish(clock)
        self._imu_pub.publish(imu)
        self._vel_pub.publish(vel)
        self._cmd_pub.publish(cmd)
        self._joint_pub.publish(joints)  # published last: it triggers inference in the node

        deadline = time.monotonic() + self._timeout
        while time.monotonic() < deadline:
            self._rclpy.spin_once(self._node, timeout_sec=0.005)
            received = self._received
            if received is not None and (received.header.stamp.sec, received.header.stamp.nanosec) == (
                stamp.sec,
                stamp.nanosec,
            ):
                self._apply(received)
                return
        self.timeouts += 1
        if self.timeouts in (1, 10) or self.timeouts % 100 == 0:
            carb.log_warn(f"m20_standalone: no action from m20_policy_node ({self.timeouts} timeouts); holding last targets")

    def _apply(self, msg) -> None:
        index = {name: i for i, name in enumerate(msg.name)}
        leg = np.array([msg.position[index[dof_names[i]]] for i in leg_ids], dtype=np.float32)
        wheel = np.array([msg.velocity[index[dof_names[i]]] for i in wheel_ids], dtype=np.float32)
        if not (np.all(np.isfinite(leg)) and np.all(np.isfinite(wheel))):
            carb.log_warn("m20_standalone: ignoring non-finite action from m20_policy_node")
            return
        apply_targets(leg, wheel)


bridge = None


def restart() -> None:
    global last_action, step_count
    robot.reset_to_default_state()
    robot.set_dof_position_targets(default_leg_pos.reshape(1, -1), dof_indices=leg_idx)
    robot.set_dof_velocity_targets(np.zeros((1, len(wheel_idx))), dof_indices=wheel_idx)
    last_action = np.zeros(16, dtype=np.float32)
    step_count = 0


def on_physics_step(step_size: float, context: object) -> None:
    global needs_init, reset_needed, failed, last_action, step_count
    if failed:
        return
    try:
        if needs_init:
            restart()
            needs_init = False
        elif reset_needed:
            reset_needed = False
            needs_init = True
        else:
            if step_count % DECIMATION == 0 and bridge is not None:
                bridge.step()
            elif step_count % DECIMATION == 0 and policy is not None:
                obs = torch.from_numpy(build_observation()).unsqueeze(0)
                with torch.inference_mode():
                    last_action = policy(obs).squeeze(0).numpy().astype(np.float32)
                apply_action(last_action)
            step_count += 1
    except Exception as error:  # noqa: BLE001 - a physics callback must not raise
        failed = True
        carb.log_error(f"m20_standalone: policy deployment failed, stopping: {error}")


set_stage_up_axis("Z")
set_stage_units(meters_per_unit=1.0)
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    raise RuntimeError("Could not find Isaac Sim assets folder")

ground = define_prim("/World/Ground", "Xform")
ground.GetReferences().AddReference(assets_root_path + "/Isaac/Environments/Grid/default_environment.usd")
define_prim("/World/PhysicsScene", "PhysicsScene")
SimulationManager.set_physics_sim_device(args.device)

m20_prim = define_prim(M20_PRIM_PATH, "Xform")
m20_prim.GetReferences().AddReference(assets_root_path + M20_ASSET_PATH)
for variant_set, selection in VARIANTS.items():
    m20_prim.GetVariantSets().GetVariantSet(variant_set).SetVariantSelection(selection)

robot = Articulation(M20_PRIM_PATH, positions=[[0.0, 0.0, STANDING_HEIGHT]], reset_xform_op_properties=True)

dof_names = list(robot.dof_names)
leg_ids = [i for i, n in enumerate(dof_names) if LEG_JOINTS.fullmatch(n)]
wheel_ids = [i for i, n in enumerate(dof_names) if WHEEL_JOINTS.fullmatch(n)]
if len(leg_ids) != 12 or len(wheel_ids) != 4:
    raise RuntimeError(f"Unexpected M20 joints: {dof_names}")
leg_idx, wheel_idx = np.array(leg_ids), np.array(wheel_ids)
default_leg_pos = np.array(
    [next(v for p, v in DEFAULT_POS if p.fullmatch(dof_names[i])) for i in leg_ids], dtype=np.float32
)
print(f"Leg joints: {[dof_names[i] for i in leg_ids]}")
print(f"Wheel joints: {[dof_names[i] for i in wheel_ids]}")

# match the actuator model used in training (implicit PD, see my_locomotion_training/assets/m20.py)
robot.set_dof_gains(LEG_STIFFNESS, LEG_DAMPING, dof_indices=leg_idx)
robot.set_dof_gains(WHEEL_STIFFNESS, WHEEL_DAMPING, dof_indices=wheel_idx)
robot.set_dof_max_efforts(LEG_MAX_EFFORT, dof_indices=leg_idx)
robot.set_dof_max_efforts(WHEEL_MAX_EFFORT, dof_indices=wheel_idx)
robot.set_dof_max_velocities(LEG_MAX_VELOCITY, dof_indices=leg_idx)
robot.set_dof_max_velocities(WHEEL_MAX_VELOCITY, dof_indices=wheel_idx)
default_positions = np.zeros((1, len(dof_names)), dtype=np.float32)
default_positions[0, leg_idx] = default_leg_pos
robot.set_default_state(
    positions=[[0.0, 0.0, STANDING_HEIGHT]],
    orientations=[[1.0, 0.0, 0.0, 0.0]],
    dof_positions=default_positions,
    dof_velocities=np.zeros_like(default_positions),
)

SimulationManager.set_physics_dt(PHYSICS_DT)
RenderingManager.set_dt(PHYSICS_DT * DECIMATION)

frame_dt = PHYSICS_DT * DECIMATION
if args.command is not None:
    frame_commands = np.tile(np.asarray(args.command, dtype=np.float32), (int(10.0 / frame_dt), 1))
else:
    frame_commands = np.concatenate(
        [np.tile(np.asarray(c, dtype=np.float32), (int(round(s / frame_dt)), 1)) for s, c in COMMAND_PHASES]
    )

if args.external:
    bridge = ExternalPolicyBridge(args.external_timeout)

callback_id = SimulationManager.register_callback(on_physics_step, IsaacEvents.POST_PHYSICS_STEP)

timeline = omni.timeline.get_timeline_interface()
timeline.play()
simulation_app.update()

i = 0
start_xy = None
while simulation_app.is_running():
    simulation_app.update()
    if failed:
        break
    if SimulationManager.is_simulating():
        if start_xy is None:
            start_xy = robot.get_world_poses()[0].numpy().reshape(-1)[:2].copy()
        if i == len(frame_commands):
            i = 0
            if args.test:
                end_xy = robot.get_world_poses()[0].numpy().reshape(-1)[:2]
                print(f"Test finished: base moved {np.linalg.norm(end_xy - start_xy):.3f} m from spawn")
                break
        base_command = frame_commands[i]
        i += 1
    else:
        reset_needed = True

timeline.stop()
SimulationManager.deregister_callback(callback_id)
if bridge is not None:
    bridge.close()
simulation_app.close(exit_code=1 if failed else 0)
