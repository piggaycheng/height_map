#!/home/user/isaacsim-6.2.0/python.sh
"""Isaac Sim 6.2 standalone: Office + Nova Carter (RTX LiDAR + IMU) for FAST-LIO2 and elevation_mapping_cupy.

Published ROS 2 topics (root namespace):
  /clock                  rosgraph_msgs/Clock
  /isaac/lidar_points     sensor_msgs/PointCloud2, frame lidar_link
  /isaac/imu              sensor_msgs/Imu, frame imu_link
  /tf_static              body -> base_link -> lidar_link / imu_link
FAST-LIO2 (world_frame: map, body_frame: body, see config/isaac_lio.yaml) adds the dynamic
map -> body transform, which completes map -> body -> base_link for elevation_mapping_cupy.
"""

import argparse
import math
import sys
import traceback

from isaacsim import SimulationApp

OFFICE_ASSET_PATH = "/Isaac/Environments/Office/office.usd"
CARTER_ASSET_PATH = "/Isaac/Robots/NVIDIA/NovaCarter/nova_carter.usd"
CARTER_PRIM_PATH = "/World/Carter"
CHASSIS_PATH = f"{CARTER_PRIM_PATH}/chassis_link"
LIDAR_PATH = f"{CHASSIS_PATH}/sensors/XT_32/PandarXT_32_10hz"
IMU_PATH = f"{LIDAR_PATH}/fastlio_imu"
SURROUNDING_BUILDINGS_PRIM_PATH = "/Root/SM_Buildings"
USE_PHYSICS_PLANE = False  # False: replace the invisible plane with stair/floor mesh colliders
GROUND_COLLISION_PLANE_PATH = "/Root/GroundPlane/CollisionPlane"
WALKABLE_NAME_KEYWORDS = ("stair", "floor")
WHEEL_JOINTS = ["joint_wheel_left", "joint_wheel_right"]
WHEEL_RADIUS = 0.14
WHEEL_BASE = 0.4132
SPAWN_HEIGHT = 0.05
LINEAR_JOG_SPEED = 0.75
ANGULAR_JOG_SPEED = 0.5
FOLLOW_CAMERA_DISTANCE = 2.5
FOLLOW_CAMERA_HEIGHT = 1.5
FOLLOW_CAMERA_LOOK_AHEAD = 0.6
FOLLOW_CAMERA_TARGET_HEIGHT = 0.5
KIT_EXTRA_ARGS = [
    "--/rtx/post/dlss/execMode=0",
    "--/app/renderer/skipGpuRenderProducts=false",
    "--/rtx/rendermode=RealTime",
]

parser = argparse.ArgumentParser(description="Launch Isaac Sim 6.2 with the Office and a Nova Carter.")
parser.add_argument("--headless", action="store_true", help="Run without the Isaac Sim GUI.")
parser.add_argument("--auto-jog", action="store_true", help="Drive forward automatically (headless tests).")
parser.add_argument("--test", action="store_true", help="Load the stage, drive one second and exit.")
parser.add_argument("--x", type=float, default=0.0, help="Spawn X (m).")
parser.add_argument("--y", type=float, default=0.0, help="Spawn Y (m).")
parser.add_argument("--yaw", type=float, default=0.0, help="Spawn yaw (rad).")
parser.add_argument("--lidar-topic", default="/isaac/lidar_points")
parser.add_argument("--imu-topic", default="/isaac/imu")
parser.add_argument("--lidar-frame", default="lidar_link")
parser.add_argument("--imu-frame", default="imu_link")
parser.add_argument("--base-frame", default="base_link", help="Robot frame used by elevation_mapping_cupy.")
parser.add_argument("--body-frame", default="body", help="FAST-LIO body_frame; must equal lio.yaml body_frame.")
parser.add_argument(
    "--lidar-motion-compensation", choices=("noncompensated", "compensated"), default="noncompensated",
)
args, _ = parser.parse_known_args()

simulation_app = SimulationApp(
    {"headless": args.headless, "enable_motion_bvh": True, "extra_args": KIT_EXTRA_ARGS}
)

import carb
import isaacsim.core.experimental.utils.app as app_utils
import omni
import omni.appwindow
import omni.graph.core as og
import usdrt
from isaacsim.core.experimental.utils.stage import is_stage_loading
from isaacsim.core.rendering_manager import ViewportManager
from isaacsim.core.simulation_manager import SimulationManager
from isaacsim.robot.experimental.wheeled_robots.controllers import DifferentialController
from isaacsim.robot.experimental.wheeled_robots.robots import WheeledRobot
from isaacsim.sensors.experimental.physics import IMU
from isaacsim.sensors.experimental.rtx import Lidar, LidarSensor
from isaacsim.storage.native import get_assets_root_path, is_file
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, Usd, UsdGeom, UsdPhysics

app_utils.enable_extension("isaacsim.ros2.bridge")
simulation_app.update()

pressed_keys = set()


def on_keyboard_event(event, *_) -> bool:
    key = (event.input if isinstance(event.input, str) else event.input.name).upper()
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        pressed_keys.add(key)
    elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
        pressed_keys.discard(key)
    return True


def get_jog_command() -> tuple[float, float]:
    if "SPACE" in pressed_keys:
        return 0.0, 0.0
    forward = int(bool(pressed_keys & {"W", "UP"}))
    backward = int(bool(pressed_keys & {"S", "DOWN"}))
    left = int(bool(pressed_keys & {"A", "LEFT"}))
    right = int(bool(pressed_keys & {"D", "RIGHT"}))
    return (forward - backward) * LINEAR_JOG_SPEED, (left - right) * ANGULAR_JOG_SPEED


def rotate_vector_by_quaternion(vector, quaternion) -> list[float]:
    vx, vy, vz = (float(v) for v in vector)
    qw, qx, qy, qz = (float(v) for v in quaternion)
    tx = 2.0 * (qy * vz - qz * vy)
    ty = 2.0 * (qz * vx - qx * vz)
    tz = 2.0 * (qx * vy - qy * vx)
    return [
        vx + qw * tx + qy * tz - qz * ty,
        vy + qw * ty + qz * tx - qx * tz,
        vz + qw * tz + qx * ty - qy * tx,
    ]


def update_follow_camera(robot, camera_path: str) -> None:
    positions, orientations = robot.get_world_poses()
    position = positions.numpy()[0]
    forward = rotate_vector_by_quaternion([1.0, 0.0, 0.0], orientations.numpy()[0])
    eye = [
        float(position[0] - forward[0] * FOLLOW_CAMERA_DISTANCE),
        float(position[1] - forward[1] * FOLLOW_CAMERA_DISTANCE),
        float(position[2] + FOLLOW_CAMERA_HEIGHT),
    ]
    target = [
        float(position[0] + forward[0] * FOLLOW_CAMERA_LOOK_AHEAD),
        float(position[1] + forward[1] * FOLLOW_CAMERA_LOOK_AHEAD),
        float(position[2] + FOLLOW_CAMERA_TARGET_HEIGHT),
    ]
    ViewportManager.set_camera_view(camera_path, eye=eye, target=target)


def sensor_mount(stage) -> Gf.Matrix4d:
    """Pose of the LiDAR (and the co-located IMU) in base_link (= chassis_link)."""
    time_code = Usd.TimeCode.Default()
    lidar_world = UsdGeom.Xformable(stage.GetPrimAtPath(LIDAR_PATH)).ComputeLocalToWorldTransform(time_code)
    chassis_world = UsdGeom.Xformable(stage.GetPrimAtPath(CHASSIS_PATH)).ComputeLocalToWorldTransform(time_code)
    return lidar_world * chassis_world.GetInverse()


def create_ros2_publishers(stage) -> None:
    keys = og.Controller.Keys
    mount = sensor_mount(stage)
    # FAST-LIO publishes map -> body; body -> base_link is the inverse of the sensor mount.
    transforms = [
        ("BodyToBase", args.body_frame, args.base_frame, mount.GetInverse()),
        ("BaseToLidar", args.base_frame, args.lidar_frame, mount),
        ("BaseToImu", args.base_frame, args.imu_frame, mount),
    ]
    tf_nodes, tf_connections, tf_values = [], [], []
    for name, parent, child, matrix in transforms:
        quat = Gf.Quatd(matrix.RemoveScaleShear().ExtractRotationQuat())
        imag = quat.GetImaginary()
        tf_nodes.append((name, "isaacsim.ros2.bridge.ROS2PublishRawTransformTree"))
        tf_connections += [
            ("OnPlaybackTick.outputs:tick", f"{name}.inputs:execIn"),
            ("ReadSimTime.outputs:simulationTime", f"{name}.inputs:timeStamp"),
        ]
        tf_values += [
            (f"{name}.inputs:parentFrameId", parent),
            (f"{name}.inputs:childFrameId", child),
            (f"{name}.inputs:topicName", "tf_static"),
            (f"{name}.inputs:staticPublisher", True),
            (f"{name}.inputs:translation", list(matrix.ExtractTranslation())),
            (f"{name}.inputs:rotation", [imag[0], imag[1], imag[2], quat.GetReal()]),
        ]
        print(f"Static TF {parent} -> {child}: t={list(matrix.ExtractTranslation())}")

    graph_path = "/World/FASTLIO_ROS2"
    og.Controller.edit(
        {"graph_path": graph_path, "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("OnPlaybackTick", "omni.graph.action.OnPlaybackTick"),
                ("ReadSimTime", "isaacsim.core.nodes.IsaacReadSimulationTime"),
                ("ReadIMU", "isaacsim.sensors.physics.IsaacReadIMU"),
                ("PublishIMU", "isaacsim.ros2.bridge.ROS2PublishImu"),
                ("PublishClock", "isaacsim.ros2.bridge.ROS2PublishClock"),
                *tf_nodes,
            ],
            keys.CONNECT: [
                ("OnPlaybackTick.outputs:tick", "ReadIMU.inputs:execIn"),
                ("ReadIMU.outputs:execOut", "PublishIMU.inputs:execIn"),
                ("ReadIMU.outputs:orientation", "PublishIMU.inputs:orientation"),
                ("ReadIMU.outputs:angVel", "PublishIMU.inputs:angularVelocity"),
                ("ReadIMU.outputs:linAcc", "PublishIMU.inputs:linearAcceleration"),
                ("ReadIMU.outputs:sensorTime", "PublishIMU.inputs:timeStamp"),
                ("OnPlaybackTick.outputs:tick", "PublishClock.inputs:execIn"),
                ("ReadSimTime.outputs:simulationTime", "PublishClock.inputs:timeStamp"),
                *tf_connections,
            ],
            keys.SET_VALUES: [
                ("PublishIMU.inputs:topicName", args.imu_topic),
                ("PublishIMU.inputs:frameId", args.imu_frame),
                ("PublishClock.inputs:topicName", "/clock"),
                *tf_values,
            ],
        },
    )
    og.Controller.set(
        og.Controller.attribute(f"{graph_path}/ReadIMU.inputs:imuPrim"),
        [usdrt.Sdf.Path(IMU_PATH)],
    )


def spawn_robot(stage, assets_root_path: str):
    usd_path = assets_root_path + CARTER_ASSET_PATH
    if not is_file(usd_path):
        raise FileNotFoundError(f"Nova Carter asset was not found: {usd_path}")
    robot = WheeledRobot(
        paths=CARTER_PRIM_PATH,
        wheel_dof_names=WHEEL_JOINTS,
        usd_path=usd_path,
        positions=[args.x, args.y, SPAWN_HEIGHT],
        orientations=[math.cos(args.yaw / 2.0), 0.0, 0.0, math.sin(args.yaw / 2.0)],
    )
    controller = DifferentialController(wheel_radius=WHEEL_RADIUS, wheel_base=WHEEL_BASE)

    if not stage.GetPrimAtPath(LIDAR_PATH).IsValid():
        raise RuntimeError(f"LiDAR prim was not found: {LIDAR_PATH}")
    lidar = Lidar(
        LIDAR_PATH,
        accumulate_outputs=None,
        aux_output_level="BASIC",
        attributes={
            "omni:sensor:Core:outputMotionCompensationState": args.lidar_motion_compensation.upper(),
        },
    )
    lidar_sensor = LidarSensor(lidar, annotators=[])
    lidar_sensor.attach_writer(
        "RtxLidarROS2PublishPointCloud",
        topicName=args.lidar_topic,
        frameId=args.lidar_frame,
        outputIntensity=True,
        outputTimestamp=True,
        outputEmitterId=False,
        outputChannelId=True,
    )
    IMU.create(
        IMU_PATH,
        translations=[[0.0, 0.0, 0.0]],
        linear_acceleration_filter_size=3,
        angular_velocity_filter_size=3,
        orientation_filter_size=3,
    )
    return robot, controller, lidar_sensor


exit_code = 0
input_interface = keyboard = keyboard_subscription = None


def enable_stair_collision(stage) -> None:
    """Replace the invisible ground plane with real colliders: exact meshes for stairs, convex hulls for floors."""
    plane = stage.GetPrimAtPath(GROUND_COLLISION_PLANE_PATH)
    if plane.IsValid():
        UsdPhysics.CollisionAPI(plane).GetCollisionEnabledAttr().Set(False)
        print(f"Disabled ground collision plane: {GROUND_COLLISION_PLANE_PATH}")

    roots = [
        p for p in stage.GetPseudoRoot().GetChild("Root").GetChildren()
        if any(k in p.GetName().lower() for k in WALKABLE_NAME_KEYWORDS)
    ]
    count = 0
    for root in roots:
        # Instance proxies cannot author collision APIs, so make the subtree local first.
        for prim in Usd.PrimRange(root):
            if prim.IsInstance():
                prim.SetInstanceable(False)
        for prim in Usd.PrimRange(root):
            if prim.IsA(UsdGeom.Mesh):
                UsdPhysics.CollisionAPI.Apply(prim).CreateCollisionEnabledAttr(True)
                # Flat floors need a convex hull: wheels get no traction on zero-thickness triangle meshes.
                approximation = "convexHull" if "floor" in root.GetName().lower() else "none"
                UsdPhysics.MeshCollisionAPI.Apply(prim).CreateApproximationAttr(approximation)
                count += 1
    print(f"Enabled mesh colliders on {count} meshes under {len(roots)} walkable prims")


try:
    assets_root_path = get_assets_root_path()
    if assets_root_path is None:
        raise RuntimeError("Could not find the Isaac Sim assets root.")
    office_usd_path = assets_root_path + OFFICE_ASSET_PATH
    if not is_file(office_usd_path):
        raise FileNotFoundError(f"Office environment asset was not found: {office_usd_path}")

    omni.usd.get_context().open_stage(office_usd_path)
    simulation_app.update()
    simulation_app.update()
    while is_stage_loading():
        simulation_app.update()

    stage = omni.usd.get_context().get_stage()
    buildings_prim = stage.GetPrimAtPath(SURROUNDING_BUILDINGS_PRIM_PATH)
    if buildings_prim.IsValid():
        UsdGeom.Imageable(buildings_prim).MakeInvisible()

    if not USE_PHYSICS_PLANE:
        enable_stair_collision(stage)

    robot, controller, lidar_sensor = spawn_robot(stage, assets_root_path)
    create_ros2_publishers(stage)
    simulation_app.update()
    while is_stage_loading():
        simulation_app.update()

    def drive(linear: float, angular: float) -> None:
        robot.apply_wheel_actions(controller.forward(command=[linear, angular]))

    SimulationManager.setup_simulation(dt=1.0 / 60.0, device="cpu")
    physics_scenes = SimulationManager.get_physics_scenes()
    if not physics_scenes:
        raise RuntimeError("Isaac Sim did not create a physics scene")
    physics_scenes[0].set_enabled_gpu_dynamics(False)

    if not args.headless:
        input_interface = carb.input.acquire_input_interface()
        keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        keyboard_subscription = input_interface.subscribe_to_keyboard_events(keyboard, on_keyboard_event)
        print("Jog controls: W/S or Up/Down = forward/backward, A/D or Left/Right = turn, Space = stop")

    print(f"Loaded Office environment: {office_usd_path}")
    print(f"ROS 2 LiDAR: {args.lidar_topic} [sensor_msgs/msg/PointCloud2]")
    print(f"ROS 2 IMU: {args.imu_topic} [sensor_msgs/msg/Imu]")

    omni.timeline.get_timeline_interface().play()
    for _ in range(10):
        simulation_app.update()

    follow_camera_path = None
    if not args.headless:
        active_viewport = get_active_viewport()
        if active_viewport is None:
            raise RuntimeError("Could not find the active viewport for the follow camera")
        follow_camera_path = str(active_viewport.camera_path)

    if args.test:
        start = robot.get_world_poses()[0].numpy()[0]
        for _ in range(60):
            drive(0.2, 0.0)
            simulation_app.update()
        drive(0.0, 0.0)
        distance = float(((robot.get_world_poses()[0].numpy()[0] - start) ** 2).sum() ** 0.5)
        if not 0.1 <= distance <= 0.3:
            raise RuntimeError(f"Nova Carter jog test failed; moved {distance:.3f} m instead of ~0.2 m")
        print(f"Nova Carter jog test passed: distance={distance:.3f} m")
    else:
        while simulation_app.is_running():
            drive(*((0.2, 0.15) if args.auto_jog else get_jog_command()))
            if follow_camera_path is not None:
                update_follow_camera(robot, follow_camera_path)
            simulation_app.update()
finally:
    # Kit's fast shutdown can terminate before Python reports a pending exception.
    if sys.exc_info()[0] is not None:
        exit_code = 1
        traceback.print_exc()
        sys.stderr.flush()
    if input_interface is not None and keyboard_subscription is not None:
        input_interface.unsubscribe_to_keyboard_events(keyboard, keyboard_subscription)
    omni.timeline.get_timeline_interface().stop()
    simulation_app.close(exit_code=exit_code)
