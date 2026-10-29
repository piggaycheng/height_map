#!/usr/bin/env bash
# Builds ros2_ws inside the ROS container. Run it with:
#   docker compose run --rm ros build [extra colcon build arguments]
set -eo pipefail

workspace_dir=/workspace/ros2_ws
livox_driver="$workspace_dir/src/livox_ros_driver2"

if [[ ! -f /.dockerenv ]]; then
  echo "Run this inside the ROS container: docker compose run --rm ros build" >&2
  exit 1
fi
if [[ ! -f "$livox_driver/package_ROS2.xml" ||
      ! -f "$workspace_dir/src/FASTLIO2_ROS2/fastlio2/package.xml" ||
      ! -f "$workspace_dir/src/elevation_mapping_cupy/elevation_mapping_cupy/package.xml" ]]; then
  echo "Submodules are missing. Run: git submodule update --init --recursive" >&2
  exit 1
fi

source /opt/ros/jazzy/setup.bash
set -u

# The Livox driver ships ROS 1 and ROS 2 manifests; use the ROS 2 one temporarily.
cleanup() {
  rm -rf "$livox_driver/launch"
  rm -f "$livox_driver/package.xml"
}
trap cleanup EXIT

cp "$livox_driver/package_ROS2.xml" "$livox_driver/package.xml"
rm -rf "$livox_driver/launch"
cp -r "$livox_driver/launch_ROS2" "$livox_driver/launch"

# Rebuild these from scratch to avoid stale CMake state from the temporary manifest.
rm -rf "$workspace_dir/build/livox_ros_driver2" "$workspace_dir/build/interface" \
  "$workspace_dir/build/fastlio2"

cd "$workspace_dir"
colcon build \
  --packages-up-to fastlio2 isaac_fastlio_adapter elevation_mapping_cupy m20_policy_node \
  "$@" \
  --cmake-args \
  -DROS_EDITION=ROS2 \
  -DDISTRO_ROS=jazzy
