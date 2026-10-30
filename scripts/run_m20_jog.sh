#!/usr/bin/env bash
# Starts only the m20_jog Qt panel (publishes /cmd_vel) in a one-off ROS container.
#   scripts/run_m20_jog.sh [--build]
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_dir"

export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-10}"
export FASTRTPS_DEFAULT_PROFILES_FILE="$project_dir/docker/fastdds_udp.xml"

if [[ "${1:-}" == "--build" || ! -f ros2_ws/install/m20_jog/share/m20_jog/package.xml ]]; then
  docker compose run --rm ros build
fi

[[ -n "${DISPLAY:-}" ]] && xhost +local:docker >/dev/null 2>&1 || true
exec docker compose run --rm ros ros2 run m20_jog m20_jog
