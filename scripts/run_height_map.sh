#!/usr/bin/env bash
# Starts the ROS container (FAST-LIO2 + elevation_mapping_cupy), RViz2 with the elevation map,
# and Isaac Sim on the host. Drive with W/A/S/D or the arrow keys in the Isaac Sim window.
#   scripts/run_height_map.sh [--build] [--no-rviz] [standalone.py args...]
# Stops everything on Ctrl+C.
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_dir"

export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-10}"
export FASTRTPS_DEFAULT_PROFILES_FILE="$project_dir/docker/fastdds_udp.xml"

build=false
rviz=true
sim_args=()
for arg in "$@"; do
  case "$arg" in
    --build) build=true ;;
    --no-rviz) rviz=false ;;
    *) sim_args+=("$arg") ;;
  esac
done

if [[ "$build" == true || ! -f ros2_ws/install/setup.bash ]]; then
  docker compose run --rm ros build
fi

if [[ "$rviz" == true && -n "${DISPLAY:-}" ]]; then
  xhost +local:docker >/dev/null 2>&1 || true
fi

sim_pid=""
cleanup() {
  trap - EXIT INT TERM
  [[ -n "$sim_pid" ]] && kill "$sim_pid" 2>/dev/null || true
  docker compose down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

ROS_COMMAND=launch ROS_LAUNCH_ARGS=$'isaac_fastlio_adapter\nheight_map.launch.py' \
  docker compose up -d --force-recreate ros

if [[ "$rviz" == true ]]; then
  docker compose exec -d ros /workspace/docker/entrypoint.sh \
    rviz2 -d /workspace/config/height_map.rviz --ros-args -p use_sim_time:=true
fi

docker compose logs -f --no-log-prefix ros &

"$project_dir/scripts/standalone.py" "${sim_args[@]}" &
sim_pid=$!
wait "$sim_pid"
