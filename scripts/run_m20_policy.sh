#!/usr/bin/env bash
# Starts the m20_policy_node (ROS container) and Isaac Sim m20_standalone.py --external on the host.
#   scripts/run_m20_policy.sh --policy /path/to/policy.pt [--policy-device cpu|cuda] [--build] [m20_standalone.py args, e.g. --headless --device cpu]
# The policy is TorchScript exported by `isaaclab play` (logs/rsl_rl/m20_flat/<run>/exported/policy.pt).
# Stops everything on Ctrl+C.
set -eo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd -P)"
cd "$project_dir"

export HOST_UID="$(id -u)" HOST_GID="$(id -g)"
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-10}"
export FASTRTPS_DEFAULT_PROFILES_FILE="$project_dir/docker/fastdds_udp.xml"

build=false
policy=""
policy_device=cpu
sim_args=()
while (($# > 0)); do
  case "$1" in
    --policy) policy="${2:?--policy needs a path}"; shift 2 ;;
    --build) build=true; shift ;;
    --policy-device) policy_device="${2:?--policy-device needs cpu or cuda}"; shift 2 ;;
    *) sim_args+=("$1"); shift ;;
  esac
done

if [[ -z "$policy" || ! -f "$policy" ]]; then
  echo "Usage: $0 --policy /path/to/policy.pt [--policy-device cpu|cuda] [--build] [m20_standalone.py args...]" >&2
  exit 2
fi
policy="$(readlink -f "$policy")"
# the policy directory is mounted read-only at /policies in the container
export POLICY_DIR="$(dirname "$policy")"

if [[ "$build" == true || ! -f ros2_ws/install/m20_policy_node/share/m20_policy_node/package.xml ]]; then
  docker compose run --rm ros build
fi

cleanup() {
  trap - EXIT INT TERM
  [[ -n "${sim_pid:-}" ]] && kill "$sim_pid" 2>/dev/null || true
  docker compose down --remove-orphans >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

ROS_COMMAND=launch \
ROS_LAUNCH_ARGS=$'m20_policy_node\nm20_policy.launch.py\npolicy_path:=/policies/'"$(basename "$policy")"$'\ndevice:='"$policy_device"$'\nuse_sim_time:=true' \
  docker compose up -d --force-recreate ros

docker compose logs -f --no-log-prefix ros &

"$project_dir/scripts/m20_standalone.py" --external "${sim_args[@]}" &
sim_pid=$!
wait "$sim_pid"
