#!/usr/bin/env bash
# Container entrypoint: sources ROS 2 and the workspace, then runs a command.
#   build        Build ros2_ws (extra arguments are passed to colcon build).
#   launch       Run `ros2 launch` with newline-separated $ROS_LAUNCH_ARGS.
#   idle         Keep the container running for `docker compose exec`.
#   <other>      Execute the command in the sourced environment.
set -eo pipefail

workspace_dir=/workspace/ros2_ws

source /opt/ros/jazzy/setup.bash
if [[ -f "$workspace_dir/install/setup.bash" ]]; then
  source "$workspace_dir/install/setup.bash"
fi

command="${1:-idle}"
if (($# > 0)); then
  shift
fi

case "$command" in
  build)
    exec /workspace/docker/build_workspace.sh "$@"
    ;;
  idle)
    exec sleep infinity
    ;;
  launch)
    if [[ ! -f "$workspace_dir/install/setup.bash" ]]; then
      echo "ROS workspace is not built. Run: docker compose run --rm ros build" >&2
      exit 1
    fi
    mapfile -t launch_args < <(printf '%s' "${ROS_LAUNCH_ARGS:-}")
    if ((${#launch_args[@]} == 0)); then
      echo "ROS_LAUNCH_ARGS must contain the launch package and file." >&2
      exit 2
    fi
    exec ros2 launch "${launch_args[@]}"
    ;;
  *)
    exec "$command" "$@"
    ;;
esac
