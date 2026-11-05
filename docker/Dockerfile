# ROS 2 Jazzy (elevation_mapping_cupy target) with CUDA for CuPy/torch, plus FAST-LIO2 build deps.
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
ARG ROS_DISTRO=jazzy
ARG USERNAME=ros
ARG USER_UID=1000
ARG USER_GID=1000

ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 ROS_DISTRO=${ROS_DISTRO}
SHELL ["/bin/bash", "-o", "pipefail", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
      locales tzdata curl gnupg2 lsb-release git build-essential cmake \
    && locale-gen en_US.UTF-8 \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
      -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo $UBUNTU_CODENAME) main" \
      > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
      libeigen3-dev \
      libpcl-dev \
      libyaml-cpp-dev \
      python3-colcon-common-extensions \
      python3-numpy \
      python3-opencv \
      python3-pip \
      python3-pyqt5 \
      python3-pytest \
      python3-ruamel.yaml \
      python3-scipy \
      python3-shapely \
      python3-sklearn \
      python3-transforms3d \
      python3-yaml \
      ros-${ROS_DISTRO}-ament-cmake-python \
      ros-${ROS_DISTRO}-cv-bridge \
      ros-${ROS_DISTRO}-grid-map-msgs \
      ros-${ROS_DISTRO}-grid-map-rviz-plugin \
      ros-${ROS_DISTRO}-message-filters \
      ros-${ROS_DISTRO}-nav-msgs \
      ros-${ROS_DISTRO}-pcl-conversions \
      ros-${ROS_DISTRO}-pcl-ros \
      ros-${ROS_DISTRO}-rclcpp \
      ros-${ROS_DISTRO}-ros-base \
      ros-${ROS_DISTRO}-rosbag2-py \
      ros-${ROS_DISTRO}-rosidl-default-generators \
      ros-${ROS_DISTRO}-rviz2 \
      ros-${ROS_DISTRO}-sensor-msgs-py \
      ros-${ROS_DISTRO}-std-srvs \
      ros-${ROS_DISTRO}-tf-transformations \
      ros-${ROS_DISTRO}-tf2-geometry-msgs \
      ros-${ROS_DISTRO}-tf2-ros \
    && rm -rf /var/lib/apt/lists/*

# CUDA 12 wheels: torch (traversability filter) and CuPy (elevation map kernels).
RUN python3 -m pip install --no-cache-dir --break-system-packages \
      --index-url https://download.pytorch.org/whl/cu126 torch torchvision \
    && python3 -m pip install --no-cache-dir --break-system-packages \
      --no-deps "cupy-cuda12x==13.6.0" fastrlock ros2-numpy \
    && python3 -m pip install --no-cache-dir --break-system-packages simple-parsing \
    && python3 -m pip install --no-cache-dir --break-system-packages --no-deps transformations pybase64

COPY ros2_ws/src/Livox-SDK2 /tmp/Livox-SDK2
RUN cmake -S /tmp/Livox-SDK2 -B /tmp/livox-build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CXX_FLAGS="-include cstdint" \
    && cmake --build /tmp/livox-build -j"$(nproc)" \
    && cmake --install /tmp/livox-build \
    && rm -rf /tmp/Livox-SDK2 /tmp/livox-build \
    && ldconfig

COPY ros2_ws/src/Sophus /tmp/Sophus
RUN cmake -S /tmp/Sophus -B /tmp/sophus-build -DCMAKE_BUILD_TYPE=Release \
      -DSOPHUS_USE_BASIC_LOGGING=ON -DBUILD_SOPHUS_TESTS=OFF \
    && cmake --build /tmp/sophus-build -j"$(nproc)" \
    && cmake --install /tmp/sophus-build \
    && rm -rf /tmp/Sophus /tmp/sophus-build

# Match the host user so files written to the mounted project stay user-owned.
RUN if getent passwd "$USER_UID" >/dev/null; then \
      userdel -r "$(getent passwd "$USER_UID" | cut -d: -f1)"; \
    fi \
    && if ! getent group "$USER_GID" >/dev/null; then \
      groupadd --gid "$USER_GID" "$USERNAME"; \
    fi \
    && useradd --uid "$USER_UID" --gid "$USER_GID" --create-home --shell /bin/bash "$USERNAME"

USER $USERNAME
WORKDIR /workspace
