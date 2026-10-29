"""ROS-independent M20 flat-terrain policy I/O, matching my_locomotion_training "Velocity-Flat-M20".

obs (60) = base lin vel(3) | base ang vel(3) | projected gravity(3) | command(3) | leg pos rel(12)
           | leg vel rel * 0.05 (12) | last action(16) | wheel vel * 0.05 (4) | wheel contact(4)
action (16) = 12 leg position offsets (* 0.25 rad, added to the default pose, clipped +-1.5)
              | 4 wheel velocities (* 20 rad/s, clipped +-60)
Leg order is hipx, hipy, knee, each fl, fr, hl, hr; wheels are fl, fr, hl, hr.
"""

import numpy as np

LEG_NAMES = [f"{leg}_{joint}_joint" for joint in ("hipx", "hipy", "knee") for leg in ("fl", "fr", "hl", "hr")]
WHEEL_NAMES = [f"{leg}_wheel_joint" for leg in ("fl", "fr", "hl", "hr")]
JOINT_NAMES = LEG_NAMES + WHEEL_NAMES
DEFAULT_LEG_POS = np.array([0.0] * 4 + [-0.65, -0.65, 0.65, 0.65] + [1.3, 1.3, -1.3, -1.3], dtype=np.float32)
OBS_SIZE = 60
ACTION_SIZE = 16
LEG_ACTION_SCALE = 0.25
LEG_TARGET_LIMIT = 1.5
WHEEL_ACTION_SCALE = 20.0
WHEEL_TARGET_LIMIT = 60.0
VELOCITY_OBS_SCALE = 0.05


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Express a world-frame vector in the frame described by ``quat_wxyz``."""
    w, x, y, z = quat_wxyz
    q_vec = np.array([-x, -y, -z])
    t = 2.0 * np.cross(q_vec, vec)
    return vec + w * t + np.cross(q_vec, t)


def build_observation(
    base_lin_vel: np.ndarray,
    base_ang_vel: np.ndarray,
    orientation_wxyz: np.ndarray,
    command: np.ndarray,
    leg_pos: np.ndarray,
    leg_vel: np.ndarray,
    wheel_vel: np.ndarray,
    last_action: np.ndarray,
    wheel_contact: np.ndarray,
) -> np.ndarray:
    """Assemble the 60-value observation; base velocities must already be in the base frame."""
    gravity = quat_rotate_inverse(np.asarray(orientation_wxyz, dtype=np.float64), np.array([0.0, 0.0, -1.0]))
    obs = np.concatenate(
        [
            base_lin_vel,
            base_ang_vel,
            gravity,
            command,
            leg_pos - DEFAULT_LEG_POS,
            leg_vel * VELOCITY_OBS_SCALE,
            last_action,
            wheel_vel * VELOCITY_OBS_SCALE,
            wheel_contact,
        ]
    ).astype(np.float32)
    if obs.shape != (OBS_SIZE,):
        raise ValueError(f"Observation has shape {obs.shape}, expected ({OBS_SIZE},)")
    return obs


def action_to_targets(action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map a raw policy action to leg position targets (rad) and wheel velocity targets (rad/s)."""
    leg_targets = np.clip(DEFAULT_LEG_POS + LEG_ACTION_SCALE * action[:12], -LEG_TARGET_LIMIT, LEG_TARGET_LIMIT)
    wheel_targets = np.clip(WHEEL_ACTION_SCALE * action[12:], -WHEEL_TARGET_LIMIT, WHEEL_TARGET_LIMIT)
    return leg_targets.astype(np.float32), wheel_targets.astype(np.float32)


class M20Policy:
    """TorchScript policy exported by `isaaclab play` (observation normalization is included)."""

    def __init__(self, path: str, device: str = "cpu") -> None:
        import torch

        self._torch = torch
        self._device = device
        self._model = torch.jit.load(path, map_location=device).eval()
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float32)

    def reset(self) -> None:
        self.last_action = np.zeros(ACTION_SIZE, dtype=np.float32)

    def infer(self, obs: np.ndarray) -> np.ndarray:
        with self._torch.inference_mode():
            tensor = self._torch.from_numpy(obs).to(self._device).unsqueeze(0)
            action = self._model(tensor).squeeze(0).cpu().numpy().astype(np.float32)
        if action.shape != (ACTION_SIZE,):
            raise ValueError(f"Policy returned shape {action.shape}, expected ({ACTION_SIZE},)")
        self.last_action = action
        return action
