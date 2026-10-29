import numpy as np

from m20_policy_node import m20_policy as m20


def test_default_pose_obs_is_zero_relative():
    obs = m20.build_observation(
        np.zeros(3), np.zeros(3), np.array([1.0, 0, 0, 0]), np.array([0.5, 0, 0.1]),
        m20.DEFAULT_LEG_POS, np.zeros(12), np.zeros(4), np.zeros(16), np.ones(4),
    )
    assert obs.shape == (m20.OBS_SIZE,)
    np.testing.assert_allclose(obs[6:9], [0, 0, -1])  # upright: gravity along -z of the base
    np.testing.assert_allclose(obs[9:12], [0.5, 0, 0.1])
    np.testing.assert_allclose(obs[12:24], 0)
    np.testing.assert_allclose(obs[-4:], 1)


def test_gravity_follows_base_tilt():
    # 90 deg roll about x: in the base frame gravity points along -y
    s = np.sqrt(0.5)
    obs = m20.build_observation(
        np.zeros(3), np.zeros(3), np.array([s, s, 0, 0]), np.zeros(3),
        m20.DEFAULT_LEG_POS, np.zeros(12), np.zeros(4), np.zeros(16), np.ones(4),
    )
    np.testing.assert_allclose(obs[6:9], [0, -1, 0], atol=1e-6)


def test_action_targets_scaled_and_clipped():
    leg, wheel = m20.action_to_targets(np.concatenate([np.full(12, 100.0), [1, -1, 100, -100]]))
    np.testing.assert_allclose(leg, 1.5)
    np.testing.assert_allclose(wheel, [20, -20, 60, -60])
    leg, _ = m20.action_to_targets(np.zeros(16))
    np.testing.assert_allclose(leg, m20.DEFAULT_LEG_POS)
