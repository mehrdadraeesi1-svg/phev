"""
Proposed-method wrappers (v4) on top of the plant-v3 environment (PHEVEnv_v13).

1. DwellObsWrapper     - appends the normalised dwell timer min(dwell, min_dwell)/min_dwell
                         to the observation (25 -> 26 dims), so the agent can see when a
                         mode change will be refused by the 10-s minimum-dwell rule.
2. VetoPenaltyWrapper  - subtracts w_veto from the reward whenever a requested mode change
                         is refused by the dwell rule (the environment counts these in n_vetoed).
3. RewardCenteringWrapper (training only) - simple reward centering (Naik et al., 2024):
                         the agent receives r - r_bar, with r_bar an exponential running
                         mean of the reward (rate eta).  Removing the constant offset
                         r_bar/(1-gamma) from the value function keeps the critic well
                         conditioned for gamma close to 1, so that the metric-anchored
                         energy term can be optimised with a horizon longer than the
                         storage times that create the discounting bias.
The physics, supervisor, dwell rule, reward terms of the base scenario and the action
mapping are unchanged.
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import PHEVEnv_v13 as E


class DwellObsWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = env.observation_space.shape[0] + 1
        self.observation_space = spaces.Box(0.0, 1.0, shape=(n,), dtype=np.float32)

    def observation(self, obs):
        inner = E._inner(self.env)
        md = max(inner.min_dwell, 1)
        d = min(inner.dwell, md) / md
        return np.append(obs, np.float32(d)).astype(np.float32)


class VetoPenaltyWrapper(gym.Wrapper):
    def __init__(self, env, w_veto=0.05):
        super().__init__(env)
        self.w_veto = float(w_veto)

    def step(self, action):
        inner = E._inner(self.env)
        n0 = inner.n_vetoed
        obs, r, term, trunc, info = self.env.step(action)
        if inner.n_vetoed > n0:
            r -= self.w_veto
            info["veto"] = 1
        return obs, r, term, trunc, info


class RewardCenteringWrapper(gym.Wrapper):
    def __init__(self, env, eta=1e-3, r_bar0=0.0):
        super().__init__(env)
        self.eta, self.r_bar = float(eta), float(r_bar0)

    def step(self, action):
        obs, r, term, trunc, info = self.env.step(action)
        self.r_bar += self.eta * (r - self.r_bar)
        info["r_raw"] = r
        info["r_bar"] = self.r_bar
        return obs, r - self.r_bar, term, trunc, info


def make_env_v4(scenario, cycles, deterministic=False, min_dwell=E.MIN_DWELL, seed=None,
                dwell_obs=True, w_veto=0.05, center_eta=0.0):
    env = E.make_env(scenario, cycles, deterministic=deterministic, min_dwell=min_dwell, seed=None)
    if w_veto > 0:
        env = VetoPenaltyWrapper(env, w_veto)
    if dwell_obs:
        env = DwellObsWrapper(env)
    if center_eta > 0:
        env = RewardCenteringWrapper(env, center_eta)
    if seed is not None:
        env.reset(seed=seed)
    return env
