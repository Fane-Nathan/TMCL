from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import Dataset

from .synthetic import Dynamics, SyntheticLaneEnv, expert_action


@dataclass
class Episode:
    tokens: np.ndarray
    actions: np.ndarray


def make_episode(
    *,
    horizon: int,
    dt: float,
    steering_gain: float,
    action_noise_std: float,
    latency_steps: int,
    seed: int,
) -> Episode:
    dynamics = Dynamics(
        steering_gain=steering_gain,
        action_noise_std=action_noise_std,
        latency_steps=latency_steps,
    )
    env = SyntheticLaneEnv(dynamics=dynamics, dt=dt, seed=seed)
    state = env.reset()
    prev_action = 0.0
    prev_reward = 0.0
    prev_done = 0.0

    tokens: list[np.ndarray] = []
    actions: list[np.ndarray] = []

    for _ in range(horizon):
        token = np.concatenate(
            [state, np.asarray([prev_action, prev_reward, prev_done], dtype=np.float32)]
        ).astype(np.float32)
        action = expert_action(state, dynamics)
        tokens.append(token)
        actions.append(np.asarray([action], dtype=np.float32))

        state, reward, done, _ = env.step(action)
        prev_action = action
        prev_reward = reward
        prev_done = float(done)
        if done:
            state = env.reset()
            prev_action = prev_reward = prev_done = 0.0

    return Episode(np.stack(tokens), np.stack(actions))


class SyntheticSequenceDataset(Dataset):
    """On-the-fly diverse episode windows for behavior pretraining."""

    def __init__(
        self,
        *,
        size: int,
        sequence_length: int,
        dt: float,
        steering_gain_range: tuple[float, float],
        action_noise_std_range: tuple[float, float],
        seed: int = 0,
    ):
        self.size = size
        self.sequence_length = sequence_length
        self.dt = dt
        self.steering_gain_range = steering_gain_range
        self.action_noise_std_range = action_noise_std_range
        self.seed = seed

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        rng = np.random.default_rng(self.seed + idx)
        episode = make_episode(
            horizon=self.sequence_length,
            dt=self.dt,
            steering_gain=float(rng.uniform(*self.steering_gain_range)),
            action_noise_std=float(rng.uniform(*self.action_noise_std_range)),
            latency_steps=0,
            seed=self.seed + 100_000 + idx,
        )
        return torch.from_numpy(episode.tokens), torch.from_numpy(episode.actions)
